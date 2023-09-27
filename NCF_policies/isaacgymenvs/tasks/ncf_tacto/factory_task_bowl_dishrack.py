import os
import math
import torch
import hydra

from isaacgym import gymapi, gymtorch, gymutil
from isaacgymenvs.utils import torch_jit_utils as torch_utils
from isaacgymenvs.tasks.ncf_tacto.factory_task_mug_cupholder import (
    NCFTaskCupholder,
)
from isaacgymenvs.tasks.ncf_tacto.factory_schema_config_env import (
    FactorySchemaConfigEnv,
)
from isaacgymenvs.tasks.ncf_tacto.factory_schema_config_base import (
    FactorySchemaConfigBase,
)
import isaacgymenvs.tasks.ncf_tacto.factory_control as fc
from isaacgymenvs.utils import torch_jit_utils

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d


class ExtrinsicContact:
    def __init__(
        self,
        num_envs,
        mesh_obj,
        mesh_dishrack,
        obj_scale,
        dishrack_scale,
        dishrack_pos,
        pointcloud_obj,
        world_ref_pointcloud,
    ) -> None:
        self.num_envs = num_envs
        self.object_trimesh = trimesh.load(mesh_obj)
        self.object_trimesh = self.object_trimesh.apply_scale(obj_scale)
        T = np.eye(4)
        T[0:3, 0:3] = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
        self.object_trimesh = self.object_trimesh.apply_transform(T)

        self.dishrack_trimesh = trimesh.load(mesh_dishrack)
        # self.dishrack_trimesh = self.dishrack_trimesh.apply_scale(dishrack_scale)
        T = np.eye(4)
        T[0:3, -1] = dishrack_pos
        T[0:3, 0:3] = R.from_euler("xyz", [0.0, 0.0, 3.14], degrees=False).as_matrix()
        self.dishrack_trimesh.apply_transform(T)

        self.dishrack = o3d.t.geometry.RaycastingScene()
        self.dishrack.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(self.dishrack_trimesh.as_open3d)
        )

        # pc = np.load(pointcloud_obj)
        # pc = trimesh.points.PointCloud(pc)
        # T = np.eye(4)
        # T[0:3, 0:3] = R.from_euler("xyz", [0.0, 0.0, -90.0], degrees=True).as_matrix()
        # pc = pc.apply_transform(T)
        # self.pointcloud_obj = pc.vertices
        self.pointcloud_obj = np.load(pointcloud_obj)

        self.n_points = self.pointcloud_obj.shape[0]

        self.gt_extrinsic_contact = torch.zeros((num_envs, self.n_points))
        self.world_ref_pointcloud = world_ref_pointcloud

    def _xyzquat_to_tf_numpy(self, position_quat: np.ndarray) -> np.ndarray:
        """
        convert [x, y, z, qx, qy, qz, qw] to 4 x 4 transformation matrices
        """
        # try:
        position_quat = np.atleast_2d(position_quat)  # (N, 7)
        N = position_quat.shape[0]
        T = np.zeros((N, 4, 4))
        T[:, 0:3, 0:3] = R.from_quat(position_quat[:, 3:]).as_matrix()
        T[:, :3, 3] = position_quat[:, :3]
        T[:, 3, 3] = 1
        # except ValueError:
        #     print("Zero quat error!")
        return T.squeeze()

    def reset_extrinsic_contact(self, env_ids):
        self.gt_extrinsic_contact[env_ids] = torch.zeros((len(env_ids), self.n_points))
        self.step = 0

    def get_extrinsic_contact(self, obj_pos, obj_quat):
        object_poses = torch.cat((obj_pos, obj_quat), dim=1)
        object_poses = self._xyzquat_to_tf_numpy(object_poses.cpu().numpy())

        coords = np.zeros((self.num_envs, self.n_points, 3))
        coords_ref = np.zeros((self.num_envs, self.n_points, 3))
        for i in range(self.num_envs):
            coords_ref[i] = self.pointcloud_obj.copy()
            object_pc_i = trimesh.points.PointCloud(self.pointcloud_obj.copy())
            object_pc_i.apply_transform(object_poses[i])
            coords[i] = np.array(object_pc_i.vertices)

        d = self.dishrack.compute_distance(
            o3d.core.Tensor.from_numpy(coords.astype(np.float32))
        ).numpy()

        c = 0.008
        d = d.flatten()
        idx_2 = np.where(d > c)[0]
        d[idx_2] = c
        d = np.clip(d, 0.0, c)

        d = 1.0 - d / c
        d = np.clip(d, 0.0, 1.0)
        d[d > 0.1] = 1.0
        d = d.reshape((self.num_envs, self.n_points))

        self.gt_extrinsic_contact = torch.tensor(d, dtype=torch.float32)
        coords = torch.tensor(coords, dtype=torch.float32)
        coords_ref = torch.tensor(coords_ref, dtype=torch.float32)

        pointcloud = coords if self.world_ref_pointcloud else coords_ref

        # ncf_mask = torch.where(self.gt_extrinsic_contact > 0.1, 1.0, 0.0)
        # ncf_mask = ncf_mask.repeat(3, 1, 1).permute(1, 2, 0)
        # point_cloud = coords * ncf_mask
        # point_cloud = coords_ref * ncf_mask

        return self.gt_extrinsic_contact, pointcloud


class NCFTaskDishrack(NCFTaskCupholder):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        NCFTaskCupholder.__init__(
            self,
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        current_file = os.path.abspath(__file__)
        self.root_dir = os.path.abspath(
            os.path.join(current_file, "..", "..", "..", "..", "..")
        )

        self.success_cases = 0
        self.failure_cases = 0
        self.result_buf = torch.zeros(0)
        self.num_stats = 1000

        # (
        #     self.ini_ctrl_target_fingertip_midpoint_pos,
        #     self.ini_ctrl_target_fingertip_midpoint_quat,
        # ) = self.load_initial_ee_poses()

        if self.cfg_task.rl.compute_contact_gt:
            dishrack_pos = [
                self.cfg_task.randomize.bolt_pos_xy_initial[0],
                self.cfg_task.randomize.bolt_pos_xy_initial[1],
                self.cfg_base.env.table_height,
            ]
            # re-initialize, dishrack scale should be 1.0
            self.extrinsic_contact_gt = ExtrinsicContact(
                mesh_obj=os.path.join(
                    self.root_dir, self.cfg_task.ncf.path_mesh_object
                ),
                mesh_dishrack=os.path.join(
                    self.root_dir, self.cfg_task.ncf.path_mesh_dishrack
                ),
                obj_scale=1.0,
                dishrack_scale=1.0,
                dishrack_pos=dishrack_pos,
                pointcloud_obj=os.path.join(
                    self.root_dir, self.cfg_task.ncf.path_pointcloud_object
                ),
                num_envs=self.num_envs,
                world_ref_pointcloud=self.cfg_task.rl.world_ref_pointcloud,
            )

    def _get_base_yaml_params(self):
        super()._get_base_yaml_params()
        self.cfg_base.env.franka_friction = 10
        self.cfg_env.env.nut_bolt_friction = 10
        self.cfg_env.env.nut_bolt_friction = 10

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_env", node=FactorySchemaConfigEnv)

        config_path = (
            "task/NCFEnvNutBolt.yaml"  # relative to Hydra search path (cfg dir)
        )
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env["task"]  # strip superfluous nesting

        asset_info_path = "../../assets/factory/yaml/factory_asset_info_nut_bolt.yaml"
        self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        # TODO: fix this hardcode and remove previous hardcode hydra code
        self.cfg_env.env["desired_subassemblies"] = ["bowl_dishrack"]
        # strip superfluous nesting
        self.asset_info_nut_bolt = self.asset_info_nut_bolt[""][""][""][""][""][""]
        self.asset_info_nut_bolt = self.asset_info_nut_bolt["assets"]["factory"]["yaml"]

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        # Compute pos of keypoints on gripper, nut, and bolt in world frame
        # hardcoded dishrack target position
        bolt_tip_pos_local = self.bolt_tip_pos_local.clone()
        bolt_tip_pos_local[:, 0] -= 0.12
        bolt_tip_pos_local[:, 1] += 0.05
        bolt_tip_pos_local[:, 2] += 0.01
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_bolt[:, idx] = torch_jit_utils.tf_combine(
                self.bolt_quat,
                self.bolt_pos,
                self.identity_quat,
                (keypoint_offset + bolt_tip_pos_local),
            )[1]

            self.keypoints_nut[:, idx] = torch_jit_utils.tf_combine(
                self.nut_quat,
                self.nut_pos,
                self.identity_quat,
                self.keypoint_offsets_nut[:, idx, :],
            )[1]

    def reset_idx(self, env_ids):
        """Reset specified environments."""
        self._reset_franka(env_ids)
        self._reset_object(env_ids)

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        # for i in range(10):
        #     self.gym.simulate(self.sim)
        #     self.render()

        # self.gym.simulate(self.sim)
        # while True:
        #     # self.gym.simulate(self.sim)
        #     self.render()

        # Close gripper onto nut
        # print('start gg')
        # # self.disable_gravity()  # to prevent nut from falling
        # for _ in range(self.cfg_task.env.num_gripper_close_sim_steps * 2):
        #     if self.device == "cpu":
        #         self.gym.fetch_results(self.sim, True)
        #     self.ctrl_target_dof_pos[env_ids, 7:9] = 0.0
        #     delta_hand_pose = torch.zeros(
        #         (self.num_envs, self.cfg_task.env.numActions), device=self.device
        #     )  # no arm motion
        #     self._apply_actions_as_ctrl_targets(
        #         actions=delta_hand_pose, ctrl_target_gripper_dof_pos=0.0, do_scale=False
        #     )
        #     self.gym.simulate(self.sim)
        #     self.render()
        # print('gg')is_nut_close_to_bolt
        # self.enable_gravity(gravity_mag=abs(self.cfg_base.sim.gravity[2]))

        self._randomize_gripper_pose(env_ids, sim_steps=5)

        self._reset_buffers(env_ids)
        if self.cfg_task.rl.log_ncf_data:
            self._init_ncf_logger_tensors(env_ids)
        if self.cfg_task.rl.compute_contact_gt:
            self.extrinsic_contact_gt.reset_extrinsic_contact(env_ids)

    def _get_center_dist(self):
        """Get center distance."""
        # -0.1, 0.05, 0.06
        goals = torch.tensor(
            [[0.0, 0.05, 0.18], [-0.05, 0.05, 0.18], [0.05, 0.05, 0.18]],
            device=self.device,
        )
        center_dist_0 = torch.norm(self.nut_pos - goals[0], p=2, dim=-1)
        center_dist_1 = torch.norm(self.nut_pos - goals[1], p=2, dim=-1)
        center_dist_2 = torch.norm(self.nut_pos - goals[2], p=2, dim=-1)

        center_dist_goals = torch.vstack(
            (center_dist_0, center_dist_1, center_dist_2)
        ).permute(1, 0)

        center_dist = torch.min(center_dist_goals, dim=-1)[0]

        # center_dist = torch.norm(center_dist, p=2, dim=-1)

        return center_dist

    def _check_nut_close_to_bolt(self):
        """Check if nut is close to bolt."""
        # center_dist = self.nut_pos - torch.tensor(
        #     [0.0, 0.065, 0.18], device=self.device
        # )
        # center_dist = torch.norm(center_dist, p=2, dim=-1)
        center_dist = self._get_center_dist()
        is_nut_close_to_bolt = torch.where(
            center_dist < 0.05,
            torch.ones_like(self.progress_buf),
            torch.zeros_like(self.progress_buf),
        )
        return is_nut_close_to_bolt

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        self._show_debug_viz()

        keypoint_reward = -self._get_center_dist()
        # keypoint_reward = -torch.exp(keypoint_reward * 15)

        # compute error in orientation
        # -0.0075,  0.0332, -0.6507,  0.7586
        # ref_nut_quat = torch.tensor(
        #     [-0.0075, 0.0332, -0.6507, 0.7586], device=self.device
        # ).repeat(len(self.nut_quat), 1)
        # nut_quat_penalty = torch.norm(self.nut_quat - ref_nut_quat, p=2, dim=-1)
        # nut_quat_penalty[nut_quat_penalty < 0.1] = 0.0
        # is_mug_oriented = nut_quat_penalty < 0.1

        dist_nut_bolt = torch.norm(self.bolt_pos - self.nut_pos, p=2, dim=-1)
        # action_penalty = torch.norm(self.actions, p=2, dim=-1)

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale

        # New change: 0.2 instead of 0.1
        # because the center of the bowl might be already off the center of the gripper
        d = torch.norm(self.finger_midpoint_pos - self.nut_pos, p=2, dim=-1)
        is_not_grasped = d >= 0.1
        self.rew_buf[:] += is_not_grasped.float() * -20
        self.reset_buf[is_not_grasped] = 1

        # new reset condition: avoid bowl laying flat on the dishrack (45 degree)
        # r, p, y = torch_utils.get_euler_xyz(self.nut_quat)
        # flipped = torch.logical_and(p > 0.78, p < 5.50)
        # if flipped.any():
        #     self.reset_buf[:] = 1

        # TODO: confirm the following sentence
        # In this policy, episode length is constant across all envs
        # print(self.rew_buf[0], self.progress_buf[0])
        # is_last_step = self.progress_buf[0] == self.max_episode_length - 1
        # if is_last_step:
        # Check if nut is close enough to bolt
        is_nut_close_to_bolt = torch.logical_and(
            self.progress_buf <= self.max_episode_length - 1,
            self._check_nut_close_to_bolt(),
        )

        success = torch.sum(
            torch.logical_and(
                self.progress_buf == self.max_episode_length - 1,
                self._check_nut_close_to_bolt(),
            )
        )
        self.success_cases += success
        failure = torch.sum(
            torch.logical_and(
                self.reset_buf, self.progress_buf != self.max_episode_length - 1
            )
        )
        failure += torch.sum(
            torch.logical_and(
                self.progress_buf == self.max_episode_length - 1,
                torch.logical_not(self._check_nut_close_to_bolt()),
            )
        )
        self.failure_cases += failure

        cur_result_buf = torch.from_numpy(
            np.array([1]) * success.item() + np.array([0]) * failure.item()
        )
        self.result_buf = torch.cat([self.result_buf, cur_result_buf])
        if self.result_buf.shape[0] >= self.num_stats:
            clip_part = self.result_buf.shape[0] - self.num_stats
            self.result_buf = self.result_buf[clip_part:]

        # is_nut_inserted = is_nut_close_to_bolt * is_mug_oriented
        self.rew_buf[:] += is_nut_close_to_bolt * 10  # self.cfg_task.rl.success_bonus
        self.extras["dish_near_slot"] = torch.mean(is_nut_close_to_bolt.float())
        # self.extras["mug_oriented"] = torch.mean(is_mug_oriented.float())
        self.extras["successes"] = self.success_cases / (
            self.success_cases + self.failure_cases + 1e-5
        )
        self.extras["successes2"] = torch.sum(self.result_buf) / (
            self.result_buf.shape[0] + 1e-5
        )
        # self.extras["successes2"] = torch.mean(is_nut_close_to_bolt.float())
        self.extras["dist_dish_dishrack"] = torch.mean(dist_nut_bolt)
        self.extras["keypoint_reward"] = torch.mean(keypoint_reward.abs())
        # self.extras["action_penalty"] = torch.mean(action_penalty)
        # self.extras["mug_quat_penalty"] = torch.mean(nut_quat_penalty)
        self.extras["steps"] = torch.mean(self.progress_buf.float())
        self.extras["dish_close"] = is_nut_close_to_bolt.float()

        if self.reset_buf.sum() > 0:
            if self.cfg_task.rl.log_ncf_data:
                self._save_ncf_data()

    def _reset_franka(self, env_ids, open_gripper=True):
        """Reset DOF states and DOF targets of Franka."""
        nut_widths_max = torch.ones_like(self.nut_widths_max)  # * 0.02
        nut_widths_max = nut_widths_max[env_ids]
        # new change: hard coded because dish is thin
        nut_widths_max[:] = 0.02
        self.dof_pos[env_ids] = torch.cat(
            (
                torch.tensor(
                    self.cfg_task.randomize.franka_arm_initial_dof_pos,
                    device=self.device,
                ).repeat((len(env_ids), 1)),
                (nut_widths_max * 0.5)
                * 1.1,  # buffer on gripper DOF pos to prevent initial contact
                (nut_widths_max * 0.5) * 1.1,
            ),  # buffer on gripper DOF pos to prevent initial contact
            dim=-1,
        )  # shape = (num_envs, num_dofs)

        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

    def _reset_object(self, env_ids):
        """Reset root states of nut and bolt."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        # Randomize root state of nut within gripper
        self.root_pos[env_ids, self.nut_actor_id_env, 0] = 0.0
        self.root_pos[env_ids, self.nut_actor_id_env, 1] = 0.0
        fingertip_midpoint_pos_reset = 0.58781  # self.fingertip_midpoint_pos at reset
        # mug
        nut_base_pos_local = 0.005
        self.root_pos[env_ids, self.nut_actor_id_env, 2] = (
            fingertip_midpoint_pos_reset - nut_base_pos_local
        )

        # self.root_pos[env_ids, self.nut_actor_id_env, 1] = 0.02

        # nut
        nut_noise_pos_in_gripper = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        nut_noise_pos_in_gripper = nut_noise_pos_in_gripper @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.nut_noise_pos_in_gripper, device=self.device
            )
        )
        nut_noise_pos_in_gripper[env_ids, 0] -= 0.12
        nut_noise_pos_in_gripper[env_ids, 1] = 0.052
        # this is for not centered dish
        # nut_noise_pos_in_gripper[env_ids, 2] -= 0.205
        nut_noise_pos_in_gripper[env_ids, 2] -= 0.195
        self.root_pos[env_ids, self.nut_actor_id_env, :] += nut_noise_pos_in_gripper[
            env_ids
        ]

        nut_rot_euler = torch.tensor(
            [0.0, 0.0, -math.pi / 2], device=self.device
        ).repeat(len(env_ids), 1)
        nut_noise_rot_in_gripper = 2 * (
            torch.rand(len(env_ids), dtype=torch.float32, device=self.device) - 0.5
        )  # [-1, 1]
        nut_noise_rot_in_gripper *= self.cfg_task.randomize.nut_noise_rot_in_gripper
        nut_rot_euler[:, 2] += nut_noise_rot_in_gripper
        nut_rot_quat = torch_utils.quat_from_euler_xyz(
            nut_rot_euler[:, 0], nut_rot_euler[:, 1], nut_rot_euler[:, 2]
        )
        self.root_quat[env_ids, self.nut_actor_id_env] = nut_rot_quat

        # Randomize root state of bolt
        bolt_noise_xy = 2 * (
            torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        bolt_noise_xy = bolt_noise_xy @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.bolt_pos_xy_noise,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.root_pos[env_ids, self.bolt_actor_id_env, 0] = (
            self.cfg_task.randomize.bolt_pos_xy_initial[0] + bolt_noise_xy[env_ids, 0]
        )
        self.root_pos[env_ids, self.bolt_actor_id_env, 1] = (
            self.cfg_task.randomize.bolt_pos_xy_initial[1] + bolt_noise_xy[env_ids, 1]
        )
        self.root_pos[env_ids, self.bolt_actor_id_env, 2] = (
            self.cfg_base.env.table_height + self.cfg_task.randomize.bolt_pos_z_offset
        )
        self.root_quat[env_ids, self.bolt_actor_id_env] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=torch.float32, device=self.device
        ).repeat(len(env_ids), 1)

        self.root_linvel[env_ids, self.bolt_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.bolt_actor_id_env] = 0.0

        nut_bolt_actor_ids_sim = torch.cat(
            (self.nut_actor_ids_sim[env_ids], self.bolt_actor_ids_sim[env_ids]), dim=0
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(nut_bolt_actor_ids_sim),
            len(nut_bolt_actor_ids_sim),
        )

    def _show_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        sphere_pose = gymapi.Transform()
        sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
        sphere_geom = gymutil.WireframeSphereGeometry(
            0.01, 8, 8, sphere_pose, color=(1, 1, 0)
        )
        sphere_geom_white = gymutil.WireframeSphereGeometry(
            0.02, 8, 8, sphere_pose, color=(1, 1, 1)
        )
        for i in range(self.num_envs):
            if i != 0:
                continue
            palm_center_transform = gymapi.Transform()
            palm_center_transform.p = gymapi.Vec3(*self.nut_pos[0])
            palm_center_transform.r = gymapi.Quat(0, 0, 0, 1)
            gymutil.draw_lines(
                sphere_geom_white,
                self.gym,
                self.viewer,
                self.env_ptrs[i],
                palm_center_transform,
            )

        for i in range(self.num_envs):
            if i != 0:
                continue
            palm_center_transform = gymapi.Transform()
            # palm_center_transform.p = gymapi.Vec3(*[-0.1, 0.05, 0.06])
            palm_center_transform.p = gymapi.Vec3(*[0.0, 0.05, 0.18])
            palm_center_transform.r = gymapi.Quat(0, 0, 0, 1)
            gymutil.draw_lines(
                sphere_geom,
                self.gym,
                self.viewer,
                self.env_ptrs[i],
                palm_center_transform,
            )

    def get_initial_ee_poses(self):
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos.clone()

        fingertip_midpoint_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device
            )
        )
        fingertip_midpoint_pos_noise[:, 2] = 0.0
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = (
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        fingertip_midpoint_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device
            )
        )
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2],
        )

        torch.save(
            self.ctrl_target_fingertip_midpoint_pos,
            "ctrl_target_fingertip_midpoint_pos.pt",
        )
        torch.save(
            self.ctrl_target_fingertip_midpoint_quat,
            "ctrl_target_fingertip_midpoint_quat.pt",
        )

    def load_initial_ee_poses(self):
        ctrl_target_fingertip_midpoint_pos = torch.load(
            "ctrl_target_fingertip_midpoint_pos.pt"
        )
        ctrl_target_fingertip_midpoint_quat = torch.load(
            "ctrl_target_fingertip_midpoint_quat.pt"
        )
        return ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat

    def set_initial_ee_pose(self, env_ids, sim_steps=10):
        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ini_ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ini_ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl["jacobian_type"],
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(
                actions=actions, ctrl_target_gripper_dof_pos=0.0, do_scale=False
            )

            self.gym.simulate(self.sim)
            self.render()

        if self.refresh_digits_flag:
            self.refresh_digit_sensors()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
