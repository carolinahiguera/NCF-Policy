# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: Class for nut-bolt place task.

Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskNutBoltPlace
"""

import time
import hydra
import math
import omegaconf
import os
import subprocess
import torch
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R

from isaacgymenvs.utils.torch_jit_utils import quat_apply, to_torch

from isaacgym import gymapi, gymtorch
from isaacgymenvs.utils import torch_jit_utils as torch_utils
import isaacgymenvs.tasks.ncf_tacto.factory_control as fc
from isaacgymenvs.tasks.ncf_tacto.factory_env_nut_bolt import FactoryEnvNutBolt
from isaacgymenvs.tasks.ncf_tacto.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.ncf_tacto.factory_schema_config_task import (
    FactorySchemaConfigTask,
)
from isaacgymenvs.utils import torch_jit_utils
import open3d as o3d

from multiprocessing import Process, Queue, Manager

# from ncf.digit_vae.models.vae import VAE

DEBUG = False


def get_rank():
    return int(os.getenv("LOCAL_RANK", "0"))


def vulkan_device_id_from_cuda_device_id(orig: int) -> int:
    """Map a CUDA device index to a Vulkan one.

    Used to populate the value of `graphic_device_id`, which in IsaacGym is a vulkan
    device ID.

    This prevents a common segfault we get when the Vulkan ID, which is by default 0,
    points to a device that isn't present in CUDA_VISIBLE_DEVICES.
    """
    # Get UUID of the torch device.
    # All of the private methods can be dropped once this PR lands:
    #     https://github.com/pytorch/pytorch/pull/99967
    try:
        cuda_uuid = torch.cuda._raw_device_uuid_nvml()[
            torch.cuda._parse_visible_devices()[orig]
        ]  # type: ignore
        assert cuda_uuid.startswith("GPU-")
        cuda_uuid = cuda_uuid[4:]
    except AttributeError:
        print("detect cuda / vulkan relation can only be done for pytorch 2.0")
        return get_rank()

    try:
        vulkaninfo_lines = subprocess.run(
            ["vulkaninfo"],
            # We unset DISPLAY to avoid this error:
            # https://github.com/KhronosGroup/Vulkan-Tools/issues/370
            env={k: v for k, v in os.environ.items() if k != "DISPLAY"},
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        ).stdout.split("\n")
    except FileNotFoundError:
        print(
            "vulkaninfo was not found; try `apt install vulkan-tools` or `apt install vulkan-utils`."
        )
        return get_rank()

    vulkan_uuids = [
        s.partition("=")[2].strip()
        for s in vulkaninfo_lines
        if s.strip().startswith("deviceUUID")
    ]
    vulkan_uuids = list(dict(zip(vulkan_uuids, vulkan_uuids)).keys())
    vulkan_uuids = [uuid for uuid in vulkan_uuids if not uuid.startswith("0000")]
    out = vulkan_uuids.index(cuda_uuid)
    print(f"Using graphics_device_id={out}", cuda_uuid)
    return out


class ExtrinsicContact:
    def __init__(
        self,
        num_envs,
        mesh_obj,
        mesh_cupholder,
        obj_scale,
        cupholder_scale,
        cupholder_pos,
        pointcloud_obj,
        world_ref_pointcloud,
    ) -> None:
        self.num_envs = num_envs
        self.object_trimesh = trimesh.load(mesh_obj)
        self.object_trimesh = self.object_trimesh.apply_scale(obj_scale)
        T = np.eye(4)
        T[0:3, 0:3] = R.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix()
        self.object_trimesh = self.object_trimesh.apply_transform(T)

        self.cupholder_trimesh = trimesh.load(mesh_cupholder)
        self.cupholder_trimesh = self.cupholder_trimesh.apply_scale(cupholder_scale)
        T = np.eye(4)
        T[0:3, -1] = cupholder_pos
        self.cupholder_trimesh.apply_transform(T)

        self.cupholder = o3d.t.geometry.RaycastingScene()
        self.cupholder.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(self.cupholder_trimesh.as_open3d)
        )

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

        d = self.cupholder.compute_distance(
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

        return self.gt_extrinsic_contact, pointcloud


class NCFTaskCupholder(FactoryEnvNutBolt, FactoryABCTask):
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
        """Initialize instance variables. Initialize environment superclass."""

        graphics_device_id = vulkan_device_id_from_cuda_device_id(graphics_device_id)

        super().__init__(
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

        self.cfg = cfg
        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        if self.viewer is not None:
            self._set_viewer_params()

        if self.cfg_task.rl.log_ncf_data:
            self._init_ncf_logger_tensors()

        # digit sensors modules
        self.refresh_digits_flag = self.cfg_task.sim.sim_digits

        if self.refresh_digits_flag and DEBUG:
            # create plot of 10x2 axes
            self.fig, self.axs = plt.subplots(5, 2, figsize=(7, 20))
            self.fig.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        if self.cfg_task.rl.compute_contact_gt:
            cupholder_pos = [
                self.cfg_task.randomize.bolt_pos_xy_initial[0],
                self.cfg_task.randomize.bolt_pos_xy_initial[1],
                self.cfg_base.env.table_height,
            ]
            self.extrinsic_contact_gt = ExtrinsicContact(
                mesh_obj=os.path.join(
                    self.root_dir, self.cfg_task.ncf.path_mesh_object
                ),
                mesh_cupholder=os.path.join(
                    self.root_dir, self.cfg_task.ncf.path_mesh_cupholder
                ),
                obj_scale=1.0,
                cupholder_scale=0.75,
                cupholder_pos=cupholder_pos,
                pointcloud_obj=os.path.join(
                    self.root_dir, self.cfg_task.ncf.path_pointcloud_object
                ),
                num_envs=self.num_envs,
                world_ref_pointcloud=self.cfg_task.rl.world_ref_pointcloud,
            )

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_task", node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = (
            self.cfg_task.rl.max_episode_length
        )  # required instance var for VecTask

        asset_info_path = "../../assets/factory/yaml/factory_asset_info_nut_bolt.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_env.env["desired_subassemblies"] = ["mug_cupholder"]
        self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        self.asset_info_nut_bolt = self.asset_info_nut_bolt[""][""][""][""][""][""][
            "assets"
        ]["factory"][
            "yaml"
        ]  # strip superfluous nesting

        ppo_path = "train/FactoryTaskNutBoltPlacePPO.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo["train"]  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        # Nut-bolt tensors
        self.nut_base_pos_local = self.bolt_head_heights * torch.tensor(
            [0.0, 0.0, -1.0], device=self.device
        ).repeat((self.num_envs, 1))
        bolt_heights = self.bolt_head_heights + self.bolt_shank_lengths
        self.bolt_tip_pos_local = bolt_heights * torch.tensor(
            [0.0, 0.0, 1.0], device=self.device
        ).repeat((self.num_envs, 1))

        # Keypoint tensors
        self.keypoint_offsets = (
            self._get_keypoint_offsets(self.cfg_task.rl.num_keypoints)
            * self.cfg_task.rl.keypoint_scale
        )
        self.keypoints_nut = torch.zeros(
            (self.num_envs, self.cfg_task.rl.num_keypoints, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.keypoints_bolt = torch.zeros_like(self.keypoints_nut, device=self.device)

        self.identity_quat = (
            torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        self.actions = torch.zeros(
            (self.num_envs, self.cfg_task.env.numActions), device=self.device
        )

        # keypoints_offset_nut
        self.keypoint_offsets_nut = torch.zeros(
            (self.num_envs, self.cfg_task.rl.num_keypoints, 3), device=self.device
        )
        # nut_heights
        for env_id in range(self.num_envs):
            h = self.nut_heights[env_id].item()
            self.keypoint_offsets_nut[env_id, :, 2] = torch.linspace(
                0, -h, self.cfg_task.rl.num_keypoints, device=self.device
            ).__reversed__()

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # Compute pos of keypoints on gripper, nut, and bolt in world frame
        bolt_tip_pos_local = self.bolt_tip_pos_local.clone()
        bolt_tip_pos_local[:, 2] -= 0.010
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

    def _init_ncf_logger_tensors(self):
        """Initialize tensors for logging."""

        self.ncf_logger = []
        for _ in range(self.num_envs):
            self.env_logger = {}
            self.env_logger["left_fingertip_pos"] = torch.zeros((1, 3))
            self.env_logger["left_fingertip_quat"] = torch.zeros((1, 4))
            self.env_logger["right_fingertip_pos"] = torch.zeros((1, 3))
            self.env_logger["right_fingertip_quat"] = torch.zeros((1, 4))
            self.env_logger["fingertip_centered_pos"] = torch.zeros((1, 3))
            self.env_logger["fingertip_centered_quat"] = torch.zeros((1, 4))
            self.env_logger["mug_pos"] = torch.zeros((1, 3))
            self.env_logger["mug_quat"] = torch.zeros((1, 4))
            self.env_logger["cupholder_pos"] = torch.zeros((1, 3))
            self.env_logger["cupholder_quat"] = torch.zeros((1, 4))
            self.ncf_logger.append(self.env_logger)

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]
        self._apply_actions_as_ctrl_targets(
            actions=self.actions, ctrl_target_gripper_dof_pos=0.0, do_scale=True
        )

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        self.refresh_base_tensors()
        self.refresh_env_tensors()

        if self.refresh_digits_flag:
            self.refresh_digit_sensors()

        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

        if self.cfg_task.rl.compute_contact_gt:
            (
                self.gt_extrinsic_contact,
                self.pointclouds_t,
            ) = self.extrinsic_contact_gt.get_extrinsic_contact(
                obj_pos=self.nut_pos, obj_quat=self.nut_quat
            )

        if self.cfg_task.rl.log_ncf_data:
            self._log_ncf_data(env_ids=torch.arange(self.num_envs))

        if self.cfg_task.rl.debug_viz:
            self._show_debug_viz()
            while True:
                self.render()

        if self.refresh_digits_flag and DEBUG:
            self.plot_digits()

    # debug viz
    def _show_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Grab relevant states to visualize
        eef_pos = self.fingertip_midpoint_pos
        eef_rot = self.fingertip_midpoint_quat
        lf_pos = self.left_fingertip_pos
        lf_rot = self.left_fingertip_quat
        rf_pos = self.right_fingertip_pos
        rf_rot = self.right_fingertip_quat
        mug_pos = self.nut_pos
        mug_rot = self.nut_quat
        cupholder_pos = self.bolt_pos
        cupholder_rot = self.bolt_quat

        # Plot visualizations
        for i in range(self.num_envs):
            # for pos, rot in zip((eef_pos, lf_pos, rf_pos, mug_pos, cupholder_pos), (eef_rot, lf_rot, rf_rot, mug_rot, cupholder_rot)):
            for pos, rot in zip((lf_pos, mug_pos), (lf_rot, mug_rot)):
                px = (
                    (
                        pos[i]
                        + quat_apply(
                            rot[i], to_torch([1, 0, 0], device=self.device) * 0.2
                        )
                    )
                    .cpu()
                    .numpy()
                )
                py = (
                    (
                        pos[i]
                        + quat_apply(
                            rot[i], to_torch([0, 1, 0], device=self.device) * 0.2
                        )
                    )
                    .cpu()
                    .numpy()
                )
                pz = (
                    (
                        pos[i]
                        + quat_apply(
                            rot[i], to_torch([0, 0, 1], device=self.device) * 0.2
                        )
                    )
                    .cpu()
                    .numpy()
                )

                p0 = pos[i].cpu().numpy()
                self.gym.add_lines(
                    self.viewer,
                    self.env_ptrs[i],
                    1,
                    [p0[0], p0[1], p0[2], px[0], px[1], px[2]],
                    [0.85, 0.1, 0.1],
                )
                self.gym.add_lines(
                    self.viewer,
                    self.env_ptrs[i],
                    1,
                    [p0[0], p0[1], p0[2], py[0], py[1], py[2]],
                    [0.1, 0.85, 0.1],
                )
                self.gym.add_lines(
                    self.viewer,
                    self.env_ptrs[i],
                    1,
                    [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],
                    [0.1, 0.1, 0.85],
                )

    def compute_observations(self):
        """Compute observations."""
        if self.obs_buf.sum() == 0:
            # initialize obs_buf
            curr_obs = [
                self.fingertip_midpoint_pos,
                self.fingertip_midpoint_quat,
            ]
            curr_obs = torch.cat(curr_obs, dim=-1)
            self.obs_buf = torch.cat((curr_obs, curr_obs, curr_obs), dim=-1)
        else:
            curr_obs = [
                self.fingertip_midpoint_pos,
                self.fingertip_midpoint_quat,
            ]
            curr_obs = torch.cat(curr_obs, dim=-1)
            prev_obs_buf = self.obs_buf[:, 0:14].clone()
            self.obs_buf = torch.cat((curr_obs, prev_obs_buf), dim=-1)

    def compute_observations_old(self):
        """Compute observations."""

        # add noise to bolt position and orientation
        bolt_pos_noise = (2 * torch.randn_like(self.bolt_pos) - 1) * 0.01  # max 1cm
        bolt_pos_noise[:, 2] = 0  # no noise in z direction
        bolt_quat_noise = (2 * torch.randn_like(self.bolt_quat) - 1) * 0.0

        # Shallow copies of tensors
        if self.cfg_task.rl.reduce_obs:
            obs_tensors = [
                self.fingertip_midpoint_pos,  # (3)
                self.fingertip_midpoint_quat,  # (4)
                self.bolt_pos + bolt_pos_noise,  # (3)
                self.bolt_quat + bolt_quat_noise,  # (4)
            ]
        else:
            obs_tensors = [
                self.fingertip_midpoint_pos,  # (3)
                self.fingertip_midpoint_quat,  # (4)
                self.fingertip_midpoint_linvel,  # (3)
                self.fingertip_midpoint_angvel,  # (3)
                self.nut_pos,  # (3)
                self.nut_quat,  # (4)
                self.bolt_pos,  # (3)
                self.bolt_quat,
            ]  # (4)

        if self.cfg_task.rl.add_obs_bolt_tip_pos:
            obs_tensors += [self.bolt_tip_pos_local]

        self.obs_buf = torch.cat(
            obs_tensors, dim=-1
        )  # shape = (num_envs, num_observations)

        return self.obs_buf

    def compute_reward(self):
        """Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.cfg_task.rl.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

    def _update_rew_buf(self):
        """Compute reward at current timestep."""

        if self.cfg_task.rl.use_keypoints_rew:
            keypoint_reward = -self._get_keypoint_dist()
        else:
            keypoint_reward = -self._get_center_dist()

        # compute error in orientation
        ref_nut_quat = torch.tensor(
            [-0.0075, 0.0332, -0.6507, 0.7586], device=self.device
        ).repeat(len(self.nut_quat), 1)
        nut_quat_penalty = torch.norm(self.nut_quat - ref_nut_quat, p=2, dim=-1)
        nut_quat_penalty[nut_quat_penalty < 0.1] = 0.0
        is_mug_oriented = nut_quat_penalty < 0.1

        dist_nut_bolt = torch.norm(self.bolt_pos - self.nut_pos, p=2, dim=-1)

        action_penalty = torch.norm(self.actions, p=2, dim=-1)

        self.rew_buf[:] = (
            keypoint_reward * self.cfg_task.rl.keypoint_reward_scale
            - action_penalty * self.cfg_task.rl.action_penalty_scale
            - nut_quat_penalty * self.cfg_task.rl.orientation_penalty_scale
            - dist_nut_bolt * self.cfg_task.rl.dist_penalty_scale
        )

        # check is object is grasped and reset if not
        d = torch.norm(self.finger_midpoint_pos - self.nut_pos, p=2, dim=-1)
        is_not_grasped = d >= 0.10
        self.reset_buf[is_not_grasped] = 1

        is_nut_close_to_bolt = self._check_nut_close_to_bolt()
        is_nut_inserted = is_nut_close_to_bolt * is_mug_oriented
        tmp = is_nut_inserted * self.progress_buf
        self.time_complete_task[self.time_complete_task == 0] = tmp[
            self.time_complete_task == 0
        ]

        # In this policy, episode length is constant across all envs
        is_last_step = self.progress_buf[0] == self.max_episode_length - 1
        if is_last_step:
            self.time_complete_task[self.time_complete_task == 0] = self.progress_buf[
                self.time_complete_task == 0
            ]
            # Check if nut is close enough to bolt
            is_nut_close_to_bolt = self._check_nut_close_to_bolt()
            is_nut_inserted = is_nut_close_to_bolt * is_mug_oriented
            self.rew_buf[:] += is_nut_inserted * self.cfg_task.rl.success_bonus
            self.extras["mug_close_to_cupholder"] = torch.mean(
                is_nut_close_to_bolt.float()
            )
            self.extras["mug_oriented"] = torch.mean(is_mug_oriented.float())
            self.extras["successes"] = torch.mean(is_nut_inserted.float())
            self.extras["dist_mug_cupholder"] = torch.mean(dist_nut_bolt)
            self.extras["keypoint_reward"] = torch.mean(keypoint_reward.abs())
            self.extras["action_penalty"] = torch.mean(action_penalty)
            self.extras["mug_quat_penalty"] = torch.mean(nut_quat_penalty)
            self.extras["steps"] = torch.mean(self.progress_buf.float())
            self.extras["mean_time_complete_task"] = torch.mean(
                self.time_complete_task.float()
            )
            a = self.time_complete_task.float() * is_nut_inserted
            self.extras["time_success_task"] = a.sum() / torch.where(a > 0)[0].shape[0]

            if True in is_nut_close_to_bolt:
                print("nut is close to bolt")

            if self.cfg_task.rl.log_ncf_data:
                self._save_ncf_data()

    def _log_ncf_data(self, env_ids):
        for env_id in env_ids:
            self.ncf_logger[env_id]["left_fingertip_pos"] = torch.cat(
                (
                    self.ncf_logger[env_id]["left_fingertip_pos"],
                    self.left_fingertip_pos[[env_id]].unsqueeze(0).cpu(),
                ),
                dim=0,
            )
            self.ncf_logger[env_id]["left_fingertip_quat"] = torch.cat(
                (
                    self.ncf_logger[env_id]["left_fingertip_quat"],
                    self.left_fingertip_quat[[env_id]].unsqueeze(0).cpu(),
                ),
                dim=0,
            )
            self.ncf_logger[env_id]["right_fingertip_pos"] = torch.cat(
                (
                    self.ncf_logger[env_id]["right_fingertip_pos"],
                    self.right_fingertip_pos[[env_id]].unsqueeze(0).cpu(),
                ),
                dim=0,
            )
            self.ncf_logger[env_id]["right_fingertip_quat"] = torch.cat(
                (
                    self.ncf_logger[env_id]["right_fingertip_quat"],
                    self.right_fingertip_quat[[env_id]].unsqueeze(0).cpu(),
                ),
                dim=0,
            )
            self.ncf_logger[env_id]["fingertip_centered_pos"] = torch.cat(
                (
                    self.ncf_logger[env_id]["fingertip_centered_pos"],
                    self.fingertip_midpoint_pos[[env_id]].unsqueeze(0).cpu(),
                ),
                dim=0,
            )
            self.ncf_logger[env_id]["fingertip_centered_quat"] = torch.cat(
                (
                    self.ncf_logger[env_id]["fingertip_centered_quat"],
                    self.fingertip_midpoint_quat[[env_id]].unsqueeze(0).cpu(),
                ),
                dim=0,
            )
            self.ncf_logger[env_id]["mug_pos"] = torch.cat(
                (
                    self.ncf_logger[env_id]["mug_pos"],
                    self.nut_pos[[env_id]].unsqueeze(0).cpu(),
                ),
                dim=0,
            )
            self.ncf_logger[env_id]["mug_quat"] = torch.cat(
                (
                    self.ncf_logger[env_id]["mug_quat"],
                    self.nut_quat[[env_id]].unsqueeze(0).cpu(),
                ),
                dim=0,
            )
            self.ncf_logger[env_id]["cupholder_pos"] = torch.cat(
                (
                    self.ncf_logger[env_id]["cupholder_pos"],
                    self.bolt_pos[[env_id]].unsqueeze(0).cpu(),
                ),
                dim=0,
            )
            self.ncf_logger[env_id]["cupholder_quat"] = torch.cat(
                (
                    self.ncf_logger[env_id]["cupholder_quat"],
                    self.bolt_quat[[env_id]].unsqueeze(0).cpu(),
                ),
                dim=0,
            )

    def _save_ncf_data(self):
        path_save_dataset = self.cfg_task.rl.path_save_ncf_data
        torch.save(
            self.ncf_logger,
            f"{path_save_dataset}/ncf_data_{datetime.now():%m-%d %H-%M-%S}.pt",
        )

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        self._reset_franka(env_ids)
        # self._reset_object_debug(env_ids)
        self._reset_object(env_ids)

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.time_complete_task = torch.zeros_like(self.progress_buf)

        # Close gripper onto nut
        self.disable_gravity()  # to prevent nut from falling
        for _ in range(self.cfg_task.env.num_gripper_close_sim_steps):
            self.ctrl_target_dof_pos[env_ids, 7:9] = 0.0
            delta_hand_pose = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )  # no arm motion
            self._apply_actions_as_ctrl_targets(
                actions=delta_hand_pose, ctrl_target_gripper_dof_pos=0.0, do_scale=False
            )
            self.gym.simulate(self.sim)
            self.render()

        self.enable_gravity(gravity_mag=abs(self.cfg_base.sim.gravity[2]))
        self._move_down_arm(
            env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps * 2
        )
        self._randomize_gripper_pose(
            env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
        )
        self._reset_buffers(env_ids)

        if self.cfg_task.rl.log_ncf_data:
            self._init_ncf_logger_tensors()
        if self.cfg_task.rl.compute_contact_gt:
            self.extrinsic_contact_gt.reset_extrinsic_contact(env_ids)

    def _reset_franka(self, env_ids, open_gripper=True):
        """Reset DOF states and DOF targets of Franka."""
        nut_widths_max = torch.ones_like(self.nut_widths_max)  # * 0.02
        nut_widths_max = nut_widths_max[env_ids]
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

    def _reset_object_debug(self, env_ids):
        """Reset root states of nut and bolt."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        # Randomize root state of nut within gripper
        self.root_pos[env_ids, self.nut_actor_id_env, 0] = 0.0
        self.root_pos[env_ids, self.nut_actor_id_env, 1] = 0.0
        self.root_pos[env_ids, self.nut_actor_id_env, 2] = 0.06

        nut_rot_euler = torch.tensor(
            [0.0, 0.0, -math.pi / 2], device=self.device
        ).repeat(len(env_ids), 1)
        nut_rot_quat = torch_utils.quat_from_euler_xyz(
            nut_rot_euler[:, 0], nut_rot_euler[:, 1], nut_rot_euler[:, 2]
        )
        self.root_quat[env_ids, self.nut_actor_id_env] = nut_rot_quat

        # Randomize root state of bolt
        self.root_pos[
            env_ids, self.bolt_actor_id_env, 0
        ] = self.cfg_task.randomize.bolt_pos_xy_initial[0]
        self.root_pos[
            env_ids, self.bolt_actor_id_env, 1
        ] = self.cfg_task.randomize.bolt_pos_xy_initial[1]
        self.root_pos[
            env_ids, self.bolt_actor_id_env, 2
        ] = self.cfg_base.env.table_height
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

    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-0.3, -0.3, 0.3)
        cam_target = gymapi.Vec3(0.0, 0.5, 0.01)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(
        self, actions, ctrl_target_gripper_dof_pos, do_scale
    ):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device)
            )
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + (
            pos_actions
        )

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]  # * 0.5
        if do_scale:
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device)
            )

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(
                    self.num_envs, 1
                ),
            )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(
            rot_actions_quat, self.fingertip_midpoint_quat
        )

        if self.cfg_ctrl["do_force_ctrl"]:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(
                        self.cfg_task.rl.force_action_scale, device=self.device
                    )
                )

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(
                        self.cfg_task.rl.torque_action_scale, device=self.device
                    )
                )

            self.ctrl_target_fingertip_contact_wrench = torch.cat(
                (force_actions, torque_actions), dim=-1
            )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _open_gripper(self, sim_steps=20):
        """Fully open gripper using controller. Called outside RL loop (i.e., after last step of episode)."""

        self._move_gripper_to_dof_pos(gripper_dof_pos=0.1, sim_steps=sim_steps)

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros(
            (self.num_envs, self.cfg_task.env.numActions), device=self.device
        )  # no arm motion
        self._apply_actions_as_ctrl_targets(
            delta_hand_pose, gripper_dof_pos, do_scale=False
        )

        # Step sim
        for _ in range(sim_steps):
            self.render()
            self.gym.simulate(self.sim)

    def _lift_gripper(self, gripper_dof_pos=0.0, lift_distance=0.3, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[:, 2] = lift_distance  # lift along z

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(
                delta_hand_pose, gripper_dof_pos, do_scale=False
            )
            self.render()
            self.gym.simulate(self.sim)

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(
            0.0, 1.0, num_keypoints, device=self.device
        )

        return keypoint_offsets

    def _get_keypoint_dist(self):
        """Get keypoint distances."""
        keypoint_dist = torch.mean(
            torch.norm(self.keypoints_bolt - self.keypoints_nut, p=2, dim=-1), dim=-1
        )
        return keypoint_dist

    def _get_center_dist(self):
        """Get center distance."""

        nut_pos = self.nut_pos - torch.tensor([0.0, 0.0, 0.04], device=self.device)
        center_dist = torch.norm(self.bolt_pos - nut_pos, p=2, dim=-1)

        return center_dist

    def _check_nut_close_to_bolt(self):
        """Check if nut is close to bolt."""
        keypoint_dist = torch.norm(
            self.keypoints_bolt - self.keypoints_nut, p=2, dim=-1
        )

        is_nut_close_to_bolt = torch.where(
            torch.mean(keypoint_dist, dim=-1) < self.cfg_task.rl.close_error_thresh,
            torch.ones_like(self.progress_buf),
            torch.zeros_like(self.progress_buf),
        )

        return is_nut_close_to_bolt

    def _check_nut_away_from_bolt(self):
        """Check if nut is away from bolt."""

        keypoint_dist = torch.norm(
            self.keypoints_bolt - self.keypoints_nut, p=2, dim=-1
        )

        is_nut_away_from_bolt = torch.where(
            torch.mean(keypoint_dist, dim=-1) >= self.cfg_task.rl.away_error_thresh,
            torch.ones_like(self.progress_buf),
            torch.zeros_like(self.progress_buf),
        )

        return is_nut_away_from_bolt

    def _move_down_arm(self, env_ids, sim_steps=20):
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(
            self.cfg_task.randomize.fingertip_midpoint_pos_initial,
            device=self.device,
        ) + torch.tensor([0.0, 0.00, -0.05], device=self.device)

        self.ctrl_target_fingertip_midpoint_pos = (
            self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
                self.num_envs, 1
            )
        )

        ctrl_target_fingertip_midpoint_euler = (
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2],
        )

        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
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

    def _randomize_gripper_pose(self, env_ids, sim_steps):
        """Move gripper to random pose."""

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos.clone()

        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(
            [0.0, 0.0, self.cfg_base.env.table_height], device=self.device
        ) + torch.tensor(
            self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device
        )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
                self.num_envs, 1
            )
        )

        fingertip_midpoint_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )
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
        )
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

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
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

    def plot_digits(self):
        for ax in self.axs.flatten():
            ax.clear()

        for env_id, ax in enumerate(self.axs.flatten()):
            img_left = self.digit_left_buf[env_id][0].cpu().numpy()
            img_right = self.digit_right_buf[env_id][0].cpu().numpy()
            img_left = cv2.resize(img_left, (240, 320), interpolation=cv2.INTER_LINEAR)
            img_right = cv2.resize(
                img_right, (240, 320), interpolation=cv2.INTER_LINEAR
            )
            # img_left = np.random.random(size=(64, 64, 3))
            img = np.concatenate((img_left, img_right), axis=1)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"env_id: {env_id}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
