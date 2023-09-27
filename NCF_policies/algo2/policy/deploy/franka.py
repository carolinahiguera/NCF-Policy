import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from isaacgymenvs.utils import torch_jit_utils as torch_utils

import torch
from torchcontrol.transform import Rotation as R
from scipy.spatial.transform import Rotation as Rot
from polymetis import RobotInterface
from digit_interface.digit import Digit
import pyrealsense2 as rs

from algo2.utils.misc import tprint


class FrankaEnv:
    def __init__(
        self,
        task_config,
        action_scale_pos,
        action_scale_rot,
        pointcloud_obj,
        with_realsense: bool = False,
        use_worldref_pointcloud: bool = True,
    ):
        print("[robot] configuring robot - setting home pose...")
        self.robot = RobotInterface(ip_address="172.16.0.1", enforce_version=False)
        self.task_config = task_config
        self.action_scale_pos = action_scale_pos
        self.action_scale_rot = action_scale_rot
        # self.home_pose = torch.tensor(self.task_config.randomize.franka_arm_initial_dof_pos)
        self.home_pose = torch.tensor(
            [0.0139, -0.0544, -0.0048, -2.2373, -0.0085, 2.1169, 0.7795]
        )

        # transformations
        # self.ee_offset = torch.tensor([0.5119, 0.0079, 0.19]) # this is the position of the cupholder
        # [0.5319, 0.0079, 0.19]
        self.ee_offset = torch.tensor(
            [0.4931, 0.0041, 0.2222]  # proprio_only
            # [0.50, -0.02, 0.19]
        )  # this is the position of the cupholder
        # [0.4792, 0.0101, 0.2111]
        self.ncf_offset = torch.tensor([0.4931, 0.0041, 0.2122]) + torch.tensor(
            [0.0, -0.02, -0.01]
        )
        # self.ncf_offset = torch.tensor([0.4792, 0.0101, 0.19])
        self.ee_quat_ref = torch.tensor([0.9231, -0.3846, -0.0021, 0.0055])
        ee_quat_ref_sim = torch.tensor([-1.21e-3, 0.99, 2.2e-3, 2.23e-3])
        self.T_real_sim = torch_utils.quat_mul(
            ee_quat_ref_sim, torch_utils.quat_conjugate(self.ee_quat_ref)
        )

        # global variables
        self.digit_sz = 64
        self.h_mug = 0.10
        self.num_envs = 1
        self.device = torch.device("cuda:0")

        self.digit_left = self.config_digit(id="D00011", name="red finger")
        self.digit_right = self.config_digit(id="D00024", name="blue finger")
        self.stabilize_digits()
        self.bg_left, self.bg_right = self.load_bgs()
        print("[robot] configuring DIGIT sensors - ready")

        rs_rear_serial = "152122075373"
        rs_front_serial = "815412070977"
        self.rs_rear_cam = self.config_realsense(rs_rear_serial)
        self.rs_front_cam = self.config_realsense(rs_front_serial)
        self.rs_rear_img = None
        self.rs_front_img = None

        self.pointcloud_obj = np.load(pointcloud_obj)
        self.n_points = self.pointcloud_obj.shape[0]
        self.use_worldref_pointcloud = use_worldref_pointcloud

        self.target_pos = torch.load("target_pos.pt")
        self.target_quat = torch.load("target_quat.pt")

        # self.reset_franka()
        print("[robot] configuring robot - done")

    def load_bgs(self):
        # get path to currrent file
        path = os.path.dirname(os.path.abspath(__file__))
        bg_left = cv2.imread(f"{path}/resources/D00011.png")
        # bg_left = cv2.cvtColor(bg_left, cv2.COLOR_RGB2BGR)
        bg_right = cv2.imread(f"{path}/resources/D00024.png")
        # bg_right = cv2.cvtColor(bg_right, cv2.COLOR_RGB2BGR)
        return bg_left, bg_right

    def config_digit(self, id: str, name: str):
        digit = Digit(id, name)
        digit.connect()
        digit.set_intensity(Digit.LIGHTING_MAX)
        qvga_res = Digit.STREAMS["QVGA"]
        digit.set_resolution(qvga_res)
        fps_30 = Digit.STREAMS["QVGA"]["fps"]["30fps"]
        digit.set_fps(fps_30)
        return digit

    def stabilize_digits(self):
        for _ in range(30):
            self.digit_left.get_frame()
            self.digit_right.get_frame()

    def config_realsense(self, serial):
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)
        return pipeline

    def _subtract_bg(self, img1, img2, offset=0.5):
        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0 + offset
        diff = np.clip(diff, 0.0, 1.0)
        diff = cv2.resize(
            diff, (self.digit_sz, self.digit_sz), interpolation=cv2.INTER_AREA
        )
        return diff

    def read_digits(self):
        img_left = self.digit_left.get_frame()
        img_right = self.digit_right.get_frame()
        img_left = self._subtract_bg(img_left, self.bg_left)
        img_right = self._subtract_bg(img_right, self.bg_right)
        img_left = cv2.flip(img_left, 0)
        img_right = cv2.flip(img_right, 0)
        return img_left, img_right

    def read_realsense(self):
        frames = self.rs_front_cam.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            img_realsense = np.asanyarray(color_frame.get_data())
            self.rs_front_img = cv2.cvtColor(img_realsense, cv2.COLOR_RGB2BGR)

        frames = self.rs_rear_cam.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame:
            img_realsense = np.asanyarray(color_frame.get_data())
            self.rs_rear_img = cv2.cvtColor(img_realsense, cv2.COLOR_RGB2BGR)

    def lift_up(self):
        delta_pos = torch.tensor([0.0, 0.0, 0.05])
        self.robot.move_to_ee_pose(position=delta_pos, time_to_go=5.0, delta=True)
        time.sleep(0.1)

    def _start_franka_controller(self):
        Kx = [250, 500, 500, 10, 10, 10]
        Kxd = [25, 5, 1, 7, 7, 7]
        a = self.robot.start_cartesian_impedance(Kx, Kxd)
        time.sleep(0.1)
        print(a)
        print("[robot] cartesian impedance controller initialized")

    def reset_franka(self, ep=-1):
        self.lift_up()
        self.robot.set_home_pose(self.home_pose)
        self.robot.go_home()
        time.sleep(0.5)
        # delta_pos = torch.tensor([0.0, 0.0, -0.05])
        # self.robot.move_to_ee_pose(position=delta_pos, time_to_go=3.0, delta=True)
        # time.sleep(0.5)

        if ep == -1:
            self._randomize_gripper_pose()
        else:
            self.set_init_pose(self.target_pos[ep], self.target_quat[ep])

        time.sleep(0.5)
        self._start_franka_controller()
        self.reset_buffers()
        print("[robot] reset robot")

    def reset_buffers(self):
        digit_seq_len = 1
        digit_sz = 64

        self.obs_ncf_buf = torch.zeros(
            (self.num_envs, 21), device=self.device, dtype=torch.float
        )
        self.obs_buf = torch.zeros(
            (self.num_envs, 21), device=self.device, dtype=torch.float
        )
        self.obs_ee_buf = torch.zeros(
            (self.num_envs, 21), device=self.device, dtype=torch.float
        )
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.digit_left_buf = torch.zeros(
            (self.num_envs, digit_seq_len, digit_sz, digit_sz, 3),
            device=self.device,
            dtype=torch.float,
        )
        self.digit_right_buf = torch.zeros(
            (self.num_envs, digit_seq_len, digit_sz, digit_sz, 3),
            device=self.device,
            dtype=torch.float,
        )
        self.pointcloud_buf = torch.zeros(
            (self.num_envs, self.n_points, 3),
            device=self.device,
            dtype=torch.float,
        )
        self.obs_dict = {}

    def reset(self, ep=-1):
        self.reset_franka(ep=ep)
        self.get_observations_proprio()
        self.get_observations_tactile()
        self.obs_dict["obs"] = self.obs_buf.to(self.device)
        self.obs_dict["obs_ee"] = self.obs_ee_buf.to(self.device)
        self.obs_dict["obs_ncf"] = self.obs_ncf_buf.to(self.device)
        self.obs_dict["digits_left"] = self.digit_left_buf.to(self.device)
        self.obs_dict["digits_right"] = self.digit_right_buf.to(self.device)
        self.obs_dict["pointclouds_t"] = self.pointcloud_buf.to(self.device)
        return self.obs_dict

    def get_ee_offset(self):
        ee_pos, ee_quat = self.robot.get_ee_pose()
        offset = torch.tensor([0.4819, 0.07, 0.14])
        ee_pos = ee_pos - offset
        ee_quat = torch_utils.quat_mul(self.T_real_sim, ee_quat)
        cupholder_pos = torch.tensor([0.0, 0.0, 0.0])
        cupholder_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
        ee_pose = torch.cat([ee_pos, ee_quat, cupholder_pos, cupholder_quat])
        return ee_pose

    def get_ee_pose(self):
        ee_pos, ee_quat = self.robot.get_ee_pose()
        # ee_pos = ee_pos.to(self.device)
        # ee_quat = ee_quat.to(self.device)
        return ee_pos, ee_quat

    def get_joint_positions(self):
        joint_positions = self.robot.get_joint_positions()
        return joint_positions

    def print_robot_status(self):
        ee_pos, ee_quat = self.robot.get_ee_pose()
        mug_pose = ee_pos - self.ee_offset
        # mug_pose[2] -= self.h_mug
        mug_quat = torch_utils.quat_mul(self.T_real_sim, ee_quat)
        info_string = (
            f"[robot] ee_pos = {ee_pos} \n"
            f"[robot] ee_quat = {ee_quat} \n"
            f"[robot] mug_pos (policy) = {mug_pose} \n"
            f"[robot] mug_quat (policy) = {mug_quat} \n"
            f""
        )
        tprint(info_string)

    def _xyzquat_to_tf_numpy(self, position_quat: np.ndarray) -> np.ndarray:
        """
        convert [x, y, z, qx, qy, qz, qw] to 4 x 4 transformation matrices
        """
        # try:
        position_quat = np.atleast_2d(position_quat)  # (N, 7)
        N = position_quat.shape[0]
        T = np.zeros((N, 4, 4))
        T[:, 0:3, 0:3] = Rot.from_quat(position_quat[:, 3:]).as_matrix()
        T[:, :3, 3] = position_quat[:, :3]
        T[:, 3, 3] = 1
        # except ValueError:
        #     print("Zero quat error!")
        return T.squeeze()

    def get_observations_proprio(self):
        ee_pos, ee_quat = self.robot.get_ee_pose()
        cupholder_pos = torch.tensor([0.0, 0.0, 0.0])
        cupholder_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
        ee_quat = torch_utils.quat_mul(self.T_real_sim, ee_quat)
        # ee_quat_ncf = torch_utils.quat_mul(torch.tensor([0, 0, 1, 0]), ee_quat)
        obs_tensors = [ee_pos - self.ee_offset, ee_quat]
        obs_ncf_tensors = [ee_pos - self.ncf_offset, ee_quat]
        obs_ee_tensors = [ee_pos, ee_quat]
        # print("ee_ncf = ", ee_pos - self.ncf_offset)

        # transform pointcloud
        if self.use_worldref_pointcloud:
            mug_pos = ee_pos - self.ee_offset
            object_pose = torch.cat((mug_pos, ee_quat), dim=-1).unsqueeze(0)
            object_pose = self._xyzquat_to_tf_numpy(object_pose.cpu().numpy())
            object_pc_i = trimesh.points.PointCloud(self.pointcloud_obj.copy())
            object_pc_i.apply_transform(object_pose)
            coords = object_pc_i.vertices
        else:
            coords = self.pointcloud_obj.copy()

        if self.obs_buf.sum() == 0:
            curr_obs = torch.cat(obs_tensors, dim=-1)
            self.obs_buf = torch.cat((curr_obs, curr_obs, curr_obs), dim=-1)
            curr_obs = torch.cat(obs_ncf_tensors, dim=-1)
            self.obs_ncf_buf = torch.cat((curr_obs, curr_obs, curr_obs), dim=-1)
            curr_obs = torch.cat(obs_ee_tensors, dim=-1)
            self.obs_ee_buf = torch.cat((curr_obs, curr_obs, curr_obs), dim=-1)
        else:
            curr_obs = torch.cat(obs_tensors, dim=-1)
            prev_obs_buf = self.obs_buf[0:14].clone()
            self.obs_buf = torch.cat((curr_obs, prev_obs_buf), dim=-1)
            curr_obs = torch.cat(obs_ncf_tensors, dim=-1)
            prev_obs_buf = self.obs_ncf_buf[0:14].clone()
            self.obs_ncf_buf = torch.cat((curr_obs, prev_obs_buf), dim=-1)
            curr_obs = torch.cat(obs_ee_tensors, dim=-1)
            prev_obs_buf = self.obs_ee_buf[0:14].clone()
            self.obs_ee_buf = torch.cat((curr_obs, prev_obs_buf), dim=-1)

        self.pointcloud_buf = torch.tensor(
            coords, device=self.device, dtype=torch.float
        )

        # print(f"[robot] obs_buf = {self.obs_buf[0:3]}")
        # print(f"[robot] obs_ncf_buf = {self.obs_ncf_buf[0:3]}")

    def get_observations_tactile(self):
        digit_seq_len = 1
        digit_sz = 64
        img_left, img_right = self.read_digits()

        self.digit_left_buf = torch.tensor(img_left, dtype=torch.float).unsqueeze(0)
        self.digit_right_buf = torch.tensor(img_right, dtype=torch.float).unsqueeze(0)

    def _randomize_gripper_pose(self):
        ee_pos, ee_quat = self.robot.get_ee_pose()
        ctrl_target_fingertip_midpoint_pos = ee_pos
        fingertip_midpoint_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32) - 0.5
        )  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(self.task_config.randomize.fingertip_midpoint_pos_noise)
        )

        # fingertip_midpoint_pos_noise[0, 2] = 0.0
        ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise.squeeze()

        x, y, z = torch_utils.get_euler_xyz(ee_quat.unsqueeze(0))
        ctrl_target_fingertip_midpoint_euler = torch.cat([x, y, z])
        fingertip_midpoint_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32) - 0.5
        )  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(self.task_config.randomize.fingertip_midpoint_rot_noise)
        )
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise.squeeze()
        ctrl_target_fingertip_midpoint_quat = (
            torch_utils.quat_from_euler_xyz(
                ctrl_target_fingertip_midpoint_euler[0],
                ctrl_target_fingertip_midpoint_euler[1],
                ctrl_target_fingertip_midpoint_euler[2],
            )
            * -1.0
        )

        self.robot.move_to_ee_pose(
            position=ctrl_target_fingertip_midpoint_pos,
            orientation=ctrl_target_fingertip_midpoint_quat,
            time_to_go=3.0,
            delta=False,
        )
        # return ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat

    def set_init_pose(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat
    ):
        self.robot.move_to_ee_pose(
            position=ctrl_target_fingertip_midpoint_pos,
            orientation=ctrl_target_fingertip_midpoint_quat,
            time_to_go=3.0,
            delta=False,
        )

    def _apply_actions_as_ctrl_targets(self, actions, do_scale=False):
        actions = actions.cpu()
        """Apply actions from policy as position/rotation targets."""
        ee_pos, ee_quat = self.get_ee_pose()
        ee_quat = torch_utils.quat_mul(self.T_real_sim, ee_quat)
        # Interpret actions as tar get pos displacements and set pos target
        pos_actions = actions[0:3] * self.action_scale_pos
        if do_scale:
            pos_actions = pos_actions @ torch.diag(
                torch.tensor(self.task_config.rl.pos_action_scale)
            )

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[3:6] * self.action_scale_rot
        if do_scale:
            rot_actions = rot_actions @ torch.diag(
                torch.tensor(self.task_config.rl.rot_action_scale)
            )
        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.task_config.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.task_config.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
            )
        rot_actions_quat = rot_actions_quat.squeeze()
        rot_actions_quat = torch_utils.quat_mul(rot_actions_quat, ee_quat)

        # print(f"pos_actions = {pos_actions}")
        # print(f"rots_actions = {rot_actions}")
        # print("")

        # pos_actions[torch.abs(pos_actions) < 0.005] *= 5.0
        # pos_actions[0] = 0.0
        # pos_actions[1] = 0.0
        self.ctrl_target_ee_pos = ee_pos + pos_actions

        self.ctrl_target_ee_quat = torch_utils.quat_mul(
            torch_utils.quat_conjugate(self.T_real_sim), rot_actions_quat
        )
        self.ctrl_target_ee_quat = torch_utils.normalize(self.ctrl_target_ee_quat)

        # self.ctrl_target_ee_quat = torch.tensor([0.9231, -0.3846, -0.0021, 0.0055])

        # send actions to robot
        try:
            res = self.robot.update_desired_ee_pose(
                position=self.ctrl_target_ee_pos, orientation=self.ctrl_target_ee_quat
            )
        except:
            self._start_franka_controller()
        # print(res)
        time.sleep(0.1)
        # except:
        #     self._start_franka_controller()

    def check_mug_far(self):
        ee_pos, ee_quat = self.robot.get_ee_pose()
        if (ee_pos[0] < 0.40 and ee_pos[2] < 0.25) or (
            torch.abs(ee_pos[1]) > 0.05 and ee_pos[0] > 0.50 and ee_pos[2] > 0.30
        ):
            return True
        else:
            return False

    def check_mug_in_cupholder(self):
        ee_pos, ee_quat = self.robot.get_ee_pose()
        mug_pos = ee_pos - self.ee_offset
        cupholder_pos = torch.tensor([0.0, 0.01, 0.02])
        err_pos = torch.norm(cupholder_pos - mug_pos, p=2, dim=-1)
        err_quat = torch.norm(self.ee_quat_ref - ee_quat, p=2, dim=-1)
        return err_pos, err_quat

    def step(self, actions):
        self.actions = actions.clone().to(self.device)
        self._apply_actions_as_ctrl_targets(actions=self.actions, do_scale=True)

        # get observations
        self.get_observations_proprio()
        self.get_observations_tactile()
        self.obs_dict["obs"] = self.obs_buf.to(self.device)
        self.obs_dict["obs_ncf"] = self.obs_ncf_buf.to(self.device)
        self.obs_dict["obs_ee"] = self.obs_ee_buf.to(self.device)
        self.obs_dict["digits_left"] = self.digit_left_buf.to(self.device)
        self.obs_dict["digits_right"] = self.digit_right_buf.to(self.device)
        self.obs_dict["pointcloud_t"] = self.pointcloud_buf.to(self.device)

        err_pos, err_quat = self.check_mug_in_cupholder()
        r = (
            self.task_config.rl.keypoint_reward_scale * err_pos
            + self.task_config.rl.orientation_penalty_scale * err_quat
        )

        is_mug_close = err_pos < 0.021  # self.task_config.rl.close_error_thresh
        is_mug_oriented = err_quat < 0.1
        is_mug_inserted = is_mug_close * is_mug_oriented
        # is_mug_far = self.check_mug_far()
        is_mug_far = False

        # is_last_step = self.progress_buf[0] == self.task_config.rl.max_episode_length - 1
        is_last_step = self.progress_buf[0] == 100 - 1
        # done = (
        #     is_last_step.item() or is_mug_close.item() or is_mug_far
        # )  # TODO: should be is_mug_inserted
        done = is_last_step.item() or is_mug_far

        # done = False

        info = {
            "is_mug_close": is_mug_close,
            "is_mug_oriented": is_mug_oriented,
            "is_mug_inserted": is_mug_inserted,
            "err_pos": err_pos,
            "err_quat": err_quat,
        }

        self.progress_buf[0] += 1

        if done:
            self.robot.terminate_current_policy()

        return self.obs_dict, r, done, info
