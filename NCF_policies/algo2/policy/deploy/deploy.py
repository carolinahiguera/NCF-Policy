import os
import time
from matplotlib import pyplot as plt
import torch
import trimesh
import numpy as np
import cv2
from termcolor import cprint
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as Rot

from vedo import Plotter, Mesh, Points, show, Text2D
from vedo import trimesh2vedo

from algo2.policy.deploy.franka import FrankaEnv
from algo2.policy.models.models import ActorCritic
from algo2.policy.models.running_mean_std import RunningMeanStd
from algo2.utils.misc import add_to_fifo
from algo2.policy.deploy.ncf_viz import NCFViz

from pynput import keyboard

EEF_MAX = np.array([0.10, 0.10, 0.20])  # m
DELTA_EEF_MAX = np.array([1e-2, 1e-2, 2e-2])  # cm
DELTA_QUAT_MAX = np.array([0.01, 0.01, 0.01, 1.0])


@dataclass
class Digit_VAE_Params:
    root_path: str = os.path.abspath(os.path.join(".."))
    image_size: int = 64
    channels: int = 3
    seq_len: int = 5
    latent_dim: int = 64
    enc_out_dim: int = 512
    source_data: str = "sim"
    checkpoint_dir = os.path.join(root_path, "digit_vae", "checkpoints")


@dataclass
class NCF_Params:
    root_path: str = os.path.abspath(os.path.join("."))
    seq_len: int = 10
    source_data: str = "sim"
    hidden_size_ncf: int = 128
    pc_subsample: float = 1.0
    vae_params = Digit_VAE_Params()


class HardwarePlayer(object):
    def __init__(self, config):
        current_file = os.path.abspath(__file__)
        self.root_dir = os.path.abspath(
            os.path.join(current_file, "..", "..", "..", "..", "..")
        )

        self.action_scale_pos = 0.2  # 0.2  # proprio only 0.3
        self.action_scale_rot = 0.3  # 0.45  # proprio only 0.2
        self.control_freq = 30
        self.device = torch.device("cuda:0")

        self.full_config = config
        self.network_config = config.train.network
        self.ppo_config = config.train.ppo
        self.task_config = config.task

        # ---- Env Info ----
        self.actions_num = 6
        self.obs_shape = (21,)

        # ---- Priv Info ----
        self.priv_info_dim = self.ppo_config["ncf_output_dim"]
        self.priv_info = self.ppo_config["ncf_info"]
        self.ncf_proprio_adapt = False

        # ---- Tactile Info ----
        self.tactile_info = self.ppo_config["tactile_info"]
        self.tactile_seq_length = self.ppo_config["tactile_seq_length"]
        self.tactile_info_embed_dim = self.ppo_config["tactile_info_embed_dim"]
        self.tactile_info_dim = (
            self.tactile_info_embed_dim * self.tactile_seq_length * 2
        )

        # ---- NCF Info ----
        self.ncf_info = self.ppo_config["ncf_info"]
        self.ncf_use_gt = self.ppo_config["ncf_use_gt"]
        self.ncf_output_dim = self.ppo_config["ncf_output_dim"]
        if self.ncf_info:
            self.tactile_info = False

        # ---- Model ----
        net_config = {
            "actor_units": self.network_config.mlp.units,
            "actions_num": self.actions_num,
            "proprio_input_shape": self.obs_shape,
            "tactile_info": self.tactile_info,
            "ncf_info": self.ncf_info,
            "priv_info": self.priv_info,
            "tactile_units": self.network_config.tactile_mlp.units,
            "ncf_units": self.network_config.ncf_mlp.units,
            "ncf_adapt_units": self.network_config.ncf_adapt_mlp.units,
            "tactile_input_shape": (self.tactile_info_dim,),
            "ncf_input_shape": (self.ncf_output_dim,),
            "ncf_proprio_adapt": self.ncf_proprio_adapt,
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()

        if self.tactile_info:
            self.obs_shape = (self.obs_shape[0] + self.tactile_info_dim,)

        # ---- PPO Param ----
        self.normalize_input = self.ppo_config["normalize_input"]
        self.normalize_point_cloud = self.ppo_config["normalize_point_cloud"]
        self.obs_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.obs_mean_std.eval()
        self.point_cloud_mean_std = RunningMeanStd(
            3,
        ).to(self.device)
        self.point_cloud_mean_std.eval()

        # ---- DIGIT autoencoder
        # path_checkpoint = self.task_config.ncf.checkpoint_vae
        # self.digit_vae = self.load_digit_autoencoder(self.root_dir, path_checkpoint)
        # self.digit_vae.eval()
        # self.digit_vae.to(self.device)
        # self.reset_digits_fifo()
        # self.reset_eef_fifo()

        # # ---- NDF
        # path_checkpoint = self.task_config.ncf.checkpoint_ndf
        # self.ndf = self.load_ndf(self.root_dir, path_checkpoint)
        # self.ndf.eval()
        # self.ndf.to(self.device)

        # # ---- NCF
        # self.ncf = self.load_ncf(self.root_dir)
        # self.ncf.eval()
        # self.ncf.to(self.device)
        # # self.ncf_viz = NCF_Viz(
        # #     pc_file=self.task_config.ncf.path_pointcloud_object,
        # #     mesh_file=self.task_config.ncf.path_mesh_object,
        # # )

        # # ---- NCF Viz
        # self.viz = NCFViz(off_screen=False, zoom=1.0, window_size=0.70)

        # ---- NCF
        path_checkpoint_vae = self.task_config.ncf.checkpoint_vae
        path_checkpoint_ndf = self.task_config.ncf.checkpoint_ndf
        path_checkpoint_ncf = "{0}/{1}_{2}/model-epoch={3:03d}.ckpt".format(
            self.root_dir,
            self.task_config.ncf.checkpoint_ncf,
            self.task_config.ncf.ncf_arch,
            self.task_config.ncf.ncf_epoch,
        )
        self.digit_vae, self.ndf, self.ncf = self.load_ncf(
            self.task_config.ncf.ncf_arch,
            self.root_dir,
            path_checkpoint_vae,
            path_checkpoint_ndf,
            path_checkpoint_ncf,
        )
        self.digit_vae.eval()
        self.digit_vae.to(self.device)
        self.ncf.eval()
        self.ncf.to(self.device)

        # ---- NCF Viz
        self.viz = NCFViz(off_screen=False, zoom=1.0, window_size=0.70)

        # ---- NCF data
        path_ndf_code_object = self.task_config.ncf.path_ndf_code_object
        self.shape_code = np.load(path_ndf_code_object)
        # self.shape_code = np.zeros((1, 768))
        self.shape_code = torch.tensor(self.shape_code, dtype=torch.float32).unsqueeze(
            0
        )

        path_pointcloud_object = self.task_config.ncf.path_pointcloud_object
        self.ncf_point_cloud = np.load(path_pointcloud_object)
        self.ncf_point_cloud = self.ncf_point_cloud - np.mean(
            self.ncf_point_cloud, axis=0
        )
        self.n_points = len(self.ncf_point_cloud)

        # ---- Contact Filter
        win_sz = 5
        self.contact_buffer = torch.zeros((win_sz, 2000), dtype=torch.float32)

        # ---- loggers
        self.reset_loggers()

        self.stop_episode = False

    def restore(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input:
            self.obs_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(
                checkpoint["point_cloud_mean_std"]
            )

    def load_digit_autoencoder(self, root_dir, path_checkpoint):
        from algo2.ncf.vae.vae import VAE

        path_checkpoint = os.path.join(root_dir, path_checkpoint)
        vae = VAE.load_from_checkpoint(
            path_checkpoint,
            enc_out_dim=512,
            latent_dim=64,
            input_height=64,
        )
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        return vae

    def load_ndf(self, root_dir, path_checkpoint):
        from algo2.ncf.ndf.ndf import NDF

        path_checkpoint = os.path.join(root_dir, path_checkpoint)
        ndf = NDF(latent_dim=256, return_features=True, sigmoid=True)
        checkpoint = torch.load(path_checkpoint)
        ndf_weights = ndf.state_dict()

        for key in ndf_weights.keys():
            # print(key)
            ndf_weights[key] = checkpoint[key]
        ndf.load_state_dict(ndf_weights)
        for param in ndf.parameters():
            param.requires_grad = False

        return ndf

    def load_ncf(
        self,
        arch,
        root_dir,
        path_checkpoint_vae,
        path_checkpoint_ndf,
        path_checkpoint_ncf,
    ):
        from algo2.ncf.ncf.ncf_mlp import NCF as NCF_mlp
        from algo2.ncf.ncf.ncf_transformer import NCF as NCF_transformer
        from algo2.ncf.config.config import NCF_Params
        from algo2.ncf.pipeline import NCF_Pipeline

        cfg = NCF_Params()
        digit_encoder = self.load_digit_autoencoder(root_dir, path_checkpoint_vae)
        ndf_model = self.load_ndf(root_dir, path_checkpoint_ndf)

        if "mlp" in arch:
            ncf_model = NCF_mlp(cfg)
        elif "transformer" in arch:
            ncf_model = NCF_transformer(cfg)
        else:
            raise NotImplementedError

        pipeline = NCF_Pipeline.load_from_checkpoint(
            path_checkpoint_ncf,
            cfg=cfg,
            digit_vae=digit_encoder,
            ndf=ndf_model,
            ncf=ncf_model,
        )

        for param in pipeline.parameters():
            param.requires_grad = False

        # digit_encoder = self.load_digit_autoencoder(root_dir, path_checkpoint_vae)
        # return digit_encoder, pipeline.ncf
        return pipeline.digit_vae, pipeline.ndf, pipeline.ncf

    def reset_digits_fifo(self):
        self.digit_left_fifo = torch.zeros(
            (1, self.tactile_seq_length, self.tactile_info_embed_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.digit_right_fifo = torch.zeros(
            (1, self.tactile_seq_length, self.tactile_info_embed_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def reset_eef_fifo(self):
        self.eef_fifo = torch.zeros(
            (1, self.tactile_seq_length, 7),
            dtype=torch.float32,
            device=self.device,
        )
        self.eef_fifo[:, :, -1] = 1.0

    def _digit_encode(self, images):
        images = images.permute(0, 3, 1, 2).unsqueeze(0)
        with torch.no_grad():
            images_hat, digit_embeddings = self.digit_vae(images)

        return digit_embeddings.unsqueeze(0), images_hat

    def _get_digit_embeddings(self, obs_dict):
        digits_imgs_left = obs_dict["digits_left"]
        digits_imgs_right = obs_dict["digits_right"]

        digits_emb_left, left_hat = self._digit_encode(digits_imgs_left)
        digits_emb_right, right_hat = self._digit_encode(digits_imgs_right)

        self.digit_left_fifo = add_to_fifo(self.digit_left_fifo, digits_emb_left)
        self.digit_right_fifo = add_to_fifo(self.digit_right_fifo, digits_emb_right)

        self.img_left = digits_imgs_left[0].cpu().numpy()
        self.img_right = digits_imgs_right[0].cpu().numpy()
        self.img_left_vae = left_hat[0].permute(1, 2, 0).cpu().numpy()
        self.img_right_vae = right_hat[0].permute(1, 2, 0).cpu().numpy()

    def info2buffers(self, obs_dict):
        self.eef_fifo = add_to_fifo(
            self.eef_fifo, obs_dict["obs_ncf"][0:7].unsqueeze(0).unsqueeze(1)
        )
        self._get_digit_embeddings(obs_dict)
        # print(f"mug ncf = {obs_dict['obs_ncf'][0:3]}")

    def _get_ee_deltas(self):
        ee_seq = self.eef_fifo.clone().squeeze(0).cpu().numpy()
        # ee_seq[:, 2] -= 0.02
        # ee_seq[:, 2] -= 0.03
        r0 = Rot.from_quat(ee_seq[0][3:]).as_matrix()
        r1 = Rot.from_quat(ee_seq[1][3:]).as_matrix()
        r2 = Rot.from_quat(ee_seq[2][3:]).as_matrix()
        q_t_1 = Rot.from_matrix(np.matmul(r0, r1.T)).as_quat()
        q_t_2 = Rot.from_matrix(np.matmul(r0, r2.T)).as_quat()
        t_t_1 = ee_seq[0][0:3] - ee_seq[1][0:3]
        t_t_2 = ee_seq[0][0:3] - ee_seq[2][0:3]
        ee_t1 = np.concatenate((t_t_1, q_t_1))
        ee_t2 = np.concatenate((t_t_2, q_t_2))

        ee_seq[:, 0:3] = ee_seq[:, 0:3] / EEF_MAX
        ee_t1[0:3] = ee_t1[0:3] / DELTA_EEF_MAX
        ee_t1[3:] = ee_t1[3:] / DELTA_QUAT_MAX
        ee_t2[0:3] = ee_t2[0:3] / DELTA_EEF_MAX
        ee_t2[3:] = ee_t2[3:] / DELTA_QUAT_MAX

        ee_seq = torch.tensor(ee_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        ee_t1 = torch.tensor(ee_t1, dtype=torch.float32).unsqueeze(0).to(self.device)
        ee_t2 = torch.tensor(ee_t2, dtype=torch.float32).unsqueeze(0).to(self.device)

        return ee_seq, ee_t1, ee_t2

    def get_pointcloud_data(self):
        idx_points = np.arange(self.n_points)
        n_query_pts = int(self.n_points * 1.0)
        idx_query = np.random.choice(idx_points, size=n_query_pts)
        ncf_query_points = self.ncf_point_cloud[idx_query]
        ncf_query_points = torch.tensor(ncf_query_points, dtype=torch.float32)
        ncf_query_points = ncf_query_points.unsqueeze(0).to(self.device)
        shape_code = self.shape_code.to(self.device)
        return shape_code, ncf_query_points, idx_query

    def _get_pointcloud_data_ndf(self):
        # points for ncf
        path_point_cloud = self.task_config.ncf.path_pointcloud_object
        path_mesh = self.task_config.ncf.path_mesh_object

        ncf_point_cloud = np.load(path_point_cloud)
        ncf_point_cloud = ncf_point_cloud - np.mean(ncf_point_cloud, axis=0)
        n_points = len(ncf_point_cloud)

        idx_points = np.arange(n_points)
        n_query_pts = int(n_points * 1.0)
        idx_query = np.random.choice(idx_points, size=n_query_pts)
        ncf_query_points = ncf_point_cloud[idx_query]

        ncf_query_points = torch.tensor(ncf_query_points, dtype=torch.float32)
        ncf_query_points = ncf_query_points.unsqueeze(0).to(self.device)

        # points for ndf
        object_trimesh = trimesh.load(path_mesh)
        T = np.eye(4)
        T[0:3, 0:3] = Rot.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix()
        object_trimesh = object_trimesh.apply_transform(T)

        ndf_point_cloud = np.array(
            trimesh.sample.sample_surface(object_trimesh, n_points // 2)[0]
        )
        ndf_point_cloud = ndf_point_cloud - np.mean(ndf_point_cloud, axis=0)
        ndf_point_cloud = torch.tensor(ndf_point_cloud, dtype=torch.float32)
        ndf_point_cloud = ndf_point_cloud.unsqueeze(0).to(self.device)

        return ndf_point_cloud, ncf_query_points, idx_query

    def _get_ncf_output(self, obs_dict):
        self.info2buffers(obs_dict)

        digits_emb_left = self.digit_left_fifo.reshape(
            -1, self.tactile_seq_length * self.tactile_info_embed_dim
        )
        digits_emb_right = self.digit_right_fifo.reshape(
            -1, self.tactile_seq_length * self.tactile_info_embed_dim
        )

        ee_seq, ee_t1, ee_t2 = self._get_ee_deltas()

        # (
        #     shape_code,
        #     ncf_query_points,
        #     idx_query,
        # ) = self.get_pointcloud_data()

        ndf_point_cloud, ncf_query_points, idx_query = self._get_pointcloud_data_ndf()
        with torch.no_grad():
            shape_code = self.ndf(ndf_point_cloud)

        inputs_ncf = {
            "digit_emb_left": digits_emb_left,
            "digit_emb_right": digits_emb_right,
            "ee_pose": ee_seq,
            "ee_pose_1": ee_t1,
            "ee_pose_2": ee_t2,
            "shape_emb": shape_code,
            "ncf_query_points": ncf_query_points,
        }
        with torch.no_grad():
            contact = self.ncf(inputs_ncf)

        # d_mug_cupholder = (
        #     torch.norm(self.env.nut_pos - self.env.bolt_pos, dim=1)
        # ) > 0.13
        d_mug_cupholder = self.eef_fifo[0][0][2] > 0.09
        if d_mug_cupholder:
            contact = torch.zeros_like(contact)
        # contact = torch.where(contact > 0.4, 1.0, 0.0)
        contact[contact > 0.4] = 1.0
        contact_gt = contact.clone()

        return contact_gt, contact, idx_query, inputs_ncf

    def log_data(self, obs_dict, inputs_ncf, ncf_contact, idx_query):
        for key in inputs_ncf.keys():
            inputs_ncf[key] = inputs_ncf[key].cpu().numpy()

        joint_pos = self.franka_env.get_joint_positions()
        self.log_joint_pos.append(joint_pos.numpy())
        self.log_obs.append(obs_dict["obs"].cpu().numpy())
        self.log_obs_ncf.append(obs_dict["obs_ncf"].cpu().numpy())
        self.log_obs_ee.append(obs_dict["obs_ee"].cpu().numpy())
        self.log_ncf_inputs.append(inputs_ncf)
        self.log_ncf_contact.append(ncf_contact.cpu().numpy())
        self.log_ncf_idx_query.append(idx_query)
        self.log_digit_left_imgs.append((self.img_left * 255).astype(np.uint8))
        self.log_digit_right_imgs.append((self.img_right * 255).astype(np.uint8))
        self.log_rs_front.append(self.franka_env.rs_front_img)
        self.log_rs_rear.append(self.franka_env.rs_rear_img)

    def reset_loggers(self):
        self.log_joint_pos = []
        self.log_obs = []
        self.log_obs_ncf = []
        self.log_obs_ee = []
        self.log_ncf_inputs = []
        self.log_ncf_contact = []
        self.log_ncf_idx_query = []
        self.log_digit_left_imgs = []
        self.log_digit_right_imgs = []
        self.log_rs_front = []
        self.log_rs_rear = []

    def save_images(self, imgs, path):
        for i, img in enumerate(imgs):
            img = img[:, :, ::-1]
            cv2.imwrite(f"{path}/{i:03d}.png", img)

    def save_logs(self, path_save):
        os.makedirs(path_save, exist_ok=True)
        path_file_save_digit_left = f"{path_save}/real_left/"
        path_file_save_digit_right = f"{path_save}/real_right/"
        path_file_save_rs_front = f"{path_save}/rs_front/"
        path_file_save_rs_rear = f"{path_save}/rs_rear/"
        os.makedirs(path_file_save_digit_left, exist_ok=True)
        os.makedirs(path_file_save_digit_right, exist_ok=True)
        os.makedirs(path_file_save_rs_front, exist_ok=True)
        os.makedirs(path_file_save_rs_rear, exist_ok=True)

        self.save_images(self.log_digit_left_imgs, path_file_save_digit_left)
        self.save_images(self.log_digit_right_imgs, path_file_save_digit_right)
        self.save_images(self.log_rs_front, path_file_save_rs_front)
        self.save_images(self.log_rs_rear, path_file_save_rs_rear)

        log_data = {
            "joint_pos": self.log_joint_pos,
            "obs": self.log_obs,
            "obs_ncf": self.log_obs_ncf,
            "obs_ee": self.log_obs_ee,
            "ncf_inputs": self.log_ncf_inputs,
            "ncf_contact": self.log_ncf_contact,
            "idx_query": self.log_ncf_idx_query,
        }
        np.savez_compressed(f"{path_save}/log_data.npz", **log_data)
        self.reset_loggers()

    def get_policy_inputs(self, obs_dict):
        if self.tactile_info:
            self.info2buffers(obs_dict)
            digits_emb_left = self.digit_left_fifo.reshape(
                -1, self.tactile_seq_length * self.tactile_info_embed_dim
            )
            digits_emb_right = self.digit_right_fifo.reshape(
                -1, self.tactile_seq_length * self.tactile_info_embed_dim
            )

            digits_emb = torch.cat([digits_emb_left, digits_emb_right], dim=-1)

            aug_obs = torch.cat([obs_dict["obs"], digits_emb[0]], dim=-1)
            processed_obs = self.obs_mean_std(aug_obs).unsqueeze(0)

        else:
            aug_obs = obs_dict["obs"]
            processed_obs = self.obs_mean_std(obs_dict["obs"]).unsqueeze(0)

        return processed_obs, aug_obs

    def on_press(self, key):
        if key == keyboard.Key.space:
            self.stop_episode = True

    def deploy(self, c):
        # create franka object and connect to robot
        # ----- Robot -------
        self.franka_env = FrankaEnv(
            self.task_config,
            self.action_scale_pos,
            self.action_scale_rot,
            with_realsense=False,
            pointcloud_obj=self.task_config.ncf.path_pointcloud_object,
            use_worldref_pointcloud=self.task_config.rl.world_ref_pointcloud,
        )

        self.viz.init_variables(
            object_mesh_path=self.task_config.ncf.path_mesh_object,
            object_pointcloud=self.task_config.ncf.path_pointcloud_object,
        )

        start_episode = 0
        n_episodes = 15
        f_sample = 30.0

        if self.tactile_info:
            path_save = (
                "/home/fair/carolina/NCF_RL/outputs_policies/paper/proprio_tactile/"
            )
        elif self.ncf_info:
            path_save = "/home/fair/carolina/NCF_RL/outputs_policies/paper/proprio_ncf_{0}_epoch_{1:02d}/".format(
                self.task_config.ncf.ncf_arch,
                self.task_config.ncf.ncf_epoch,
            )
        else:
            path_save = (
                "/home/fair/carolina/NCF_RL/outputs_policies/paper/proprio_only/"
            )
        os.makedirs(path_save, exist_ok=True)

        self.stop_episode = False
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        for ep in range(start_episode, start_episode + n_episodes):
            cprint(f"Starting episode {ep}", "green", attrs=["bold"])

            obs_dict = self.franka_env.reset(ep)
            self.franka_env.read_realsense()

            self.franka_env.print_robot_status()
            self.reset_digits_fifo()
            self.reset_eef_fifo()

            for _ in range(10):
                self.info2buffers(obs_dict)

            step = 0
            path_save_logger = f"{path_save}/test_{ep}/"
            self.stop_episode = False

            while True:
                # for i in range(10):
                self.franka_env.read_realsense()
                # processed_obs, aug_obs, ncf_output = self.get_policy_inputs(obs_dict)
                # input_dict = {
                #     "obs": processed_obs,
                # }
                processed_obs, aug_obs = self.get_policy_inputs(obs_dict)
                # processed_obs = self.obs_mean_std(obs_dict["obs"]).unsqueeze(0)
                point_cloud = None

                if self.priv_info:
                    (
                        ncf_priv_info,
                        pred_contact,
                        idx_query,
                        inputs_ncf,
                    ) = self._get_ncf_output(obs_dict)
                    point_cloud = obs_dict["pointclouds_t"]
                    ncf_mask = torch.where(ncf_priv_info > 0.1, 1.0, 0.0)
                    ncf_mask = ncf_mask.repeat(3, 1, 1).permute(1, 2, 0)
                    point_cloud = point_cloud * ncf_mask
                    point_cloud = point_cloud.unsqueeze(0)
                    obs_dict["pointclouds_t"] = point_cloud

                    # if point_cloud[0].sum() > 0.0:
                    #     print("contact")

                    if self.normalize_point_cloud:
                        point_cloud = self.point_cloud_mean_std(
                            point_cloud.reshape(-1, 3)
                        ).reshape((processed_obs.shape[0], -1, 3))

                input_dict = {
                    "obs": processed_obs,
                    "priv_point_cloud": point_cloud,
                }

                action = self.model.act_inference(input_dict).squeeze()
                action = torch.clamp(action, -1.0, 1.0)
                obs_dict, r, done, info = self.franka_env.step(action)
                # self.franka_env.print_robot_status()

                if not self.ncf_info:
                    (
                        ncf_priv_info,
                        pred_contact,
                        idx_query,
                        inputs_ncf,
                    ) = self._get_ncf_output(obs_dict)

                self.log_data(obs_dict, inputs_ncf, pred_contact, idx_query)

                info["episode"] = ep
                info["step"] = step
                info["policy"] = (
                    "proprioception only"
                    if not self.ncf_info
                    else "proprioception + gt contact"
                )

                if self.stop_episode:
                    print("Stopping episode...")
                    break

                # path_save_step = path_save + "ep_{0:02d}_step_{1:02d}.png".format(
                #     ep, step
                # )
                # path_save_logger_step = path_save + "ep_{0:02d}.npz".format(ep)

                # self.viz.update(
                #     info=info,
                #     rs_img=[self.franka_env.rs_front_img, self.franka_env.rs_rear_img],
                #     digit_left=self.img_left,
                #     digit_right=self.img_right,
                #     vae_left=self.img_left_vae,
                #     vae_right=self.img_right_vae,
                #     contact_vector=pred_contact,
                #     idx_query=idx_query,
                #     image_savepath=None,
                # )

                if done:
                    self.franka_env.print_robot_status()
                    print("")
                    print("is_mug_inserted = {0}".format(info["is_mug_inserted"]))
                    print("is_mug_close = {0}".format(info["is_mug_close"]))
                    print("is_mug_oriented = {0}".format(info["is_mug_oriented"]))
                    print("error pos = {0:04f}".format(info["err_pos"]))
                    print("error quat = {0:04f}".format(info["err_quat"]))
                    print("")
                    break

                step += 1
                time.sleep(1 / f_sample)
            # self.viz.save_imgs()
            self.save_logs(path_save_logger)
        self.viz.close()
