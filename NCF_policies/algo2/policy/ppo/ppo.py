# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

import os
import time
import cv2
import torch
import torch.distributed as dist
import numpy as np

from algo2.policy.ppo.experience import ExperienceBuffer
from algo2.policy.models.models import ActorCritic
from algo2.policy.models.running_mean_std import RunningMeanStd

from algo2.utils.misc import AverageScalarMeter
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
from algo2.utils.misc import add_to_fifo, multi_gpu_aggregate_stats
from algo2.ncf.utils.utils_ncf import NCF_dataloader, NCF_Viz


DEBUG = False


class PPO(object):
    def __init__(self, env, output_dir, full_config):
        current_file = os.path.abspath(__file__)
        self.root_dir = os.path.abspath(
            os.path.join(current_file, "..", "..", "..", "..", "..")
        )

        # ---- MultiGPU ----
        self.multi_gpu = full_config.train.ppo.multi_gpu
        if self.multi_gpu:
            self.rank = int(os.getenv("LOCAL_RANK", "0"))
            self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)
            self.device = "cuda:" + str(self.rank)
            print(f"current rank: {self.rank} and use device {self.device}")
        else:
            self.rank = -1
            self.device = full_config["rl_device"]
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo

        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config["num_actors"]
        action_space = self.env.action_space
        self.actions_num = action_space.shape[0]
        self.actions_low = (
            torch.from_numpy(action_space.low.copy()).float().to(self.device)
        )
        self.actions_high = (
            torch.from_numpy(action_space.high.copy()).float().to(self.device)
        )
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape

        # ---- Priv Info ----
        self.priv_info_dim = self.ppo_config["ncf_output_dim"]
        self.priv_info = self.ppo_config[
            "ncf_info"
        ]  # and self.ppo_config["ncf_use_gt"]
        # self.proprio_adapt = self.ppo_config["proprio_adapt"]

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
        self.ncf_proprio_adapt = False

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

        if self.tactile_info:
            self.obs_shape = (self.obs_shape[0] + self.tactile_info_dim,)

        self.obs_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.point_cloud_mean_std = RunningMeanStd(
            3,
        ).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)

        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, "stage1_nn")
        self.tb_dif = os.path.join(self.output_dir, "stage1_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)

        # ---- Optim ----
        self.last_lr = float(self.ppo_config["learning_rate"])
        self.weight_decay = self.ppo_config.get("weight_decay", 0.0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.last_lr, weight_decay=self.weight_decay
        )

        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config["e_clip"]
        self.clip_value = self.ppo_config["clip_value"]
        self.entropy_coef = self.ppo_config["entropy_coef"]
        self.critic_coef = self.ppo_config["critic_coef"]
        self.bounds_loss_coef = self.ppo_config["bounds_loss_coef"]
        self.gamma = self.ppo_config["gamma"]
        self.tau = self.ppo_config["tau"]
        self.truncate_grads = self.ppo_config["truncate_grads"]
        self.grad_norm = self.ppo_config["grad_norm"]
        self.value_bootstrap = self.ppo_config["value_bootstrap"]
        self.normalize_advantage = self.ppo_config["normalize_advantage"]
        self.normalize_input = self.ppo_config["normalize_input"]
        self.normalize_value = self.ppo_config["normalize_value"]
        self.normalize_point_cloud = self.ppo_config["normalize_point_cloud"]
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config["horizon_length"]
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config["minibatch_size"]
        self.mini_epochs_num = self.ppo_config["mini_epochs"]
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.kl_threshold = self.ppo_config["kl_threshold"]
        self.scheduler = AdaptiveScheduler(self.kl_threshold)
        # ---- Snapshot
        self.save_freq = self.ppo_config["save_frequency"]
        self.save_best_after = self.ppo_config["save_best_after"]
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        # ---- Rollout GIFs ----
        self.gif_frame_counter = 0
        self.gif_save_every_n = 7500
        self.gif_save_length = 300
        self.gif_frames = []

        self.episode_rewards = AverageScalarMeter(50)
        self.episode_lengths = AverageScalarMeter(50)
        self.episode_success = AverageScalarMeter(50)
        self.obs = None
        self.epoch_num = 0

        self.storage = ExperienceBuffer(
            self.num_actors,
            self.horizon_length,
            self.batch_size,
            self.minibatch_size,
            self.obs_shape[0],
            self.actions_num,
            self.priv_info_dim,
            self.ncf_output_dim,
            self.device,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(
            current_rewards_shape, dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config["max_agent_steps"]
        self.best_rewards = -10000

        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

        # ---- DIGIT autoencoder
        if self.tactile_info:
            path_checkpoint = full_config["task"]["ncf"]["checkpoint_vae"]
            self.digit_vae = load_digit_autoencoder(self.root_dir, path_checkpoint)
            self.digit_vae.to(self.device)
            self.digit_vae.eval()

        # ---- NCF
        if DEBUG:
            self.ncf_viz = NCF_Viz()

        if (self.ncf_info and not self.ncf_use_gt) or DEBUG:
            path_pointcloud_object = os.path.join(
                self.root_dir, full_config["task"]["ncf"]["path_pointcloud_object"]
            )
            path_ndf_code_object = os.path.join(
                self.root_dir, full_config["task"]["ncf"]["path_ndf_code_object"]
            )
            path_mesh_object = os.path.join(
                self.root_dir, full_config["task"]["ncf"]["path_mesh_object"]
            )
            self.ncf_dataloader = NCF_dataloader(
                num_envs=self.num_actors,
                path_mesh_object=path_mesh_object,
                path_pointcloud_object=path_pointcloud_object,
                path_ndf_code_object=path_ndf_code_object,
                pc_subsample=self.ppo_config["ncf_pc_subsample"],
            )
            path_checkpoint_vae = full_config["task"]["ncf"]["checkpoint_vae"]
            path_checkpoint_ndf = full_config["task"]["ncf"]["checkpoint_ndf"]

            path_checkpoint_ncf = "{0}/{1}_{2}/model-epoch={3:03d}.ckpt".format(
                self.root_dir,
                full_config["task"]["ncf"]["checkpoint_ncf"],
                full_config["task"]["ncf"]["ncf_arch"],
                full_config["task"]["ncf"]["ncf_epoch"],
            )

            self.digit_vae, self.ncf = load_ncf(
                full_config["task"]["ncf"]["ncf_arch"],
                self.root_dir,
                path_checkpoint_vae,
                path_checkpoint_ndf,
                path_checkpoint_ncf,
                self.device,
            )
            self.digit_vae.eval()
            self.digit_vae.to(self.device)
            self.ncf.eval()
            self.ncf.to(self.device)

        if (self.ncf_info or self.tactile_info) or DEBUG:
            self.reset_ncf_buffers()

    def reset_digits_fifo(self):
        self.digit_left_fifo = torch.zeros(
            (self.num_actors, self.tactile_seq_length, self.tactile_info_embed_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.digit_right_fifo = torch.zeros(
            (self.num_actors, self.tactile_seq_length, self.tactile_info_embed_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def reset_eef_fifo(self):
        self.eef_fifo = torch.zeros(
            (self.num_actors, self.tactile_seq_length, 7),
            dtype=torch.float32,
            device=self.device,
        )
        self.eef_fifo[:, :, -1] = 1.0

    def reset_ncf_buffers(self):
        self.reset_digits_fifo()
        self.reset_eef_fifo()

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls, grad_norms):
        self.writer.add_scalar(
            "performance/RLTrainFPS",
            self.agent_steps / self.rl_train_time,
            self.agent_steps,
        )
        self.writer.add_scalar(
            "performance/EnvStepFPS",
            self.agent_steps / self.data_collect_time,
            self.agent_steps,
        )

        self.writer.add_scalar(
            "losses/actor_loss", torch.mean(a_losses).item(), self.agent_steps
        )
        self.writer.add_scalar(
            "losses/bounds_loss", torch.mean(b_losses).item(), self.agent_steps
        )
        self.writer.add_scalar(
            "losses/critic_loss", torch.mean(c_losses).item(), self.agent_steps
        )
        self.writer.add_scalar(
            "losses/entropy", torch.mean(entropies).item(), self.agent_steps
        )

        self.writer.add_scalar("info/last_lr", self.last_lr, self.agent_steps)
        self.writer.add_scalar("info/e_clip", self.e_clip, self.agent_steps)
        self.writer.add_scalar("info/kl", torch.mean(kls).item(), self.agent_steps)
        self.writer.add_scalar(
            "info/grad_norms", torch.mean(grad_norms).item(), self.agent_steps
        )

        for k, v in self.extra_info.items():
            self.writer.add_scalar(f"{k}", v, self.agent_steps)

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.obs_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.obs_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.train()

    def _get_ncf_output(self, obs_dict):
        if self.ncf_use_gt:
            contact_gt = obs_dict["gt_extrinsic_contact"]
            contact = contact_gt
            idx_query = np.arange(contact.shape[1])

        else:
            self._add_to_ncf_buffers(obs_dict)
            ee_pose, ee_t1, ee_t2 = self.ncf_dataloader.get_ee_sequence(self.eef_fifo)

            digits_emb_left = self.digit_left_fifo.reshape(
                -1, self.tactile_seq_length * self.tactile_info_embed_dim
            )
            digits_emb_right = self.digit_right_fifo.reshape(
                -1, self.tactile_seq_length * self.tactile_info_embed_dim
            )

            (
                shape_code,
                ncf_query_points,
                idx_query,
            ) = self.ncf_dataloader.get_pointcloud_data()

            inputs_ncf = {
                "digit_emb_left": digits_emb_left,
                "digit_emb_right": digits_emb_right,
                "ee_pose": ee_pose.to(self.device),
                "ee_pose_1": ee_t1.to(self.device),
                "ee_pose_2": ee_t2.to(self.device),
                "shape_emb": shape_code.to(self.device),
                "ncf_query_points": ncf_query_points.to(self.device),
            }
            with torch.no_grad():
                contact = self.ncf(inputs_ncf)

            d_mug_cupholder = (
                torch.norm(self.env.nut_pos - self.env.bolt_pos, dim=1)
            ) > 0.13
            contact[d_mug_cupholder] = 0.0
            contact = torch.where(contact > 0.4, 1.0, 0.0)
            contact_gt = contact.clone()

        if DEBUG:
            idx = 0
            object_pose = torch.cat(
                (self.env.nut_pos[idx], self.env.nut_quat[idx]), dim=0
            ).unsqueeze(0)
            cupholder_pose = torch.cat(
                (self.env.bolt_pos[idx], self.env.bolt_quat[idx]), dim=0
            ).unsqueeze(0)
            # p = contact[idx].cpu().numpy()
            p = np.zeros((contact[idx].shape[0]))
            p[idx_query] = contact[idx].cpu().numpy()
            # p[p > 0.3] = 1.0

            p_gt = contact_gt[idx].cpu().numpy()

            # img = self.ncf_viz.show_scene(
            #     object_pose, cupholder_pose, p_gt, p_pred, idx_query, interactive=False
            # )
            img_gt, img_pred = self.ncf_viz.show_gt_pred(
                object_pose,
                cupholder_pose,
                p_gt,
                p,
            )
            img = np.concatenate((img_gt, img_pred), axis=1)
            cv2.imshow("ncf", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return contact_gt, contact, idx_query

    def get_policy_inputs(self, obs_dict):
        if self.tactile_info:
            self._add_to_ncf_buffers(obs_dict)
            digits_emb_left = self.digit_left_fifo.reshape(
                -1, self.tactile_seq_length * self.tactile_info_embed_dim
            )
            digits_emb_right = self.digit_right_fifo.reshape(
                -1, self.tactile_seq_length * self.tactile_info_embed_dim
            )

            digits_emb = torch.cat([digits_emb_left, digits_emb_right], dim=-1)

            aug_obs = torch.cat([obs_dict["obs"], digits_emb], dim=-1)
            processed_obs = self.obs_mean_std(aug_obs)

        else:
            aug_obs = obs_dict["obs"]
            processed_obs = self.obs_mean_std(obs_dict["obs"])

        return processed_obs, aug_obs

    def model_act(self, obs_dict):
        processed_obs, aug_pos = self.get_policy_inputs(obs_dict)
        point_cloud = None

        if self.priv_info:
            ncf_priv_info, pred_contact, _ = self._get_ncf_output(obs_dict)
            point_cloud = obs_dict["pointclouds_t"]
            ncf_mask = torch.where(ncf_priv_info > 0.1, 1.0, 0.0)
            ncf_mask = ncf_mask.repeat(3, 1, 1).permute(1, 2, 0)
            point_cloud = point_cloud * ncf_mask
            obs_dict["pointclouds_t"] = point_cloud

            if self.normalize_point_cloud:
                point_cloud = self.point_cloud_mean_std(
                    point_cloud.reshape(-1, 3)
                ).reshape((processed_obs.shape[0], -1, 3))

        input_dict = {
            "obs": processed_obs,
            "priv_point_cloud": point_cloud,
        }

        res_dict = self.model.act(input_dict)
        res_dict["values"] = self.value_mean_std(res_dict["values"], True)
        return res_dict, aug_pos

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        self.reset_digits_fifo()
        self.agent_steps = (
            self.batch_size if not self.multi_gpu else self.batch_size * self.rank_size
        )

        if self.multi_gpu:
            torch.cuda.set_device(self.rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            (
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                grad_norms,
            ) = self.train_epoch()

            self.storage.data_dict = None

            (
                a_losses,
                b_losses,
                c_losses,
                entropies,
                kls,
                grad_norms,
            ) = multi_gpu_aggregate_stats(
                [a_losses, b_losses, c_losses, entropies, kls, grad_norms]
            )
            mean_rewards, mean_lengths, mean_success = multi_gpu_aggregate_stats(
                [
                    torch.Tensor([self.episode_rewards.get_mean()])
                    .float()
                    .to(self.device),
                    torch.Tensor([self.episode_lengths.get_mean()])
                    .float()
                    .to(self.device),
                    torch.Tensor([self.episode_success.get_mean()])
                    .float()
                    .to(self.device),
                ]
            )
            for k, v in self.extra_info.items():
                if type(v) is not torch.Tensor:
                    v = torch.Tensor([v]).float().to(self.device)
                self.extra_info[k] = multi_gpu_aggregate_stats(v[None].to(self.device))

            if not self.multi_gpu or (self.multi_gpu and self.rank == 0):
                all_fps = self.agent_steps / (time.time() - _t)
                last_fps = (
                    self.batch_size
                    if not self.multi_gpu
                    else self.batch_size * self.rank_size
                ) / (time.time() - _last_t)
                _last_t = time.time()
                info_string = (
                    f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                    f"Last FPS: {last_fps:.1f} | "
                    f"Collect Time: {self.data_collect_time / 60:.1f} min | "
                    f"Train RL Time: {self.rl_train_time / 60:.1f} min | "
                    f"Current Best: {self.best_rewards:.2f}"
                )
                print(info_string)

                self.write_stats(
                    a_losses, c_losses, b_losses, entropies, kls, grad_norms
                )
                self.writer.add_scalar(
                    "episode_rewards/step", mean_rewards, self.agent_steps
                )
                self.writer.add_scalar(
                    "episode_lengths/step", mean_lengths, self.agent_steps
                )
                self.writer.add_scalar(
                    "episode_success/step", mean_success, self.agent_steps
                )
                checkpoint_name = f"ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}"

                if self.save_freq > 0:
                    if (self.epoch_num % self.save_freq == 0) and (
                        mean_rewards <= self.best_rewards
                    ):
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                        self.save(os.path.join(self.nn_dir, f"last"))

                if (
                    mean_rewards > self.best_rewards
                    and self.agent_steps >= self.save_best_after
                    and mean_rewards != 0.0
                ):
                    print(f"save current best reward: {mean_rewards:.2f}")
                    # remove previous best file
                    prev_best_ckpt = os.path.join(
                        self.nn_dir, f"best_reward_{self.best_rewards:.2f}.pth"
                    )
                    if os.path.exists(prev_best_ckpt):
                        os.remove(prev_best_ckpt)
                    self.best_rewards = mean_rewards
                    self.save(
                        os.path.join(self.nn_dir, f"best_reward_{mean_rewards:.2f}")
                    )

        print("max steps achieved")

    def save(self, name):
        weights = {
            "model": self.model.state_dict(),
        }
        if self.obs_mean_std:
            weights["running_mean_std"] = self.obs_mean_std.state_dict()
        if self.value_mean_std:
            weights["value_mean_std"] = self.value_mean_std.state_dict()
        if self.point_cloud_mean_std:
            weights["point_cloud_mean_std"] = self.point_cloud_mean_std.state_dict()
        torch.save(weights, f"{name}.pth")

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        for k in list(checkpoint["model"].keys()):
            print(k)

        self.model.load_state_dict(checkpoint["model"])
        self.obs_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(
                checkpoint["point_cloud_mean_std"]
            )

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input:
            self.obs_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(
                checkpoint["point_cloud_mean_std"]
            )
        self.played_games = 0

    def test(self):
        num_episodes = 5

        self.metrics = {}
        self.metrics["success"] = []
        self.metrics["steps"] = []
        self.metrics["dist_mug_cupholder"] = []
        self.metrics["error_aligment_keypoints"] = []
        self.metrics["error_quat"] = []
        self.metrics["mug_close_to_cupholder"] = []
        self.metrics["mug_oriented"] = []
        self.metrics["mean_time_complete_task"] = []
        imgs_env = []

        for n in range(num_episodes):
            self.set_eval()
            obs_dict = self.env.reset()
            self.reset_digits_fifo()

            while True:
                processed_obs, _ = self.get_policy_inputs(obs_dict)
                point_cloud = None

                if self.priv_info:
                    ncf_priv_info, pred_contact, _ = self._get_ncf_output(obs_dict)
                    point_cloud = obs_dict["pointclouds_t"]
                    ncf_mask = torch.where(ncf_priv_info > 0.1, 1.0, 0.0)
                    ncf_mask = ncf_mask.repeat(3, 1, 1).permute(1, 2, 0)
                    point_cloud = point_cloud * ncf_mask
                    obs_dict["pointclouds_t"] = point_cloud

                    if self.normalize_point_cloud:
                        point_cloud = self.point_cloud_mean_std(
                            point_cloud.reshape(-1, 3)
                        ).reshape((processed_obs.shape[0], -1, 3))

                input_dict = {
                    "obs": processed_obs,
                    "priv_point_cloud": point_cloud,
                }

                # if DEBUG:
                #     self._get_ncf_output(obs_dict)

                mu = self.model.act_inference(input_dict)
                mu = torch.clamp(mu, -1.0, 1.0)
                obs_dict, r, done, info = self.env.step(mu)

    def train_epoch(self):
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += time.time() - _t
        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls, grad_norms = [], [], []
        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                # value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                #     returns, actions, obs, priv_info = self.storage[i]

                (
                    value_preds,
                    old_action_log_probs,
                    advantage,
                    old_mu,
                    old_sigma,
                    returns,
                    actions,
                    obs,
                    _,
                    point_cloud_info,
                ) = self.storage[i]

                obs = self.obs_mean_std(obs)
                if self.priv_info:
                    if self.normalize_point_cloud:
                        point_cloud_info = self.point_cloud_mean_std(
                            point_cloud_info.reshape(-1, 3)
                        ).reshape((obs.shape[0], -1, 3))

                batch_dict = {
                    "prev_actions": actions,
                    "obs": obs,
                    "priv_point_cloud": point_cloud_info,
                    # 'priv_info': priv_info,
                }
                res_dict = self.model(batch_dict)
                action_log_probs = res_dict["prev_neglogp"]
                values = res_dict["values"]
                entropy = res_dict["entropy"]
                mu = res_dict["mus"]
                sigma = res_dict["sigmas"]

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(
                    ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
                )
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(
                    -self.e_clip, self.e_clip
                )
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = torch.zeros_like(mu)
                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]
                ]

                loss = (
                    a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef
                )

                self.optimizer.zero_grad()
                loss.backward()

                if self.multi_gpu:
                    # batch all_reduce ops: see https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []
                    for param in self.model.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset : offset + param.numel()].view_as(
                                    param.grad.data
                                )
                                / self.rank_size
                            )
                            offset += param.numel()

                grad_norms.append(
                    torch.norm(
                        torch.cat([p.reshape(-1) for p in self.model.parameters()])
                    )
                )

                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_norm
                    )
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.rank_size
            kls.append(av_kls)

            self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            if self.multi_gpu:
                lr_tensor = torch.tensor([self.last_lr], device=self.device)
                dist.broadcast(lr_tensor, 0)
                lr = lr_tensor.item()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.last_lr

        self.rl_train_time += time.time() - _t
        return a_losses, c_losses, b_losses, entropies, kls, grad_norms

    def play_steps(self):
        for n in range(self.horizon_length):
            res_dict, aug_pos = self.model_act(self.obs)

            # collect o_t
            # self.storage.update_data("obses", n, self.obs["obs"])
            self.storage.update_data("obses", n, aug_pos)
            if self.ncf_info:
                self.storage.update_data(
                    "point_cloud_info", n, self.obs["pointclouds_t"]
                )
            # self.storage.update_data('priv_info', n, self.obs['priv_info'])
            for k in ["actions", "neglogpacs", "values", "mus", "sigmas"]:
                self.storage.update_data(k, n, res_dict[k])
            # do env step
            actions = torch.clamp(res_dict["actions"], -1.0, 1.0)

            # render() is called during env.step()
            # to save time, save gif only per gif_save_every_n steps
            # 1 step = #gpu * #envs agent steps
            record_frame = False
            # if self.gif_frame_counter % self.gif_save_every_n < self.gif_save_length:
            #     record_frame = True
            # record_frame = record_frame and int(os.getenv('LOCAL_RANK', '0')) == 0
            # self.env.enable_camera_sensors = record_frame
            # self.gif_frame_counter += 1

            self.obs, rewards, self.dones, infos = self.env.step(actions)

            if record_frame:
                self.gif_frames.append(self.env.capture_frame())
                # add frame to GIF
                if len(self.gif_frames) == self.gif_save_length:
                    frame_array = np.array([f["color"] for f in self.gif_frames])[
                        None
                    ]  # add batch axis
                    self.writer.add_video(
                        "rollout_gif",
                        frame_array,
                        global_step=self.agent_steps,
                        dataformats="NTHWC",
                        fps=20,
                    )
                    self.writer.flush()
                    self.gif_frames.clear()

            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.storage.update_data("dones", n, self.dones)
            shaped_rewards = 0.01 * rewards.clone()
            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += (
                    self.gamma
                    * res_dict["values"]
                    * infos["time_outs"].unsqueeze(1).float()
                )
            self.storage.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            assert isinstance(infos, dict), "Info Should be a Dict"
            self.extra_info = {}
            for k, v in infos.items():
                # only log scalars
                if (
                    isinstance(v, float)
                    or isinstance(v, int)
                    or (isinstance(v, torch.Tensor) and len(v.shape) == 0)
                ):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict, aug_pos = self.model_act(self.obs)
        last_values = res_dict["values"]

        self.agent_steps = (
            (self.agent_steps + self.batch_size)
            if not self.multi_gpu
            else self.agent_steps + self.batch_size * self.rank_size
        )
        self.storage.computer_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict["returns"]
        values = self.storage.data_dict["values"]
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict["values"] = values
        self.storage.data_dict["returns"] = returns

    def _digit_encode(self, images):
        images = images.permute(0, 1, 4, 2, 3)
        with torch.no_grad():
            images_hat, digit_embeddings = self.digit_vae(images)

        return digit_embeddings.unsqueeze(1), images_hat

    def _add_to_ncf_buffers(self, obs_dict):
        if obs_dict["obs"].sum() != 0:
            if self.eef_fifo[:, :, 0:3].sum() == 0:
                self.eef_fifo = (
                    obs_dict["obs"][:, 0:7]
                    .unsqueeze(1)
                    .repeat(1, self.tactile_seq_length, 1)
                )
            else:
                self.eef_fifo = add_to_fifo(
                    self.eef_fifo, obs_dict["obs"][:, 0:7].unsqueeze(1)
                )

        digits_imgs_left = obs_dict["digits_left"]
        digits_imgs_right = obs_dict["digits_right"]

        digits_emb_left, left_hat = self._digit_encode(digits_imgs_left)
        digits_emb_right, right_hat = self._digit_encode(digits_imgs_right)

        self.digit_left_fifo = add_to_fifo(self.digit_left_fifo, digits_emb_left)
        self.digit_right_fifo = add_to_fifo(self.digit_right_fifo, digits_emb_right)


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma / p0_sigma + 1e-5)
    c2 = (p0_sigma**2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma**2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


def load_digit_autoencoder(root_dir, path_checkpoint):
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


def load_ndf(root_dir, path_checkpoint):
    from algo2.ncf.ndf.ndf import NDF

    path_checkpoint = os.path.join(root_dir, path_checkpoint)
    ndf = NDF(latent_dim=256, return_features=True, sigmoid=True)
    checkpoint = torch.load(path_checkpoint)
    ndf_weights = ndf.state_dict()

    for key in ndf_weights.keys():
        ndf_weights[key] = checkpoint[key]
    ndf.load_state_dict(ndf_weights)
    for param in ndf.parameters():
        param.requires_grad = False

    return ndf


def load_ncf(
    arch,
    root_dir,
    path_checkpoint_vae,
    path_checkpoint_ndf,
    path_checkpoint_ncf,
    device_sim,
):
    from algo2.ncf.ncf.ncf_mlp import NCF as NCF_mlp
    from algo2.ncf.ncf.ncf_transformer import NCF as NCF_transformer
    from algo2.ncf.config.config import NCF_Params
    from algo2.ncf.pipeline import NCF_Pipeline

    cfg = NCF_Params()
    digit_encoder = load_digit_autoencoder(root_dir, path_checkpoint_vae)
    ndf_model = load_ndf(root_dir, path_checkpoint_ndf)

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

    return pipeline.digit_vae, pipeline.ncf

    # checkpoint = torch.load(path_checkpoint_ncf)
    # ncf_weights = ncf_model.state_dict()

    # for key in ncf_weights.keys():
    #     ncf_weights[key] = checkpoint["state_dict"]["ncf." + key]
    # ncf_model.load_state_dict(ncf_weights)
    # for param in ncf_model.parameters():
    #     param.requires_grad = False

    # return digit_encoder, ncf_model


# def load_ncf_old(
#     arch,
#     root_dir,
#     path_checkpoint_vae,
#     path_checkpoint_ndf,
#     path_checkpoint_ncf,
#     device_sim,
#     device_ndf,
#     device_ncf,
# ):
#     from algo.ncf.ncf.ncf_mlp import NCF as NCF_mlp
#     from algo.ncf.ncf.ncf_transformer import NCF as NCF_transformer
#     from algo.ncf.pipeline import NCF_Pipeline
#     from algo.ncf.config.config import NCF_Params

#     cfg = NCF_Params(ncf_arch=arch)

#     digit_encoder = load_digit_autoencoder(root_dir, path_checkpoint_vae).to(device_sim)
#     ndf_model = load_ndf(root_dir, path_checkpoint_ndf, device_ndf)
#     # ndf_model = torch.nn.DataParallel(ndf_model, device_ids=[1, 2, 3])
#     ndf_model.to(device_ndf)

#     if arch == "mlp":
#         ncf_model = NCF_mlp(cfg)
#     elif arch == "transformer":
#         ncf_model = NCF_transformer(cfg)
#     else:
#         raise NotImplementedError

#     checkpoint_ncf = os.path.join(root_dir, path_checkpoint_ncf)
#     ncf_model.load_state_dict(torch.load(checkpoint_ncf, map_location=device_ncf))
#     if not DEBUG:
#         ncf_model = torch.nn.DataParallel(ncf_model, device_ids=[2, 3])
#     ncf_model.to(device_ncf)
#     ncf_model.eval()
#     for param in ncf_model.parameters():
#         param.requires_grad = False
#     return digit_encoder, ndf_model, ncf_model

#     # ncf_model.to(device_ncf)
#     # checkpoint_ncf = os.path.join(root_dir, path_checkpoint_ncf)
#     # ncf_pipeline = NCF_Pipeline.load_from_checkpoint(
#     #     checkpoint_ncf,
#     #     cfg=cfg,
#     #     digit_vae=digit_encoder,
#     #     ndf=ndf_model,
#     #     ncf=ncf_model,
#     # )
#     # ncf_pipeline.eval()
#     # ncf_pipeline.to(device_ncf)
#     # for param in ncf_pipeline.parameters():
#     #     param.requires_grad = False

#     # return ncf_pipeline


# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
