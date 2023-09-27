# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size, with_last_activation=True):
        super(MLP, self).__init__()
        # use with_last_activation=False when we need the network to output raw values before activation
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        if not with_last_activation:
            layers.pop()
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# class ProprioAdaptTConv(nn.Module):
#     def __init__(self):
#         super(ProprioAdaptTConv, self).__init__()
#         self.channel_transform = nn.Sequential(
#             nn.Linear(16 + 16, 32),
#             nn.ReLU(inplace=True),
#             nn.Linear(32, 32),
#             nn.ReLU(inplace=True),
#         )
#         self.temporal_aggregation = nn.Sequential(
#             nn.Conv1d(32, 32, (9,), stride=(2,)),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(32, 32, (5,), stride=(1,)),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(32, 32, (5,), stride=(1,)),
#             nn.ReLU(inplace=True),
#         )
#         self.low_dim_proj = nn.Linear(32 * 3, 8)

#     def forward(self, x):
#         x = self.channel_transform(x)  # (N, 50, 32)
#         x = x.permute((0, 2, 1))  # (N, 32, 50)
#         x = self.temporal_aggregation(x)  # (N, 32, 3)
#         x = self.low_dim_proj(x.flatten(1))
#         return x


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop("actions_num")
        self.proprio_input_shape = kwargs.pop("proprio_input_shape")[0]
        self.units = kwargs.pop("actor_units")
        self.tactile_info = kwargs.pop("tactile_info")
        self.ncf_info = kwargs.pop("ncf_info")
        self.tactile_units = kwargs.pop("tactile_units")
        self.ncf_units = kwargs.pop("ncf_units")
        self.ncf_adapt_units = kwargs.pop("ncf_adapt_units")
        tactile_input_shape = kwargs.pop("tactile_input_shape")
        ncf_input_shape = kwargs.pop("ncf_input_shape")
        self.ncf_proprio_adapt = kwargs.pop("ncf_proprio_adapt")

        # mlp_input_shape = input_shape[0]
        out_size = self.units[-1]

        # self.priv_info = kwargs['priv_info']
        # self.priv_info_stage2 = kwargs['proprio_adapt']

        if self.tactile_info:
            self.tactile_mlp = MLP(
                units=self.tactile_units, input_size=tactile_input_shape[0]
            )
            mlp_input_shape = self.proprio_input_shape + self.tactile_units[-1]
        elif self.ncf_info:
            # self.ncf_mlp = MLP(units=self.ncf_units, input_size=ncf_input_shape[0])
            self.ncf_mlp = MLP(units=self.ncf_units, input_size=3)

            if self.ncf_proprio_adapt:
                # self.adapt_ncf = MLP(
                #     units=self.ncf_units, input_size=ncf_input_shape[0]
                # )
                self.adapt_ncf = MLP(units=self.ncf_adapt_units, input_size=3)

            mlp_input_shape = self.proprio_input_shape + self.ncf_units[-1]
        else:
            mlp_input_shape = self.proprio_input_shape

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
            requires_grad=True,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            "neglogpacs": -distr.log_prob(selected_action).sum(
                1
            ),  # self.neglogp(selected_action, mu, sigma, logstd),
            "values": value,
            "actions": selected_action,
            "mus": mu,
            "sigmas": sigma,
        }
        return result

    def ncf_pointcloud_embedding(self, point_cloud_info: torch.Tensor) -> torch.Tensor:
        pcs = self.ncf_mlp(point_cloud_info)
        pcs_filtered = torch.ones_like(pcs) * -1000.0

        a = torch.nonzero(point_cloud_info)
        if len(a) == 0:
            # return torch.zeros((pcs.shape[0], pcs.shape[2])).float().to(pcs.device)
            return torch.zeros((pcs.shape[0], pcs.shape[2])).float().to(pcs.device)
        b = a[:, [0, 1]].cpu().numpy()
        pcs_filtered[b[:, 0], b[:, 1]] = self.ncf_mlp(point_cloud_info)[
            b[:, 0], b[:, 1]
        ]
        emb = torch.max(pcs_filtered, 1)[0]
        emb[emb == -1000.0] = 0.0
        return emb

        # pc_idx = a[:, 0].unique()
        # if len(pc_idx) > 2:
        #     print("aqui")
        # for i in pc_idx:
        #     pc_contact_points = a[torch.where(a[:, 0] == i)[0], 1].unique()
        #     emb[i] = torch.max(pcs[i, pc_contact_points], 0)[0]

        # return torch.max(pcs, 1)[0]
        # return emb

    def ncf_adapt_pointcloud_embedding(
        self, point_cloud_info: torch.Tensor
    ) -> torch.Tensor:
        pcs = self.adapt_ncf(point_cloud_info)
        return torch.max(pcs, 1)[0]

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        # obs = obs_dict["obs"]
        # tactile_input = obs_dict["tactile_inputs"]
        # ncf_gt_input = obs_dict["ncf_gt_inputs"]
        # ncf_pred_input = obs_dict["ncf_pred_inputs"]
        obs = obs_dict["obs"][:, 0 : self.proprio_input_shape]
        ncf_emb, ncf_emb_gt = None, None

        if self.tactile_info:
            tactile_input = obs_dict["obs"][:, self.proprio_input_shape :]
            tactile = self.tactile_mlp(tactile_input)
            obs = torch.cat([obs, tactile], dim=-1)

        elif self.ncf_info:
            priv_point_cloud = obs_dict["priv_point_cloud"]
            ncf_emb_gt = self.ncf_pointcloud_embedding(priv_point_cloud)

            if self.ncf_proprio_adapt:
                ncf_point_cloud = obs_dict["ncf_point_cloud"]
                ncf_emb = self.ncf_adapt_pointcloud_embedding(ncf_point_cloud)
                obs = torch.cat([obs, ncf_emb], dim=-1)
            else:
                obs = torch.cat([obs, ncf_emb_gt], dim=-1)

        x = self.actor_mlp(obs)
        value = self.value(x)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, mu * 0 + sigma, value, ncf_emb, ncf_emb_gt

    def forward(self, input_dict):
        prev_actions = input_dict.get("prev_actions", None)
        rst = self._actor_critic(input_dict)
        mu, logstd, value, extrin, extrin_gt = rst
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            "prev_neglogp": torch.squeeze(prev_neglogp),
            "values": value,
            "entropy": entropy,
            "mus": mu,
            "sigmas": sigma,
            "extrin": extrin,
            "extrin_gt": extrin_gt,
        }
        return result
