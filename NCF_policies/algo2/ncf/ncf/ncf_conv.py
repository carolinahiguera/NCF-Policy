import torch
import torch.nn as nn
import torch.nn.functional as F


class NCF(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.seq_len = cfg.seq_len

        # digit encoder
        self.digit_emb_dim = cfg.vae_params.latent_dim
        digits_emb_in_dim = self.digit_emb_dim * self.seq_len
        digits_emb_out_dim = 128
        digit_emb_mid_dim = 128
        self.digits_down = nn.Sequential(
            nn.Linear(digits_emb_in_dim, digit_emb_mid_dim),
            nn.ReLU(),
            nn.Linear(digit_emb_mid_dim, digits_emb_out_dim),
            nn.ReLU(),
        )

        # ncf model
        ncf_dim = 2049
        input_dim = ncf_dim + (digits_emb_out_dim * 2) + (7 * 3)
        self.bn_in = nn.BatchNorm1d(input_dim)
        self.network = nn.Sequential(
            nn.Conv1d(input_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        emb_ndf = inputs["emb_ndf"]
        images_left = inputs["digit_emb_left"]
        images_right = inputs["digit_emb_right"]
        ee_pose = inputs["ee_pose"][:, 0, [0, 1, 2, 4, 5, 6, 3]]
        ee_t1 = inputs["ee_pose_1"]
        ee_t2 = inputs["ee_pose_2"]
        contact_t1 = inputs["contact_oe_t1"].unsqueeze(dim=2)
        contact_t2 = inputs["contact_oe_t2"].unsqueeze(dim=2)

        images_left_down = self.digits_down(
            images_left.reshape(-1, self.seq_len * self.digit_emb_dim)
        )
        images_right_down = self.digits_down(
            images_right.reshape(-1, self.seq_len * self.digit_emb_dim)
        )
        digit_feat_all = torch.cat([images_left_down, images_right_down], dim=1)

        n_pts = emb_ndf.shape[1]
        ee_t0 = ee_pose[:, None, :].repeat(1, n_pts, 1)
        ee_t1 = ee_t1[:, None, :].repeat(1, n_pts, 1)
        ee_t2 = ee_t2[:, None, :].repeat(1, n_pts, 1)
        digit_feat_all = digit_feat_all[:, None, :].repeat(1, n_pts, 1)
        # x_in = torch.cat((emb_ndf, contact_t1, contact_t2, ee_t1, ee_t2), 2)
        x_in = torch.cat((emb_ndf, digit_feat_all, ee_t0, ee_t1, ee_t2), 2)
        x_in = x_in.permute(0, 2, 1)

        x_in = self.bn_in(x_in)
        out = self.network(x_in)
        out = out.squeeze(dim=1)
        return out
