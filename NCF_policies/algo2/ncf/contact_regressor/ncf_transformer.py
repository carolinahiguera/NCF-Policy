import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerConv(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerConv, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # self.positional_encoding = PositionalEncoding(input_dim)

        self.conv_layer1 = TransformerConvAttentionLayer(
            input_dim, hidden_dim[0], num_heads
        )
        # self.conv_layer2 = TransformerConvAttentionLayer(
        #     hidden_dim[0], hidden_dim[1], num_heads
        # )
        self.reg = nn.Sequential(
            nn.Conv1d(hidden_dim[0], hidden_dim[1], 1),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim[1], 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x = self.positional_encoding(x)
        x = x.permute(0, 2, 1)  # Reshape to (batch, d, n)
        x = self.conv_layer1(x)
        # x = self.conv_layer2(x)
        x = self.reg(x)
        x = x.squeeze(1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class TransformerConvAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.2):
        super(TransformerConvAttentionLayer, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, 1)
        # self.conv2 = nn.Conv1d(hidden_dim, input_dim, 1)
        self.norm1 = nn.BatchNorm1d(input_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # self.attention = nn.MultiheadAttention(hidden_dim, num_heads)

        self.linear_q = nn.Conv1d(input_dim, input_dim, 1)
        self.linear_k = nn.Conv1d(input_dim, input_dim, 1)
        self.linear_v = nn.Conv1d(input_dim, input_dim, 1)

    def scaled_dot_product(self, q, k, v, mask=None):
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        values = values.permute(0, 2, 1)
        attention = attention.permute(0, 2, 1)
        return values, attention

    def forward(self, x):
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        attn_output, _ = self.scaled_dot_product(x_q, x_k, x_v)

        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm2(x)

        return x


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
        ncf_dim = 768 + 3
        input_dim = ncf_dim + (digits_emb_out_dim * 2) + (7 * 3)
        hidden_dim = [512, 128]
        self.bn_in = nn.BatchNorm1d(input_dim)
        self.network = TransformerConv(
            input_dim=input_dim, num_heads=2, hidden_dim=hidden_dim, num_layers=2
        )

    def forward(self, inputs):
        images_left = inputs["digit_emb_left"]
        images_right = inputs["digit_emb_right"]
        # ee_pose = inputs["ee_pose"][:, 0, [0, 1, 2, 4, 5, 6, 3]]
        ee_pose = inputs["ee_pose"][:, 0]
        ee_t1 = inputs["ee_pose_1"]
        ee_t2 = inputs["ee_pose_2"]
        shape_emb = inputs["shape_emb"]
        ncf_query_points = inputs["ncf_query_points"]

        images_left_down = self.digits_down(
            images_left.reshape(-1, self.seq_len * self.digit_emb_dim)
        )
        images_right_down = self.digits_down(
            images_right.reshape(-1, self.seq_len * self.digit_emb_dim)
        )
        digit_feat_all = torch.cat([images_left_down, images_right_down], dim=1)

        n_pts = ncf_query_points.shape[1]
        ee_t0 = ee_pose[:, None, :].repeat(1, n_pts, 1)
        ee_t1 = ee_t1[:, None, :].repeat(1, n_pts, 1)
        ee_t2 = ee_t2[:, None, :].repeat(1, n_pts, 1)
        digit_feat_all = digit_feat_all[:, None, :].repeat(1, n_pts, 1)

        shape_emb = shape_emb[:, None, :].repeat(1, n_pts, 1)
        # x_in = torch.cat((emb_ndf, contact_t1, contact_t2, ee_t1, ee_t2), 2)
        # x_in = torch.cat((emb_ndf, digit_feat_all, ee_t0, ee_t1, ee_t2), 2)
        x_in = torch.cat(
            (ncf_query_points, shape_emb, digit_feat_all, ee_t0, ee_t1, ee_t2), 2
        )

        out = self.network(x_in)

        return out
