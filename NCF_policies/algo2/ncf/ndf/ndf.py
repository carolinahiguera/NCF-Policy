import torch
import torch.nn as nn

EPS = 1e-6


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


class NDF(nn.Module):
    def __init__(
        self,
        latent_dim,
        sigmoid=True,
        return_features=False,
        acts="all",
        scaling=10.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.scaling = (
            scaling  # scaling up the point cloud/query points to be larger helps
        )

        self.model_type = "pointnet"
        self.encoder = VNN_ResnetPointnet(c_dim=latent_dim)

    def forward(self, input):
        enc_in = input * self.scaling
        enc_out = self.encoder(enc_in)
        enc_out = torch.flatten(enc_out, start_dim=1)
        return enc_out


class VNN_ResnetPointnet(nn.Module):
    """
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    """

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output

        self.conv_pos = VNLinearLeakyReLU(
            3, 128, negative_slope=0.2, share_nonlinearity=False, use_batchnorm=False
        )
        self.fc_pos = VNLinear(128, 2 * hidden_dim)
        self.block_0 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)

        self.actvn_c = VNLeakyReLU(
            hidden_dim, negative_slope=0.2, share_nonlinearity=False
        )
        self.pool = meanpool

        if meta_output == "invariant_latent":
            self.std_feature = VNStdFeature(
                c_dim, dim=3, normalize_frame=True, use_batchnorm=False
            )
        elif meta_output == "invariant_latent_linear":
            self.std_feature = VNStdFeature(
                c_dim, dim=3, normalize_frame=True, use_batchnorm=False
            )
            self.vn_inv = VNLinear(c_dim, 3)
        elif meta_output == "equivariant_latent_linear":
            self.vn_inv = VNLinear(c_dim, 3)

    def forward(self, p):
        batch_size = p.size(0)
        p = p.unsqueeze(1).transpose(2, 3)
        # mean = get_graph_mean(p, k=self.k)
        # mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        feat = self.get_graph_feature_cross(p, k=self.k)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)

        net = self.fc_pos(net)

        net = self.block_0(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))

        if self.meta_output == "invariant_latent":
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == "invariant_latent_linear":
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std
        elif self.meta_output == "equivariant_latent_linear":
            c_std = self.vn_inv(c)
            return c, c_std

        return c

    def get_graph_feature_cross(self, x, k=20, idx=None):
        batch_size = x.size(0)
        num_points = x.size(3)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        # device = torch.device("cuda") if USE_CUDA else torch.device("cpu")
        device = x.device

        idx_base = (
            torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        )

        idx = idx + idx_base

        idx = idx.view(-1)

        _, num_dims, _ = x.size()
        num_dims = num_dims // 3

        x = x.transpose(
            2, 1
        ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims, 3)
        x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
        cross = torch.cross(feature, x, dim=-1)

        feature = (
            torch.cat((feature - x, x, cross), dim=3)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )

        return feature


class VNLinearLeakyReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dim=5,
        share_nonlinearity=False,
        use_batchnorm=True,
        negative_slope=0.2,
    ):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

        # Conv
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm == True:
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        # LeakyReLU
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        # Conv
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        # InstanceNorm
        if self.use_batchnorm == True:
            p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        norm = torch.sqrt((x * x).sum(2))
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn

        return x


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        x_out = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNStdFeature(nn.Module):
    def __init__(
        self,
        in_channels,
        dim=4,
        normalize_frame=False,
        share_nonlinearity=False,
        use_batchnorm=True,
    ):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        self.share_nonlinearity = share_nonlinearity
        self.use_batchnorm = use_batchnorm

        self.vn1 = VNLinearLeakyReLU(
            in_channels,
            in_channels // 2,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            use_batchnorm=use_batchnorm,
        )
        self.vn2 = VNLinearLeakyReLU(
            in_channels // 2,
            in_channels // 4,
            dim=dim,
            share_nonlinearity=share_nonlinearity,
            use_batchnorm=use_batchnorm,
        )
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels // 4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels // 4, 3, bias=False)

    def forward(self, x):
        """
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        """
        z0 = self.vn1(x)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:, 0, :]
            # u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdim=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdim=True) * u1
            # u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdim=True))
            u2 = v2 / (v2_norm + EPS)

            # compute the cross product of the two output vectors
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum("bijm,bjkm->bikm", x, z0)
        elif self.dim == 3:
            x_std = torch.einsum("bij,bjk->bik", x, z0)
        elif self.dim == 5:
            x_std = torch.einsum("bijmn,bjkmn->bikmn", x, z0)

        return x_std, z0


# Resnet Blocks
class VNResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = VNLinear(size_in, size_h)
        self.fc_1 = VNLinear(size_h, size_out)
        self.actvn_0 = VNLeakyReLU(
            size_in, negative_slope=0.2, share_nonlinearity=False
        )
        self.actvn_1 = VNLeakyReLU(size_h, negative_slope=0.2, share_nonlinearity=False)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = VNLinear(size_in, size_out)
        # Initialization
        nn.init.zeros_(self.fc_1.map_to_feat.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn_0(x))
        dx = self.fc_1(self.actvn_1(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
