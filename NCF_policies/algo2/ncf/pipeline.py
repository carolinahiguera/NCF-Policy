from typing import Any
import pytorch_lightning as pl
# from pytorch3d.loss import chamfer_distance
# from pytorch3d.ops import knn_points
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2
import numpy as np

# from utils.viz_3d import Viz3d

DEBUG = False


class NCF_Pipeline(pl.LightningModule):
    def __init__(self, cfg, digit_vae, ncf, ndf):
        super().__init__()

        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.digit_emb_dim = cfg.vae_params.latent_dim

        self.digit_vae = digit_vae
        self.ncf = ncf
        self.ndf = ndf

        self.mse_loss = nn.MSELoss(reduction="none")
        self.bce_loss = nn.BCELoss(reduction="none")
        self.l1_loss = nn.L1Loss(reduction="none")

        # if DEBUG:
        #     self.viz3d = Viz3d(cfg.viz3d_params)

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

    def configure_optimizers(self):
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
        # )

        lr = self.cfg.lr if self.cfg.ncf_arch == "transformer" else self.cfg.lr * 0.5

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if self.cfg.ncf_arch == "transformer":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.1
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=2, gamma=0.1
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
            },
        }
        # return {"optimizer": optimizer, "monitor": "val_loss"}

    def _get_digit_embeddings(self, images):
        batch_size = images.shape[0]
        seq_len = images.shape[1]
        images_hat = torch.zeros_like(images)

        digit_embeddings = torch.zeros(self.seq_len, batch_size, self.digit_emb_dim).to(
            images.device
        )
        for i in range(seq_len):
            s = images[:, i, :, :, :].unsqueeze(dim=1)
            self.digit_vae.eval()
            x_hat, emb = self.digit_vae(s)
            images_hat[:, i, :, :, :] = x_hat
            digit_embeddings[i, :, :] = emb

        return digit_embeddings.permute(1, 0, 2), images_hat

    def compute_loss_mse(self, gt, pred):
        loss = self.mse_loss(pred, gt)
        return loss.mean()

    def compute_loss_l1(self, gt, pred):
        loss = self.l1_loss(pred, gt)
        return loss.mean()

    def compute_loss_bce(self, gt, pred):
        pred = pred.squeeze()
        gt = gt.squeeze()
        loss = self.bce_loss(pred, gt)
        return loss.mean()

    def compute_loss_classic(self, gt, pred):
        label = gt.squeeze()
        # label = (label + 1) / 2.
        loss = (
            -1
            * (
                label * torch.log(pred + 1e-5)
                + (1 - label) * torch.log(1 - pred + 1e-5)
            ).mean()
        )
        return loss

    # def compute_loss_pc(self, gt, pred, batch):
    #     gt = gt.squeeze()
    #     pred = pred.squeeze()
    #     idx_query = batch["idx_query"].squeeze()
    #     pc_obj = batch["ref_point_cloud"].squeeze()
    #     idx_query_gt = idx_query[gt >= 0.1]
    #     idx_query_pred = idx_query[pred >= 0.1]
    #     pc_gt = pc_obj[idx_query_gt].unsqueeze(dim=0)
    #     pc_pred = pc_obj[idx_query_pred].unsqueeze(dim=0)

    #     if pc_gt.shape[1] > 0 and pc_pred.shape[1] > 0:
    #         gt2pred = knn_points(
    #             pc_gt, pc_pred, lengths1=None, lengths2=None, K=1, return_nn=True
    #         )
    #         gt2pred_dist = gt2pred.dists.sqrt()[:, :, 0]

    #         pred2gt = knn_points(
    #             pc_pred, pc_gt, lengths1=None, lengths2=None, K=1, return_nn=True
    #         )
    #         pred2gt_dist = pred2gt.dists.sqrt()[:, :, 0]

    #         # loss = (gt2pred_dist.max() + pred2gt_dist.max()) / 2.0
    #         loss = torch.max(pred2gt_dist.max(), gt2pred_dist.max())
    #     elif pc_gt.shape[1] == 0 and pc_pred.shape[1] == 0:
    #         loss = torch.tensor(0.0).to(pred.device)
    #     else:
    #         loss = torch.tensor(0.08).to(pred.device)
    #     return loss

    def compute_loss(self, pred, batch):
        gt = batch["p_contact_oe_t0"]

        if self.cfg.loss == "mse":
            loss = self.compute_loss_mse(gt, pred)
        elif self.cfg.loss == "l1":
            loss = self.compute_loss_l1(gt, pred)
        elif self.cfg.loss == "bce":
            loss = self.compute_loss_bce(gt, pred)
        elif self.cfg.loss == "classic":
            loss = self.compute_loss_classic(gt, pred)
        else:
            return NotImplementedError

        loss = loss * self.cfg.weight_loss
        return loss

    def forward(self, batch):
        ee_pose = batch["ee_pose"]
        embs_left = batch["digits_emb_left"]
        embs_right = batch["digits_emb_right"]
        ndf_point_cloud = batch["ndf_point_cloud"]
        ndf_query_point_cloud = batch["ndf_query_point_cloud"]
        # contact_oe_t1 = batch["p_contact_oe_t1"]
        # contact_oe_t2 = batch["p_contact_oe_t2"]

        # get digit embeddings
        # embs_left, images_hat_left = self._get_digit_embeddings(images_left)
        # embs_right, images_hat_right = self._get_digit_embeddings(images_right)

        # get ndf embeddings
        ndf_input = {}
        ndf_input["coords"] = ndf_query_point_cloud
        ndf_input["point_cloud"] = ndf_point_cloud
        out_ndf = self.ndf(ndf_input)

        inputs_ncf = {
            "digit_emb_left": embs_left,
            "digit_emb_right": embs_right,
            "ee_pose": ee_pose,
            "ee_pose_1": batch["ee_pose_1"],
            "ee_pose_2": batch["ee_pose_2"],
            "emb_ndf": out_ndf["features"],
            # "contact_oe_t1": contact_oe_t1,
            # "contact_oe_t2": contact_oe_t2,
        }

        pred_contact = self.ncf(inputs_ncf)

        # out = {
        #     "pred_contact": pred_contact,
        #     "images_hat_left": images_hat_left,
        #     "images_hat_right": images_hat_right,
        # }
        # return out

        return pred_contact

    def plot_contact(self, pred_contact, batch, interactive):
        idx = 0
        gt = batch["p_contact_oe_t0"][idx].cpu().numpy()
        idx_query = batch["idx_query"][idx].cpu().numpy()
        pred = pred_contact[idx].detach().cpu().numpy()
        pc = np.load(self.cfg.objects_assets + "/mug_pc.npy")

        gt_full = np.zeros((pc.shape[0]))
        gt_full[idx_query] = gt
        pred_full = np.zeros((pc.shape[0]))
        pred_full[idx_query] = pred

        gt_img = self.viz3d.show_obj_probabilities(pc, gt_full, interactive)

        if not interactive:
            pred_img = self.viz3d.show_obj_probabilities(pc, pred_full)
            img = cv2.hconcat([gt_img, pred_img])
            plt.imshow(img)
            plt.show()

    def training_step(self, batch, batch_idx):
        out_ncf = self.forward(batch)
        pred_contact_oe = out_ncf["pred_contact"]
        loss = self.compute_loss(pred_contact_oe, batch)
        # print(loss.item())
        if DEBUG:
            self.plot_contact(pred_contact_oe, batch, interactive=DEBUG)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        return {"loss": loss, "pred_contact_oe": pred_contact_oe}

    def validation_step(self, batch, batch_idx):
        out_ncf = self.forward(batch)
        pred_contact_oe = out_ncf["pred_contact"]
        loss = self.compute_loss(pred_contact_oe, batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        return {"loss": loss, "pred_contact_oe": pred_contact_oe}
