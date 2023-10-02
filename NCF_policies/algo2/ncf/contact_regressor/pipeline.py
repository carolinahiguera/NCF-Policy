from typing import Any
import pytorch_lightning as pl

import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2
import numpy as np

# import trimesh
# import vedo
# from vedo import Points, show, trimesh2vedo
from scipy.spatial.transform import Rotation as R

DEBUG = False


def show_obj_probabilities(pc, mesh_file, dist, interactive=False):
    object_3d_cam = dict(
        position=(-7.54483e-3, -0.0849045, -0.250212),
        focal_point=(-4.82255e-3, -2.87705e-3, 0),
        viewup=(0.580505, -0.775588, 0.247946),
        distance=0.263329,
        clipping_range=(0.174192, 0.376170),
    )

    pc = Points(pc, r=15)
    pc = pc.cmap(
        "plasma",
        dist,
        vmin=0.0,
        vmax=1.0,
    )

    object_trimesh = trimesh.load_mesh(mesh_file)
    T = np.eye(4)
    T[0:3, 0:3] = R.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix()
    object_trimesh = object_trimesh.apply_transform(T)
    mesh_vedo = trimesh2vedo(object_trimesh).clone()
    # mesh_vedo.subdivide(n=10, method=2)
    mesh_vedo = mesh_vedo.interpolate_data_from(
        pc, n=5, on="points", kernel="gaussian"
    ).cmap("plasma", vmin=0.0, vmax=1.0)

    if interactive:
        show([mesh_vedo, pc], axes=1, camera=object_3d_cam)
    else:
        plt = vedo.Plotter(offscreen=True, size=(500, 500))
        plt.show([mesh_vedo, pc]).screenshot("contact.png").close()
        print("")
        # img = show(
        #     [mesh_vedo, pc],
        #     axes=1,
        #     camera=object_3d_cam,
        #     interactive=False,
        #     offscreen=True,
        # ).screenshot("contact.png")
        # ).screenshot(asarray=True)
        # return img


class NCF_Pipeline(pl.LightningModule):
    def __init__(self, cfg_ncf, cfg_train, digit_vae, ncf, ndf):
        super().__init__()

        self.cfg_ncf = cfg_ncf
        self.cfg_train = cfg_train
        self.seq_len = cfg_ncf.seq_len
        self.digit_emb_dim = cfg_ncf.vae_params.latent_dim

        self.digit_vae = digit_vae
        self.ncf = ncf
        self.ndf = ndf

        self.mse_loss = nn.MSELoss(reduction="none")
        self.bce_loss = nn.BCELoss(reduction="none")
        self.l1_loss = nn.L1Loss(reduction="none")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg_train.lr)

    # def configure_optimizers(self):
    #     # Using a scheduler is optional but can be helpful.
    #     # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
    #     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     #     optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5
    #     # )

    #     lr = (
    #         self.cfg_train.lr
    #         if self.cfg_train.ncf_arch == "transformer"
    #         else self.cfg_train.lr * 0.5
    #     )

    #     optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    #     if self.cfg_train.ncf_arch == "transformer":
    #         scheduler = torch.optim.lr_scheduler.StepLR(
    #             optimizer, step_size=5, gamma=0.1
    #         )
    #     else:
    #         scheduler = torch.optim.lr_scheduler.StepLR(
    #             optimizer, step_size=2, gamma=0.1
    #         )

    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #         },
    #     }
    #     # return {"optimizer": optimizer, "monitor": "val_loss"}

    def _get_digit_embeddings(self, images):
        batch_size = images.shape[0]
        seq_len = images.shape[1]
        images_hat = torch.zeros_like(images)

        digit_embeddings = torch.zeros(self.seq_len, batch_size, self.digit_emb_dim).to(
            images.device
        )

        with torch.no_grad():
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

    def compute_loss(self, pred, batch):
        gt = batch["p_contact_t0"]

        if self.cfg_train.loss == "mse":
            loss = self.compute_loss_mse(gt, pred)
        elif self.cfg_train.loss == "l1":
            loss = self.compute_loss_l1(gt, pred)
        elif self.cfg_train.loss == "bce":
            loss = self.compute_loss_bce(gt, pred)
        elif self.cfg_train.loss == "classic":
            loss = self.compute_loss_classic(gt, pred)
        else:
            return NotImplementedError
        return loss

    def forward(self, batch):
        images_left = batch["digit_imgs_left"]
        images_right = batch["digit_imgs_right"]
        ndf_point_cloud = batch["ndf_point_cloud"]
        ncf_query_points = batch["ncf_query_points"]

        # get digit embeddings
        embs_left, images_hat_left = self._get_digit_embeddings(images_left)
        embs_right, images_hat_right = self._get_digit_embeddings(images_right)

        with torch.no_grad():
            shape_emb = self.ndf(ndf_point_cloud)

        inputs_ncf = {
            "digit_emb_left": embs_left,
            "digit_emb_right": embs_right,
            "ee_pose": batch["ee_pose"],
            "ee_pose_1": batch["ee_pose_1"],
            "ee_pose_2": batch["ee_pose_2"],
            "shape_emb": shape_emb,
            "ncf_query_points": ncf_query_points,
        }

        pred_contact = self.ncf(inputs_ncf)

        out = {
            "pred_contact": pred_contact,
            "images_hat_left": images_hat_left,
            "images_hat_right": images_hat_right,
        }
        return out

        # return pred_contact

    def plot_contact(self, pred_contact, batch, interactive):
        idx = 0
        gt = batch["p_contact_t0"][idx].cpu().numpy()
        idx_query = batch["idx_query"][idx].cpu().numpy()
        pred = pred_contact[idx].detach().cpu().numpy()
        pc = np.load(batch["pc_file"][idx])
        mesh_file = batch["mesh_file"][idx]

        gt_full = np.zeros((pc.shape[0]))
        gt_full[idx_query] = gt
        pred_full = np.zeros((pc.shape[0]))
        pred_full[idx_query] = pred

        gt_img = show_obj_probabilities(pc, mesh_file, gt_full, interactive)

        if not interactive:
            pred_img = show_obj_probabilities(pc, pred_full)
            img = cv2.hconcat([gt_img, pred_img])
            plt.imshow(img)
            plt.show()

    def training_step(self, batch, batch_idx):
        out_ncf = self.forward(batch)
        pred_contact_oe = out_ncf["pred_contact"]
        loss = self.compute_loss(pred_contact_oe, batch)
        batch_size = pred_contact_oe.shape[0]
        # print(loss.item())
        if DEBUG:
            self.plot_contact(pred_contact_oe, batch, interactive=False)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        return {"loss": loss, "pred_contact": pred_contact_oe}

    def validation_step(self, batch, batch_idx):
        out_ncf = self.forward(batch)
        pred_contact_oe = out_ncf["pred_contact"]
        loss = self.compute_loss(pred_contact_oe, batch)
        batch_size = pred_contact_oe.shape[0]
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
        )
        return {"loss": loss, "pred_contact": pred_contact_oe}
