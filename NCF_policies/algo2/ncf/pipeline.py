from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
        }

        pred_contact = self.ncf(inputs_ncf)

        # out = {
        #     "pred_contact": pred_contact,
        #     "images_hat_left": images_hat_left,
        #     "images_hat_right": images_hat_right,
        # }
        # return out

        return pred_contact

    def training_step(self, batch, batch_idx):
        out_ncf = self.forward(batch)
        pred_contact_oe = out_ncf["pred_contact"]
        loss = self.compute_loss(pred_contact_oe, batch)
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
