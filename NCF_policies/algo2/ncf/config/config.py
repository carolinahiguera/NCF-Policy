from dataclasses import dataclass, field
import yaml
import os
from typing import List


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
    seq_len: int = 5
    source_data: str = "sim"
    hidden_size_ncf: int = 128
    pc_subsample: float = 1.0
    vae_params = Digit_VAE_Params()


# @dataclass
# class Digit_VAE_Params:
#     root_path: str = os.path.abspath(os.path.join(".."))
#     image_size: int = 64
#     channels: int = 3
#     seq_len: int = 5
#     latent_dim: int = 64
#     enc_out_dim: int = 512
#     source_data: str = "sim"
#     checkpoint_dir = os.path.join(root_path, "digit_vae", "checkpoints")

#     # def __post_init__(self):
#     #     root_path = os.path.join(self.root_path, self.name_project)
#     #     self.checkpoint_dir = os.path.join(
#     #         root_path, "outputs", "checkpoints", "digit_vae"
#     #     )


# @dataclass
# class Viz3d_Params:
#     object_mesh_path: str = ""
#     obj_mesh_scale: float = 1.0
#     dist_min: float = 0.0
#     dist_max: float = 0.01
#     rotation_euler: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
#     viz_scene: bool = False


# @dataclass
# class NCF_Params:
#     pc_run: str = "meta"  # meta
#     root_path: str = os.path.abspath(os.path.join("."))

#     # params digit data
#     seq_len: int = 5
#     source_data: str = "sim"

#     # params sdf
#     dist_min: float = 0.0
#     dist_max: float = 1.0

#     # params training
#     ncf_arch: str = "transformer"  # classic, mlp, conv1, transformer
#     name_run: str = f""
#     batch_size: int = 10
#     num_workers: int = 8
#     num_epochs: int = 10
#     lr: float = 1e-3
#     seed: int = 1
#     loss: str = "classic"  # mse, bce, l1, classic
#     weight_loss: float = 10.0
#     weight_chamfer_loss: float = 0.0

#     # params ncf
#     hidden_size_ncf: int = 128
#     pc_subsample: float = 1.0

#     # checkpoints
#     objects_assets: str = os.path.join(root_path, "assets")

#     vae_params = Digit_VAE_Params()
#     viz3d_params = Viz3d_Params()

#     def __post_init__(self):
#         # root_path = os.path.join(self.root_path, self.name_project)
#         root_path = self.root_path

#         self.name_run: str = f"ncf_{self.ncf_arch}"

#         if self.pc_run == "meta":
#             self.path_dataset: str = (
#                 "/home/chiguera/Documents/NCF/ncf_cupholder_dataset/"
#             )
#         else:
#             self.path_dataset: str = "/media/chiguera/2TB/ncf_cupholder_dataset/"

#         if self.pc_run == "meta":
#             self.logs_dir: str = (
#                 f"/home/chiguera/Documents/training_logs/ncf_cupholder/{self.name_run}"
#             )
#         else:
#             self.logs_dir: str = (
#                 f"/media/chiguera/2TB/training_logs/ncf_cupholder/{self.name_run}/"
#             )

#         self.checkpoint_ncf: str = os.path.join(root_path, "checkpoints", self.name_run)
#         self.checkpoint_ndf = os.path.join(
#             root_path, "models", "ndf", "ndf_weights", "ndf_demo_mug_weights.pth"
#         )
#         self.results_dir: str = os.path.join(root_path, "outputs", self.name_run)

#         self.viz3d_params = Viz3d_Params(
#             object_mesh_path=os.path.join(self.objects_assets, "mug_ycb.obj"),
#             obj_mesh_scale=0.0090,
#             dist_min=self.dist_min,
#             dist_max=self.dist_max,
#             rotation_euler=[0.0, 0.0, 90.0],
#             # viz_scene=True if self.source_data == "real" else False,
#             viz_scene=False,
#         )

#         self.vae_params = Digit_VAE_Params(
#             seq_len=self.seq_len, source_data=self.source_data
#         )
