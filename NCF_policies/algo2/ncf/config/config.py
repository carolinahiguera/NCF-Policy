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