import argparse
import logging
import os
from datetime import datetime
import threading
from typing import Optional

# noinspection PyUnresolvedReferences
import isaacgym

import hydra
from termcolor import cprint

from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank

# from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import isaacgym_task_map
from omegaconf import DictConfig, OmegaConf
import gym

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

# from algo.policy.ppo.ppo import PPO
# from algo.policy.padapt.padapt import ProprioAdapt
from algo2.policy.ppo.ppo import PPO as PPO2
# from algo2.policy.padapt.padapt import ProprioAdapt as ProprioAdapt2

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# @hydra.main(config_name="config", config_path="./cfg")
def run(cfg: DictConfig, config_path: Optional[str] = None):
    # hydra.initialize(config_path="./cfg", job_name="train")

    if config_path is not None:
        assert cfg is None
        cfg = OmegaConf.load(config_path)

    if cfg.checkpoint:
        if "*" in cfg.checkpoint:
            from glob import glob

            _ckpt = glob(cfg.checkpoint)
            assert len(_ckpt) == 1
            cfg.checkpoint = _ckpt[0]
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    if cfg.train.ppo.multi_gpu:
        rank = int(os.getenv("LOCAL_RANK", "0"))
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f"cuda:{rank}"
        cfg.rl_device = f"cuda:{rank}"
        cfg.graphics_device_id = int(rank)
        # sets seed. if seed is -1 will pick a random one
        cfg.seed = set_seed(cfg.seed + rank)
    else:
        rank = -1
        cfg.seed = set_seed(cfg.seed)

    # sets seed. if seed is -1 will pick a random one
    # cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    cprint("Start Building the Environment", "green", attrs=["bold"])

    envs = isaacgym_task_map[cfg.task_name](
        cfg=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        rl_device=cfg.rl_device,
        # graphics_device_id=-1,  # if cfg.headless else cfg.graphics_device_id,
        graphics_device_id=0,
        # graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=False,  # cfg.capture_video,
        force_render=cfg.force_render if not cfg.headless else False,
    )

    output_dif = os.path.join(cfg.outputs_root_dir, cfg.train.ppo.output_name)
    os.makedirs(output_dif, exist_ok=True)
    agent = eval(cfg.train.algo)(envs, output_dif, full_config=cfg)

    if cfg.test:
        assert cfg.train.load_path
        agent.restore_test(cfg.train.load_path)
        agent.test()
        # sim_timer = cfg.task.env.sim_timer
        # num_trials = 3
        # cprint(f"Running simulation for {num_trials} trials", "green", attrs=["bold"])
        # thread_stop = threading.Event()
        # agent.restore_test(cfg.train.load_path)
        # sim_thread = threading.Thread(
        #     name="agent.test()", target=agent.test, args=[thread_stop]
        # )
        # threading.Thread(
        #     name="sim_time", target=agent.play_games, args=[thread_stop, num_trials]
        # ).start()

        # sim_thread.start()
        # sim_thread.join()
        # cprint(f"Simulation terminated", "green", attrs=["bold"])
    else:
        if rank <= 0:
            date = str(datetime.now().strftime("%m%d%H"))
            # print(git_diff_config('./'))
            # os.system(f'git diff HEAD > {output_dif}/gitdiff.patch')
            with open(os.path.join(output_dif, f"config_{date}.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))

        # check whether execute train by mistake:
        best_ckpt_path = os.path.join(
            "outputs",
            cfg.train.ppo.output_name,
            "stage1_nn" if cfg.train.algo == "PPO" else "stage2_nn",
            "best.pth",
        )
        if os.path.exists(best_ckpt_path):
            user_input = input(
                f"are you intentionally going to overwrite files in {cfg.train.ppo.output_name}, type yes to continue \n"
            )
            if user_input != "yes":
                exit()

        agent.restore_train(cfg.train.load_path)
        agent.train()

    cprint("Finished", "green", attrs=["bold"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", type=argparse.FileType("r"), help="Path to hydra config."
    )
    args = parser.parse_args()
    run(None, args.config_path)
