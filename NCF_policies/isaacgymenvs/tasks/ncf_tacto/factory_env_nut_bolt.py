# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: class for nut-bolt env.

Inherits base class and abstract environment class. Inherited by nut-bolt task classes. Not directly executed.

Configuration defined in FactoryEnvNutBolt.yaml. Asset info defined in factory_asset_info_nut_bolt.yaml.
"""

import cv2
import hydra
import numpy as np
import os
import torch

from isaacgym import gymapi
from isaacgymenvs.tasks.ncf_tacto.factory_base import FactoryBase
import isaacgymenvs.tasks.ncf_tacto.factory_control as fc
from isaacgymenvs.tasks.ncf_tacto.factory_schema_class_env import FactoryABCEnv
from isaacgymenvs.tasks.ncf_tacto.factory_schema_config_env import (
    FactorySchemaConfigEnv,
)

from isaacgymenvs.tasks.ncf_tacto.digit_renderer import digit_renderer
from isaacgymenvs.tasks.ncf_tacto.pose import (
    xyzquat_to_tf_numpy,
    euler_angles_to_matrix,
)
from dataclasses import dataclass
import matplotlib.pyplot as plt

from multiprocessing import Process, Queue, Manager


@dataclass
class Cfg_Tacto:
    pixmm: float = 0.03
    width: int = 240
    height: int = 320
    cam_dist: float = 0.022
    shear_mag: float = 0.0
    pen_min: float = 0.0005
    pen_max: float = 0.002
    randomize: bool = False
    bg_id: int = 10


class FactoryEnvNutBolt(FactoryBase, FactoryABCEnv):
    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""
        self.t_digits = 0
        self.subsample_digits = 5  # 5
        self._get_env_yaml_params()

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()
        if cfg["sim"]["sim_digits"]:
            self.refresh_digit_sensors()

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_env", node=FactorySchemaConfigEnv)

        config_path = (
            "task/NCFEnvNutBolt.yaml"  # relative to Hydra search path (cfg dir)
        )
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env["task"]  # strip superfluous nesting

        asset_info_path = "../../assets/factory/yaml/factory_asset_info_nut_bolt.yaml"
        self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        self.asset_info_nut_bolt = self.asset_info_nut_bolt[""][""][""][""][""][""][
            "assets"
        ]["factory"][
            "yaml"
        ]  # strip superfluous nesting

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(
            -self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0
        )
        upper = gymapi.Vec3(
            self.cfg_base.env.env_spacing,
            self.cfg_base.env.env_spacing,
            self.cfg_base.env.env_spacing,
        )
        num_per_row = int(np.sqrt(self.num_envs))

        self.print_sdf_warning()
        franka_asset, table_asset = self.import_franka_assets()
        (
            nut_asset,
            bolt_asset,
            nut_mesh_file,
            nut_mesh_scale,
            nut_pointcloud_file,
        ) = self._import_env_assets()

        # self.init_digit_render(object_mesh=nut_mesh_file, object_mesh_scale=nut_mesh_scale)

        self._create_actors(
            lower,
            upper,
            num_per_row,
            franka_asset,
            nut_asset,
            bolt_asset,
            table_asset,
            nut_mesh_file,
            nut_mesh_scale,
            nut_pointcloud_file,
        )

    def _import_env_assets(self):
        """Set nut and bolt asset options. Import assets."""

        urdf_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "assets", "factory", "urdf"
        )
        mesh_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "assets", "factory", "mesh"
        )

        nut_options = gymapi.AssetOptions()
        nut_options.flip_visual_attachments = False
        nut_options.fix_base_link = False
        nut_options.thickness = 0.0  # default = 0.02
        nut_options.armature = 0.0  # default = 0.0
        nut_options.use_physx_armature = True
        nut_options.linear_damping = 0.0  # default = 0.0
        nut_options.max_linear_velocity = 1000.0  # default = 1000.0
        nut_options.angular_damping = 0.0  # default = 0.5
        nut_options.max_angular_velocity = 64.0  # default = 64.0
        nut_options.disable_gravity = False
        nut_options.enable_gyroscopic_forces = True
        nut_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        nut_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            nut_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        bolt_options = gymapi.AssetOptions()
        bolt_options.flip_visual_attachments = False
        bolt_options.fix_base_link = True
        bolt_options.thickness = 0.0  # default = 0.02
        bolt_options.armature = 0.0  # default = 0.0
        bolt_options.use_physx_armature = True
        bolt_options.linear_damping = 0.0  # default = 0.0
        bolt_options.max_linear_velocity = 1000.0  # default = 1000.0
        bolt_options.angular_damping = 0.0  # default = 0.5
        bolt_options.max_angular_velocity = 64.0  # default = 64.0
        bolt_options.disable_gravity = False
        bolt_options.enable_gyroscopic_forces = True
        bolt_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        bolt_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            bolt_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        nut_assets = []
        bolt_assets = []
        for subassembly in self.cfg_env.env.desired_subassemblies:
            components = list(self.asset_info_nut_bolt[subassembly])
            nut_file = (
                self.asset_info_nut_bolt[subassembly][components[0]]["urdf_path"]
                + ".urdf"
            )
            bolt_file = (
                self.asset_info_nut_bolt[subassembly][components[1]]["urdf_path"]
                + ".urdf"
            )
            nut_options.density = self.cfg_env.env.nut_bolt_density
            bolt_options.density = self.cfg_env.env.nut_bolt_density
            nut_asset = self.gym.load_asset(self.sim, urdf_root, nut_file, nut_options)
            bolt_asset = self.gym.load_asset(
                self.sim, urdf_root, bolt_file, bolt_options
            )
            nut_assets.append(nut_asset)
            bolt_assets.append(bolt_asset)

        nut_mesh_file = (
            mesh_root
            + "/"
            + self.asset_info_nut_bolt[subassembly][components[0]]["mesh_path"]
        )
        nut_mesh_scale = self.asset_info_nut_bolt[subassembly][components[0]][
            "mesh_scale"
        ]

        nut_pointcloud_file = (
            mesh_root
            + "/"
            + self.asset_info_nut_bolt[subassembly][components[0]]["pointcloud_path"]
        )

        return (
            nut_assets,
            bolt_assets,
            nut_mesh_file,
            nut_mesh_scale,
            nut_pointcloud_file,
        )

    def _create_actors(
        self,
        lower,
        upper,
        num_per_row,
        franka_asset,
        nut_assets,
        bolt_assets,
        table_asset,
        nut_mesh_file,
        nut_mesh_scale,
        nut_pointcloud_file,
    ):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        franka_pose = gymapi.Transform()
        franka_pose.p.x = self.cfg_base.env.franka_depth
        franka_pose.p.y = 0.0
        franka_pose.p.z = 0.0
        franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        table_pose = gymapi.Transform()
        table_pose.p.x = 0.0
        table_pose.p.y = 0.0
        table_pose.p.z = self.cfg_base.env.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.env_ptrs = []
        self.franka_handles = []
        self.nut_handles = []
        self.bolt_handles = []
        self.table_handles = []
        self.shape_ids = []
        self.franka_actor_ids_sim = []  # within-sim indices
        self.nut_actor_ids_sim = []  # within-sim indices
        self.bolt_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        self.nut_heights = []
        self.nut_widths_max = []
        self.bolt_widths = []
        self.bolt_head_heights = []
        self.bolt_shank_lengths = []
        self.thread_pitches = []

        self.nut_mesh_scale = nut_mesh_scale
        self.nut_mesh_file = nut_mesh_file
        nut_pc = np.load(nut_pointcloud_file)
        self.nut_pointcloud = torch.tensor(nut_pc, dtype=torch.float32).to(self.device)

        print(f"[isaacgym] creating {self.num_envs} envs with digit sensors")

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(
                    env_ptr,
                    franka_asset,
                    franka_pose,
                    "franka",
                    i + self.num_envs,
                    0,
                    0,
                )
            else:
                franka_handle = self.gym.create_actor(
                    env_ptr, franka_asset, franka_pose, "franka", i, 0, 0
                )
            self.franka_actor_ids_sim.append(actor_count)
            actor_count += 1

            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))
            subassembly = self.cfg_env.env.desired_subassemblies[j]
            components = list(self.asset_info_nut_bolt[subassembly])

            nut_pose = gymapi.Transform()
            nut_pose.p.x = 0.0
            nut_pose.p.y = self.cfg_env.env.nut_lateral_offset
            nut_pose.p.z = self.cfg_base.env.table_height
            nut_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            nut_color = gymapi.Vec3(0.4, 0.0, 0.1)
            nut_handle = self.gym.create_actor(
                env_ptr, nut_assets[j], nut_pose, "nut", i, 0, 0
            )
            self.gym.set_rigid_body_color(
                env_ptr, nut_handle, 0, gymapi.MESH_VISUAL, nut_color
            )
            self.nut_actor_ids_sim.append(actor_count)
            actor_count += 1

            nut_height = self.asset_info_nut_bolt[subassembly][components[0]]["height"]
            nut_width_max = self.asset_info_nut_bolt[subassembly][components[0]][
                "width_max"
            ]
            self.nut_heights.append(nut_height)
            self.nut_widths_max.append(nut_width_max)

            bolt_pose = gymapi.Transform()
            bolt_pose.p.x = 0.0
            bolt_pose.p.y = 0.0
            bolt_pose.p.z = self.cfg_base.env.table_height
            bolt_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            bolt_color = gymapi.Vec3(0.1, 0.6, 0.1)
            bolt_handle = self.gym.create_actor(
                env_ptr, bolt_assets[j], bolt_pose, "bolt", i, 0, 0
            )
            self.gym.set_rigid_body_color(
                env_ptr, bolt_handle, 0, gymapi.MESH_VISUAL, bolt_color
            )
            self.bolt_actor_ids_sim.append(actor_count)
            actor_count += 1

            bolt_width = self.asset_info_nut_bolt[subassembly][components[1]]["width"]
            bolt_head_height = self.asset_info_nut_bolt[subassembly][components[1]][
                "head_height"
            ]
            bolt_shank_length = self.asset_info_nut_bolt[subassembly][components[1]][
                "shank_length"
            ]
            self.bolt_widths.append(bolt_width)
            self.bolt_head_heights.append(bolt_head_height)
            self.bolt_shank_lengths.append(bolt_shank_length)

            thread_pitch = self.asset_info_nut_bolt[subassembly]["thread_pitch"]
            self.thread_pitches.append(thread_pitch)

            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, 0, 0
            )
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

            link7_id = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_link7", gymapi.DOMAIN_ACTOR
            )
            hand_id = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_hand", gymapi.DOMAIN_ACTOR
            )
            left_finger_id = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_leftfinger", gymapi.DOMAIN_ACTOR
            )
            right_finger_id = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_rightfinger", gymapi.DOMAIN_ACTOR
            )
            self.shape_ids = [link7_id, hand_id, left_finger_id, right_finger_id]

            franka_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, franka_handle
            )
            franka_shape_index = self.gym.get_actor_rigid_body_shape_indices(
                env_ptr, franka_handle
            )
            for shape_idx in self.shape_ids:
                shape_id = franka_shape_index[shape_idx].start
                franka_shape_props[
                    shape_id
                ].friction = self.cfg_base.env.franka_friction
                franka_shape_props[shape_id].rolling_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].torsion_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].restitution = 0.0  # default = 0.0
                franka_shape_props[shape_id].compliance = 0.0  # default = 0.0
                franka_shape_props[shape_id].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, franka_handle, franka_shape_props
            )

            nut_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, nut_handle
            )
            nut_shape_props[0].friction = self.cfg_env.env.nut_bolt_friction
            nut_shape_props[0].rolling_friction = 0.0  # default = 0.0
            nut_shape_props[0].torsion_friction = 0.0  # default = 0.0
            nut_shape_props[0].restitution = 0.0  # default = 0.0
            nut_shape_props[0].compliance = 0.0  # default = 0.0
            nut_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, nut_handle, nut_shape_props
            )

            bolt_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, bolt_handle
            )
            bolt_shape_props[0].friction = self.cfg_env.env.nut_bolt_friction
            bolt_shape_props[0].rolling_friction = 0.0  # default = 0.0
            bolt_shape_props[0].torsion_friction = 0.0  # default = 0.0
            bolt_shape_props[0].restitution = 0.0  # default = 0.0
            bolt_shape_props[0].compliance = 0.0  # default = 0.0
            bolt_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, bolt_handle, bolt_shape_props
            )

            table_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, table_handle
            )
            table_color = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_color
            )

            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0  # default = 0.0
            table_shape_props[0].torsion_friction = 0.0  # default = 0.0
            table_shape_props[0].restitution = 0.0  # default = 0.0
            table_shape_props[0].compliance = 0.0  # default = 0.0
            table_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, table_handle, table_shape_props
            )

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)

            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)
            self.franka_handles.append(franka_handle)
            self.nut_handles.append(nut_handle)
            self.bolt_handles.append(bolt_handle)
            self.table_handles.append(table_handle)

        print(f"[isaacgym] environment creation complete")

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(
            self.franka_actor_ids_sim, dtype=torch.int32, device=self.device
        )
        self.nut_actor_ids_sim = torch.tensor(
            self.nut_actor_ids_sim, dtype=torch.int32, device=self.device
        )
        self.bolt_actor_ids_sim = torch.tensor(
            self.bolt_actor_ids_sim, dtype=torch.int32, device=self.device
        )

        # For extracting root pos/quat
        self.nut_actor_id_env = self.gym.find_actor_index(
            env_ptr, "nut", gymapi.DOMAIN_ENV
        )
        self.bolt_actor_id_env = self.gym.find_actor_index(
            env_ptr, "bolt", gymapi.DOMAIN_ENV
        )

        # For extracting body pos/quat, force, and Jacobian
        self.nut_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, nut_handle, "nut", gymapi.DOMAIN_ENV
        )
        self.bolt_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, bolt_handle, "bolt", gymapi.DOMAIN_ENV
        )
        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_hand", gymapi.DOMAIN_ENV
        )
        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_leftfinger", gymapi.DOMAIN_ENV
        )
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_rightfinger", gymapi.DOMAIN_ENV
        )
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_fingertip_centered", gymapi.DOMAIN_ENV
        )
        self.left_fingertip_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_leftfinger_tip", gymapi.DOMAIN_ENV
        )
        self.right_fingertip_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_rightfinger_tip", gymapi.DOMAIN_ENV
        )

        # For computing body COM pos
        self.nut_heights = torch.tensor(self.nut_heights, device=self.device).unsqueeze(
            -1
        )
        self.bolt_head_heights = torch.tensor(
            self.bolt_head_heights, device=self.device
        ).unsqueeze(-1)

        # For setting initial state
        self.nut_widths_max = torch.tensor(
            self.nut_widths_max, device=self.device
        ).unsqueeze(-1)
        self.bolt_shank_lengths = torch.tensor(
            self.bolt_shank_lengths, device=self.device
        ).unsqueeze(-1)

        # For defining success or failure
        self.bolt_widths = torch.tensor(self.bolt_widths, device=self.device).unsqueeze(
            -1
        )
        self.thread_pitches = torch.tensor(
            self.thread_pitches, device=self.device
        ).unsqueeze(-1)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        self.nut_pos = self.root_pos[:, self.nut_actor_id_env, 0:3]
        self.nut_quat = self.root_quat[:, self.nut_actor_id_env, 0:4]
        self.nut_linvel = self.root_linvel[:, self.nut_actor_id_env, 0:3]
        self.nut_angvel = self.root_angvel[:, self.nut_actor_id_env, 0:3]

        self.bolt_pos = self.root_pos[:, self.bolt_actor_id_env, 0:3]
        self.bolt_quat = self.root_quat[:, self.bolt_actor_id_env, 0:4]

        self.nut_force = self.contact_force[:, self.nut_body_id_env, 0:3]

        self.bolt_force = self.contact_force[:, self.bolt_body_id_env, 0:3]

        self.nut_com_pos = fc.translate_along_local_z(
            pos=self.nut_pos,
            quat=self.nut_quat,
            offset=self.bolt_head_heights + self.nut_heights * 0.5,
            device=self.device,
        )
        self.nut_com_quat = self.nut_quat  # always equal
        self.nut_com_linvel = self.nut_linvel + torch.cross(
            self.nut_angvel, (self.nut_com_pos - self.nut_pos), dim=1
        )
        self.nut_com_angvel = self.nut_angvel  # always equal

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        self.nut_com_pos = fc.translate_along_local_z(
            pos=self.nut_pos,
            quat=self.nut_quat,
            offset=self.bolt_head_heights + self.nut_heights * 0.5,
            device=self.device,
        )
        self.nut_com_linvel = self.nut_linvel + torch.cross(
            self.nut_angvel, (self.nut_com_pos - self.nut_pos), dim=1
        )

    def render_digit(
        self, offset, nut_poses, left_finger_poses, right_finger_poses, queue
    ):
        sz = self.cfg["ncf"]["digits_size"]

        cfg_tacto = Cfg_Tacto()
        # add digit handle
        self.digit_render_left = digit_renderer(
            cfg=cfg_tacto,
            obj_path=self.nut_mesh_file,
            obj_quat=np.array([0.0, 0.0, 0.0, 1.0]),
            obj_scale=self.nut_mesh_scale,
            bg_id=np.random.randint(0, 25),
            headless=True,
        )

        self.digit_render_right = digit_renderer(
            cfg=cfg_tacto,
            obj_path=self.nut_mesh_file,
            obj_quat=np.array([0.0, 0.0, 0.0, 1.0]),
            obj_scale=self.nut_mesh_scale,
            bg_id=np.random.randint(0, 25),
            headless=True,
        )

        bg_left = self.digit_render_left.renderer.background
        bg_right = self.digit_render_right.renderer.background

        lf = np.zeros((len(nut_poses), sz, sz, 3))
        rf = np.zeros((len(nut_poses), sz, sz, 3))

        for i in range(len(nut_poses)):
            self.digit_render_left.update_object_pose(nut_poses[i])
            self.digit_render_left.update_pose_given_pose(left_finger_poses[i])
            im_lf, hm_lf, mask_lf = self.digit_render_left.render()
            im_lf = self._subtract_bg(im_lf, bg_left)
            im_lf = cv2.resize(im_lf, (sz, sz), interpolation=cv2.INTER_AREA)

            self.digit_render_right.update_object_pose(nut_poses[i])
            self.digit_render_right.update_pose_given_pose(right_finger_poses[i])
            im_rf, hm_rf, mask_rf = self.digit_render_right.render()
            im_rf = self._subtract_bg(im_rf, bg_right)
            im_rf = cv2.resize(im_rf, (sz, sz), interpolation=cv2.INTER_AREA)

            lf[i] = im_lf
            rf[i] = im_rf

        queue.put((lf, rf, offset))

    def refresh_digit_sensors(self):
        if self.t_digits == 0:
            sz = self.cfg["ncf"]["digits_size"]

            left_digit_poses = torch.cat(
                (self.left_fingertip_pos, self.left_fingertip_quat), dim=-1
            )
            right_digit_poses = torch.cat(
                (self.right_fingertip_pos, self.right_fingertip_quat), dim=-1
            )
            nut_poses = torch.cat((self.nut_pos, self.nut_quat), dim=-1)

            tf = np.eye(4)
            tf[0:3, 0:3] = euler_angles_to_matrix(
                euler_angles=torch.tensor([[0, 90, 0]]), convention="XYZ"
            ).numpy()

            left_finger_poses = xyzquat_to_tf_numpy(left_digit_poses.cpu().numpy())
            left_finger_poses = left_finger_poses @ np.array(
                np.linalg.inv(
                    [
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            )
            left_finger_poses = left_finger_poses @ tf

            right_finger_poses = xyzquat_to_tf_numpy(right_digit_poses.cpu().numpy())
            right_finger_poses = right_finger_poses @ np.array(
                np.linalg.inv(
                    [
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            )
            right_finger_poses = right_finger_poses @ tf

            nut_poses = xyzquat_to_tf_numpy(nut_poses.cpu().numpy())

            # if self.num_envs > 0:
            groups = 8 if self.num_envs > 8 else 1
            offset = np.ceil(self.num_envs / groups).astype(int)
            processes = []
            results = []
            with Manager() as manager:
                for g in range(groups):
                    env_start = offset * g
                    env_end = np.clip(offset * (g + 1), 0, self.num_envs)
                    nut_poses_g = nut_poses[env_start:env_end]
                    left_finger_poses_g = left_finger_poses[env_start:env_end]
                    right_finger_poses_g = right_finger_poses[env_start:env_end]
                    queue = manager.Queue()
                    process = Process(
                        target=self.render_digit,
                        args=(
                            env_start,
                            nut_poses_g,
                            left_finger_poses_g,
                            right_finger_poses_g,
                            queue,
                        ),
                    )
                    process.start()
                    processes.append(process)
                    results.append(queue)

                for process in processes:
                    process.join()

                for queue in results:
                    left, right, offset = queue.get()
                    self.digit_left_buf[
                        offset : offset + len(left), 0, :
                    ] = torch.tensor(left)
                    self.digit_right_buf[
                        offset : offset + len(right), 0, :
                    ] = torch.tensor(right)

        self.t_digits = (
            self.t_digits + 1 if (self.t_digits < self.subsample_digits - 1) else 0
        )

    def _subtract_bg(self, img1, img2, offset=0.5):
        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0 + offset
        return diff
