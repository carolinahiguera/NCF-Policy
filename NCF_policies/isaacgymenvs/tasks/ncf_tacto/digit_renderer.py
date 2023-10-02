# --------------------------------------------------------
# Monte-Carlo inference over distributions across sliding touch 
# https://arxiv.org/abs/2210.14210
# https://github.com/facebookresearch/MidasTouch
# Sudharshan Suresh, Zilin Si, Stuart Anderson, Michael Kaess, Mustafa Mukadam
# --------------------------------------------------------

from os import path as osp
import numpy as np
import tacto
import cv2


import torch
from scipy.spatial.transform import Rotation as R
import trimesh
import random

DEBUG = False


class digit_renderer:
    def __init__(
        self,
        cfg,
        obj_path: str = None,
        obj_quat: np.ndarray = [0.0, 0.0, 0.0, 1.0],
        obj_scale: float = 1.0,
        headless=False,
        bg_id=None,
    ):
        self.render_config = cfg
        self.randomize = self.render_config.randomize
        if self.randomize:
            bg_id = random.randint(0, 9)
        elif bg_id is None:
            bg_id = cfg.bg_id

        # Create renderer
        self.renderer = tacto.Renderer(
            width=cfg.width,
            height=cfg.height,
            background=cv2.imread(tacto.get_background_image_path(bg_id)),
            config_path=tacto.get_digit_shadow_config_path(),
            # config_path=tacto.get_digit_config_path(),
            headless=headless,
        )
        self.cam_dist = cfg.cam_dist
        self.pixmm = cfg.pixmm

        if not DEBUG:
            _, self.bg_depth = self.renderer.render()
            self.bg_depth = self.bg_depth[0]
            self.bg_depth_pix = self.correct_pyrender_height_map(self.bg_depth)

        if obj_path is not None:
            # self.obj_loader = object_loader(obj_path)
            # self.obj_loader.set_object_pose(pose=obj_pose)
            obj_trimesh = trimesh.load(obj_path)
            obj_trimesh = obj_trimesh.apply_scale(obj_scale)
            self.obj_mesh = obj_trimesh
            obj_euler = R.from_quat(obj_quat).as_euler("xyz", degrees=False)
            self.renderer.add_object(obj_trimesh, "object", orientation=obj_euler)

        self.press_depth = 0.001

    def randomize_light(self):
        self.renderer.randomize_light()

    def get_background(self, frame="gel"):
        """
        Return cached bg image
        """
        return self.bg_depth_pix if frame == "gel" else self.bg_depth

    def pix2meter(self, pix):
        """
        Convert pixel to meter
        """
        return pix * self.pixmm / 1000.0

    def meter2pix(self, m):
        """
        Convert meter to pixels
        """
        return m * 1000.0 / self.pixmm

    def update_object_pose(self, pose):
        # (x, y, z) and euler angles
        self.renderer.update_object_pose(obj_name="object", pose=pose)

    def update_pose_given_point(self, point, press_depth, shear_mag, delta):
        """
        Convert meter to pixels
        """
        dist = np.linalg.norm(point - self.obj_loader.obj_vertices, axis=1)
        idx = np.argmin(dist)

        # idx: the idx vertice, get a new pose
        new_position = self.obj_loader.obj_vertices[idx].copy()
        new_orientation = self.obj_loader.obj_normals[idx].copy()

        delta = np.random.uniform(low=0.0, high=2 * np.pi, size=(1,))[0]
        new_pose = pose_from_vertex_normal(
            new_position, new_orientation, shear_mag, delta
        ).squeeze()
        self.update_pose_given_pose(press_depth, new_pose)

    def update_pose_given_pose(self, cam_pose):
        """
        Given tf gel_pose and press_depth, update tacto camera
        """
        self.renderer.update_camera_pose_from_matrix(cam_pose)

    def add_press(self, pose):
        """
        Add sensor penetration
        """
        pen_mat = np.eye(4)
        pen_mat[0, 3] = self.press_depth
        return np.matmul(pose, pen_mat)

    def gel2cam(self, gel_pose):
        """
        Convert gel_pose to cam_pose
        """
        cam_tf = np.eye(4)
        cam_tf[2, 3] = self.cam_dist
        return np.matmul(gel_pose, cam_tf)

    def cam2gel(self, cam_pose):
        """
        Convert cam_pose to gel_pose
        """
        cam_tf = np.eye(4)
        cam_tf[2, 3] = -self.cam_dist
        return np.matmul(cam_pose, cam_tf)

    # input depth is in camera frame here
    def render(self):
        """
        render [tactile image + depth + mask] @ current pose
        """
        color, depth = self.renderer.render()
        color, depth = color[0], depth[0]
        diff_depth = (self.bg_depth) - depth
        contact_mask = diff_depth > np.abs(self.press_depth * 0.2)
        gel_depth = self.correct_pyrender_height_map(depth)  #  pix in gel frame
        # cam_depth = self.correct_image_height_map(gel_depth) #  pix in gel frame
        # assert np.allclose(cam_depth, depth), "Conversion to pixels is incorrect"
        if self.randomize:
            self.renderer.randomize_light()
        return color, gel_depth, contact_mask

    def correct_pyrender_height_map(self, height_map):
        """
        Input: height_map in meters, in camera frame
        Output: height_map in pixels, in gel frame
        """
        # move to the gel center
        height_map = (self.cam_dist - height_map) * (1000 / self.pixmm)
        return height_map

    def correct_image_height_map(self, height_map, output_frame="cam"):
        """
        Input: height_map in pixels, in gel frame
        Output: height_map in meters, in camera/gel frame
        """
        height_map = (
            -height_map * (self.pixmm / 1000)
            + float(output_frame == "cam") * self.cam_dist
        )
        return height_map

    def get_cam_pose_matrix(self):
        """
        return camera pose matrix of renderer
        """
        return self.renderer.camera_nodes[0].matrix

    def get_cam_pose(self):
        """
        return camera pose of renderer
        """
        # print(f"Cam pose: {tf_to_xyzquat(self.get_cam_pose_matrix())}")
        return self.get_cam_pose_matrix()

    def get_gel_pose_matrix(self):
        """
        return gel pose matrix of renderer
        """
        return self.cam2gel(self.get_cam_pose_matrix())

    def get_gel_pose(self):
        """
        return gel pose of renderer
        """
        # print(f"Gel pose: {tf_to_xyzquat(self.get_gel_pose_matrix())}")
        return self.get_gel_pose_matrix()

    def heightmap2Pointcloud(self, depth, contact_mask=None):
        """
        Convert heightmap + contact mask to point cloud
        [Input]  depth: (width, height) in pixels, in gel frame, Contact mask: binary (width, height)
        [Output] pointcloud: [(width, height) - (masked off points), 3] in meters in camera frame
        """
        depth = self.correct_image_height_map(depth, output_frame="cam")

        if contact_mask is not None:
            heightmapValid = depth * contact_mask  # apply contact mask
        else:
            heightmapValid = depth

        f, w, h = self.renderer.f, self.renderer.width / 2.0, self.renderer.height / 2.0

        if not torch.is_tensor(heightmapValid):
            heightmapValid = torch.from_numpy(heightmapValid)
        # (0, 640) and (0, 480)
        xvals = torch.arange(heightmapValid.shape[1], device=heightmapValid.device)
        yvals = torch.arange(heightmapValid.shape[0], device=heightmapValid.device)
        [x, y] = torch.meshgrid(xvals, yvals)
        x, y = torch.transpose(x, 0, 1), torch.transpose(
            y, 0, 1
        )  # future warning: https://github.com/pytorch/pytorch/issues/50276

        # x and y in meters
        x = ((x - w)) / f
        y = ((y - h)) / f

        x *= depth
        y *= -depth

        heightmap_3d = torch.hstack(
            (x.reshape((-1, 1)), y.reshape((-1, 1)), heightmapValid.reshape((-1, 1)))
        )

        heightmap_3d[:, 2] *= -1
        heightmap_3d = heightmap_3d[heightmap_3d[:, 2] != 0]
        return heightmap_3d