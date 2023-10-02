import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from algo2.ncf.utils import torch_util

EEF_MAX = np.array([0.10, 0.10, 0.20])  # m
DELTA_EEF_MAX = np.array([1e-2, 1e-2, 2e-2])  # cm
DELTA_QUAT_MAX = np.array([0.01, 0.01, 0.01, 1.0])


class NCF_dataloader:
    def __init__(
        self,
        num_envs,
        path_mesh_object,
        path_pointcloud_object,
        path_ndf_code_object,
        pc_subsample=1.0,
    ):
        self.num_envs = num_envs
        self.path_mesh_object = path_mesh_object
        self.path_pointcloud_object = path_pointcloud_object
        self.pc_subsample = pc_subsample

        self.shape_code = np.load(path_ndf_code_object)
        self.shape_code = torch.tensor(self.shape_code, dtype=torch.float32)

        self.ncf_point_cloud = np.load(path_pointcloud_object)
        self.ncf_point_cloud = self.ncf_point_cloud - np.mean(
            self.ncf_point_cloud, axis=0
        )
        self.n_points = len(self.ncf_point_cloud)


    def get_ee_sequence(self, ee_poses):
        num_envs = ee_poses.shape[0]
        ee_poses = ee_poses.cpu().numpy()
        # ee_poses[:, :, 2] -= 0.040

        ee_poses_envs = np.zeros_like(ee_poses)
        ee_t1_envs = np.zeros((num_envs, 7))
        ee_t2_envs = np.zeros((num_envs, 7))

        for i in range(num_envs):
            ee_seq = ee_poses[i]
            ee_seq_cp = np.copy(ee_seq)

            r0 = R.from_quat(ee_seq_cp[0][3:]).as_matrix()
            r1 = R.from_quat(ee_seq_cp[1][3:]).as_matrix()
            r2 = R.from_quat(ee_seq_cp[2][3:]).as_matrix()
            q_t_1 = R.from_matrix(np.matmul(r0, r1.T)).as_quat()
            q_t_2 = R.from_matrix(np.matmul(r0, r2.T)).as_quat()
            t_t_1 = ee_seq_cp[0][0:3] - ee_seq_cp[1][0:3]
            t_t_2 = ee_seq_cp[0][0:3] - ee_seq_cp[2][0:3]
            ee_t1 = np.concatenate((t_t_1, q_t_1))
            ee_t2 = np.concatenate((t_t_2, q_t_2))

            ee_poses_envs[i][:, 0:3] = ee_seq[:, 0:3] / EEF_MAX
            ee_poses_envs[i][:, 3:] = ee_seq[:, 3:]
            ee_t1_envs[i][0:3] = ee_t1[0:3] / DELTA_EEF_MAX
            ee_t1_envs[i][3:] = ee_t1[3:] / DELTA_QUAT_MAX
            ee_t2_envs[i][0:3] = ee_t2[0:3] / DELTA_EEF_MAX
            ee_t2_envs[i][3:] = ee_t2[3:] / DELTA_QUAT_MAX

        ee_poses_envs = torch.tensor(ee_poses_envs, dtype=torch.float32)
        ee_t1_envs = torch.tensor(ee_t1_envs, dtype=torch.float32)
        ee_t2_envs = torch.tensor(ee_t2_envs, dtype=torch.float32)

        return ee_poses_envs, ee_t1_envs, ee_t2_envs

    def get_pointcloud_data(self):
        idx_points = np.arange(self.n_points)
        n_query_pts = int(self.n_points * self.pc_subsample)
        idx_query = np.random.choice(idx_points, size=n_query_pts)
        ncf_query_points = self.ncf_point_cloud[idx_query]
        ncf_query_points = torch.tensor(ncf_query_points, dtype=torch.float32).repeat(
            self.num_envs, 1, 1
        )
        shape_code = self.shape_code.repeat(self.num_envs, 1)

        return shape_code, ncf_query_points, idx_query

    def get_pointcloud_data_ndf(self):
        # points for ncf

        ncf_point_cloud = np.load(self.path_pointcloud_object)
        ncf_point_cloud = ncf_point_cloud - np.mean(ncf_point_cloud, axis=0)
        n_points = len(ncf_point_cloud)

        idx_points = np.arange(n_points)
        n_query_pts = int(n_points * 1.0)
        idx_query = np.random.choice(idx_points, size=n_query_pts)
        ncf_query_points = ncf_point_cloud[idx_query]

        ncf_query_points = torch.tensor(ncf_query_points, dtype=torch.float32).repeat(
            self.num_envs, 1, 1
        )

        # points for ndf
        object_trimesh = trimesh.load(self.path_mesh_object)
        T = np.eye(4)
        T[0:3, 0:3] = R.from_euler("xyz", [0, 0, 90], degrees=True).as_matrix()
        object_trimesh = object_trimesh.apply_transform(T)

        ndf_point_cloud = np.array(
            trimesh.sample.sample_surface(object_trimesh, n_points // 2)[0]
        )
        ndf_point_cloud = ndf_point_cloud - np.mean(ndf_point_cloud, axis=0)
        ndf_point_cloud = torch.tensor(ndf_point_cloud, dtype=torch.float32).repeat(
            self.num_envs, 1, 1
        )

        return ndf_point_cloud, ncf_query_points, idx_query


import trimesh
from vedo import trimesh2vedo, Points, show, Text2D
from .pose import xyzquat_to_tf_numpy, euler_angles_to_matrix


class NCF_Viz:
    def __init__(self, path_assets=None, cupholder_scale=0.75):
        # path_assets = (
        #     "/home/chiguera/Documents/NCF/NCFgym/assets/factory/mesh/ncf_mug_cupholder/"
        # )
        if path_assets is None:
            path_assets = "/home/chiguera/Documents/NCF/NCFgym/assets/factory/mesh/ncf_mug_cupholder/"

        object_path = f"{path_assets}/obj_assets/1/obj_collision.obj"
        object_pc_path = f"{path_assets}/obj_assets/1/obj_pointcloud.npy"
        cupholder_path = f"{path_assets}/cupholder/cupholder_v2.obj"

        self.mesh_object = trimesh.load_mesh(object_path)
        self.mesh_object.apply_scale(1.0)
        T = np.eye(4)
        T[0:3, 0:3] = R.from_euler("xyz", [0.0, 0.0, 90.0], degrees=True).as_matrix()
        self.mesh_object = self.mesh_object.apply_transform(T)

        self.cupholder_trimesh = trimesh.load(cupholder_path)
        self.cupholder_trimesh = self.cupholder_trimesh.apply_scale(cupholder_scale)

        self.pc = np.load(object_pc_path)

        self.object_3d_cam = dict(
            position=(-7.54483e-3, -0.0849045, -0.250212),
            focal_point=(-4.82255e-3, -2.87705e-3, 0),
            viewup=(0.580505, -0.775588, 0.247946),
            distance=0.263329,
            clipping_range=(0.174192, 0.376170),
        )

    def show_obj_probabilities(self, dist, interactive=False):
        pc = Points(self.pc, r=15)
        pc = pc.cmap(
            "plasma",
            dist,
            vmin=0.0,
            vmax=1.0,
        )
        mesh_vedo = trimesh2vedo(self.mesh_object).clone()
        # mesh_vedo.subdivide(n=10, method=2)
        mesh_vedo = mesh_vedo.interpolate_data_from(
            pc, n=5, on="points", kernel="gaussian"
        ).cmap("plasma", vmin=0.0, vmax=1.0)

        if interactive:
            show([mesh_vedo, pc], axes=1, camera=self.object_3d_cam)
        else:
            img = show(
                [mesh_vedo, pc],
                axes=1,
                camera=self.object_3d_cam,
                interactive=False,
                offscreen=True,
            ).screenshot(asarray=True)
            return img

    def show_scene(self, object_pose, cupholder_pose, probs, label, interactive=False):
        # cam = dict(
        #     position=(0.227997, -0.375556, -0.239064),
        #     focal_point=(-0.0186972, 0.0437448, 0.0995462),
        #     viewup=(0.265986, -0.506072, 0.820452),
        #     distance=0.592729,
        #     clipping_range=(0.297838, 0.853488),
        # )
        cam = dict(
            position=(0.335978, -0.270905, -0.0235918),
            focal_point=(-0.0186971, 0.0437446, 0.0995460),
            viewup=(0.194292, -0.159759, 0.967847),
            distance=0.489859,
            clipping_range=(0.240417, 0.708860),
        )
        tf = np.eye(4)
        tf[0:3, 0:3] = euler_angles_to_matrix(
            euler_angles=torch.tensor([[0, 0, -np.pi]]), convention="XYZ"
        ).numpy()

        object_pose = xyzquat_to_tf_numpy(object_pose.cpu().numpy())  # @ tf
        cupholder_pose = xyzquat_to_tf_numpy(cupholder_pose.cpu().numpy())  # @ tf

        cupholder_i = self.cupholder_trimesh.copy()
        cupholder_i.apply_transform(cupholder_pose)

        obj_i = self.mesh_object.copy()
        obj_i.apply_transform(object_pose)

        object_pc_i = trimesh.points.PointCloud(self.pc.copy())
        object_pc_i.apply_transform(object_pose)

        mesh = trimesh2vedo(obj_i).clone()
        mesh.subdivide(n=3, method=2)

        holder_mesh = trimesh2vedo(cupholder_i).clone()

        pts_external_contact = Points(object_pc_i.vertices, r=10).cmap(
            "plasma", probs, vmin=0.0, vmax=1.0
        )

        mesh.interpolate_data_from(
            pts_external_contact, n=3, on="points", kernel="gaussian"
        ).cmap("plasma", vmin=0.0, vmax=1.0)

        holder_mesh.alpha(0.5).color("gray")

        if interactive:
            show([mesh, holder_mesh, pts_external_contact], axes=0, camera=cam).close()
        else:
            text = Text2D(label, pos="bottom-left", s=1, c="black")
            img = show(
                [mesh, holder_mesh, pts_external_contact, text],
                axes=0,
                camera=cam,
                interactive=False,
                offscreen=True,
            ).screenshot(asarray=True)
            show().close()
            return img

    def show_gt_pred(self, object_pose, cupholder_pose, p_gt, p_pred):
        img_gt = self.show_scene(
            object_pose, cupholder_pose, p_gt, "Ground Truth", interactive=False
        )
        img_pred = self.show_scene(
            object_pose, cupholder_pose, p_pred, "NCF estimation", interactive=False
        )
        return img_gt, img_pred

    def show_gt_sim_real(
        self, object_pose, cupholder_pose, p_gt, p_sim, p_sim_real, p_real
    ):
        img_gt = self.show_scene(
            object_pose, cupholder_pose, p_gt, "Ground Truth", interactive=False
        )
        img_sim = self.show_scene(
            object_pose, cupholder_pose, p_sim, "NCF sim", interactive=False
        )

        img_sim_real = self.show_scene(
            object_pose,
            cupholder_pose,
            p_sim_real,
            "NCF sim and real inputs",
            interactive=False,
        )

        img_real = self.show_scene(
            object_pose, cupholder_pose, p_real, "NCF real", interactive=False
        )
        return img_gt, img_sim, img_sim_real, img_real
