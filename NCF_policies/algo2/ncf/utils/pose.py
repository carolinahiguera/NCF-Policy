# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    SE(3) pose utilities 
"""

# from isaacgym import gymapi
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import copy 

def tf_to_xyzquat_numpy(pose: torch.Tensor) -> torch.Tensor:
    """
    convert 4 x 4 transformation matrices to [x, y, z, qx, qy, qz, qw]
    """
    pose = np.atleast_3d(pose)

    r = R.from_matrix(np.array(pose[:, 0:3, 0:3]))
    q = r.as_quat()  # qx, qy, qz, qw
    t = pose[:, :3, 3]
    xyz_quat = np.concatenate((t, q), axis=1)

    return xyz_quat  # (N, 7)


def xyzquat_to_tf_numpy(position_quat: np.ndarray) -> np.ndarray:
    """
    convert [x, y, z, qx, qy, qz, qw] to 4 x 4 transformation matrices
    """
    # try:
    position_quat = np.atleast_2d(position_quat)  # (N, 7)
    N = position_quat.shape[0]
    T = np.zeros((N, 4, 4))
    T[:, 0:3, 0:3] = R.from_quat(position_quat[:, 3:]).as_matrix()
    T[:, :3, 3] = position_quat[:, :3]
    T[:, 3, 3] = 1
    # except ValueError:
    #     print("Zero quat error!")
    return T.squeeze()


def tf2isaacgympose(cam_pose):
    r = R.from_matrix(cam_pose[:3, :3])
    axis_angle = r.as_rotvec()
    norm = np.linalg.norm(axis_angle) 
    transform = gymapi.Transform()
    transform.p = gymapi.Vec3(
        cam_pose[0, 3], cam_pose[1, 3], cam_pose[2, 3]
    )
    transform.r = gymapi.Quat.from_axis_angle(
        gymapi.Vec3(axis_angle[0], axis_angle[1], axis_angle[2]), norm
    )
    return transform

def flip_camera_z(cam_pose):
    '''
    flip camera pose by 180 deg in the X-axis, so depth is negative Z
    '''
    quat_real_to_tac_neural = gymapi.Quat.from_euler_zyx(np.radians(180), 0.0, 0.0)
    flip_cam_pose = copy.deepcopy(cam_pose)
    flip_cam_pose.r = cam_pose.r * quat_real_to_tac_neural
    return flip_cam_pose

def isaacgym2tacneural(cam_pose):
    cam_pose = np.array(
        [
            cam_pose.p.x,
            cam_pose.p.y,
            cam_pose.p.z,
            cam_pose.r.x,
            cam_pose.r.y,
            cam_pose.r.z,
            cam_pose.r.w,
        ]
    )
    return xyzquat_to_tf_numpy(cam_pose)
    
def xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """
    Convention change: [x, y, z, qx, qy, qz, qw] --> [x, y, z, qw, qx, qy, qz]
    """
    if quat.shape[1] == 7:
        return quat[:, [0, 1, 2, 6, 3, 4, 5]]
    else:
        return quat[:, [3, 0, 1, 2]]


def wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """
    Convention change: [x, y, z, qw, qx, qy, qz] --> [x, y, z, qx, qy, qz, qw]
    """
    if quat.shape[1] == 7:
        return quat[:, [0, 1, 2, 4, 5, 6, 3]]
    else:
        return quat[:, [1, 2, 3, 0]]


def transform_pc(pointclouds: np.ndarray, poses: np.ndarray):
    """
    Transform pointclouds by poses
    """

    if type(pointclouds) is not list:
        temp = pointclouds
        pointclouds = [None] * 1
        pointclouds[0] = temp
        poses = np.expand_dims(poses, axis=2)

    if len(poses.shape) < 3:
        poses = xyzquat_to_tf_numpy(poses)

    transformed_pointclouds = pointclouds
    # TODO: vectorize
    for i, (pointcloud, pose) in enumerate(zip(pointclouds, poses)):
        pointcloud = pointcloud.T
        # 3D affine transform
        pointcloud = pose @ np.vstack([pointcloud, np.ones((1, pointcloud.shape[1]))])
        pointcloud = pointcloud / pointcloud[3, :]
        pointcloud = pointcloud[:3, :].T
        transformed_pointclouds[i] = pointcloud
    return (
        transformed_pointclouds[0] if len(pointclouds) == 1 else transformed_pointclouds
    )


def wrap_angles(angles: torch.Tensor) -> torch.Tensor:
    """
    angles : (N, 3) angles in degrees
    Wraps to [-np.pi, np.pi] or [-180, 180]
    """

    mask = angles > 180.0
    angles[mask] -= 2.0 * 180.0

    mask = angles < -180.0
    angles[mask] += 2.0 * 180.0
    return angles


def quat2euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternions to euler angles
    """
    quat = np.atleast_2d(quat)
    r = R.from_quat(quat)
    return r.as_euler("xyz", degrees=True)


def rot2euler(rot: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to euler angles
    Adapted from so3_rotation_angle() in  https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/so3.html
    """
    rot_trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    phi_cos = torch.acos((rot_trace - 1.0) * 0.5)
    return torch.rad2deg(phi_cos)

    # r = R.from_matrix(np.atleast_3d(rot.cpu().numpy()))
    # eul = r.as_euler('xyz', degrees = True)
    # return torch.tensor(eul, device= rot.device)


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.euler_angles_to_matrix
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return matrices[0] @ matrices[1] @ matrices[2]


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def skew_matrix(v: np.ndarray) -> np.ndarray:
    """
    Get skew-symmetric matrix from vector
    """
    v = np.atleast_2d(v)
    # vector to its skew matrix
    mat = np.zeros((3, 3, v.shape[0]))
    mat[0, 1, :] = -1 * v[:, 2]
    mat[0, 2, :] = v[:, 1]

    mat[1, 0, :] = v[:, 2]
    mat[1, 2, :] = -1 * v[:, 0]

    mat[2, 0, :] = -1 * v[:, 1]
    mat[2, 1, :] = v[:, 0]
    return mat


def pose_from_vertex_normal(
    vertices: np.ndarray, normals: np.ndarray, shear_mag: float, delta: np.ndarray
) -> np.ndarray:
    """
    Generate SE(3) pose given
    vertices: (N, 3), normals: (N, 3), shear_mag: scalar, delta: (N, 1)
    """
    vertices = np.atleast_2d(vertices)
    normals = np.atleast_2d(normals)

    num_samples = vertices.shape[0]
    T = np.zeros((num_samples, 4, 4))  # transform from point coord to world coord
    T[:, 3, 3] = 1
    T[:, :3, 3] = vertices  # t

    # resolve ambiguous DoF
    """Find rotation of shear_vector so its orientation matches normal: np.dot(Rot, shear_vector) = normal
    https://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another """

    cos_shear_mag = np.random.uniform(
        low=np.cos(shear_mag), high=1.0, size=(num_samples,)
    )  # Base of shear cone
    shear_phi = np.random.uniform(
        low=0.0, high=2 * np.pi, size=(num_samples,)
    )  # Circle of shear cone

    # Axis v = (shear_vector \cross normal)/(||shear_vector \cross normal||)
    shear_vector = np.array(
        [
            np.sqrt(1 - cos_shear_mag**2) * np.cos(shear_phi),
            np.sqrt(1 - cos_shear_mag**2) * np.sin(shear_phi),
            cos_shear_mag,
        ]
    ).T
    shear_vector_skew = skew_matrix(shear_vector)
    v = np.einsum("ijk,jk->ik", shear_vector_skew, normals.T).T
    v = v / np.linalg.norm(v, axis=1).reshape(-1, 1)

    # find corner cases
    check = np.einsum("ij,ij->i", normals, np.array([[0, 0, 1]]))
    zero_idx_up = check > 0.9  # pointing up
    zero_idx_down = check < -0.9  # pointing down

    v_skew, sampledNormals_skew = skew_matrix(v), skew_matrix(normals)

    # Angle theta = \arccos(z_axis \dot normal)
    # elementwise: theta = np.arccos(np.dot(shear_vector,normal)/(np.linalg.norm(shear_vector)*np.linalg.norm(normal)))
    theta = np.arccos(
        np.einsum("ij,ij->i", shear_vector, normals)
        / (np.linalg.norm(shear_vector, axis=1) * np.linalg.norm(normals, axis=1))
    )

    identity_3d = np.zeros(v_skew.shape)
    np.einsum("iij->ij", identity_3d)[:] = 1
    # elementwise: Rot = np.identity(3) + v_skew*np.sin(theta) + np.linalg.matrix_power(v_skew,2) * (1-np.cos(theta)) # rodrigues
    Rot = (
        identity_3d
        + v_skew * np.sin(theta)
        + np.einsum("ijn,jkn->ikn", v_skew, v_skew) * (1 - np.cos(theta))
    )  # rodrigues

    if np.any(zero_idx_up):
        Rot[:3, :3, zero_idx_up] = np.dstack([np.identity(3)] * np.sum(zero_idx_up))
    if np.any(zero_idx_down):
        Rot[:3, :3, zero_idx_down] = np.dstack(
            [np.array([[1, 0, 0], [0, -1, -0], [0, 0, -1]])] * np.sum(zero_idx_down)
        )

    # Rotation about Z axis is still ambiguous, generating random rotation b/w [0, 2pi] about normal axis
    # elementwise: RotDelta = np.identity(3) + normal_skew*np.sin(delta[i]) + np.linalg.matrix_power(normal_skew,2) * (1-np.cos(delta[i])) # rodrigues
    RotDelta = (
        identity_3d
        + sampledNormals_skew * np.sin(delta)
        + np.einsum("ijn,jkn->ikn", sampledNormals_skew, sampledNormals_skew)
        * (1 - np.cos(delta))
    )  # rodrigues

    # elementwise:  RotDelta @ Rot
    tfs = np.einsum("ijn,jkn->ikn", RotDelta, Rot)
    T[:, :3, :3] = np.rollaxis(tfs, 2)
    return T


def cam2gel(cam_pose, cam_dist):
    """
    Convert cam_pose to gel_pose
    """
    cam_tf = torch.eye(4, device=cam_pose.device)
    cam_tf[2, 3] = -cam_dist
    return cam_pose @ cam_tf[None, :]