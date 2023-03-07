#!/usr/bin/env python3

import numpy as np
import numpy.matlib as matlib
import numpy.linalg as linalg
import math

from .quaternion import Quaternion

import logging
logger = logging.getLogger('robamine.utils.orientation')


def quat2rot(q, shape="wxyz"):
    """
    Transforms a quaternion to a rotation matrix.
    """
    if shape == "wxyz":
        n  = q[0]
        ex = q[1]
        ey = q[2]
        ez = q[3]
    elif shape == "xyzw":
        n  = q[3]
        ex = q[0]
        ey = q[1]
        ez = q[2]
    else:
        raise RuntimeError("The shape of quaternion should be wxyz or xyzw. Given " + shape + " instead")

    R = matlib.eye(3)

    R[0, 0] = 2 * (n * n + ex * ex) - 1
    R[0, 1] = 2 * (ex * ey - n * ez)
    R[0, 2] = 2 * (ex * ez + n * ey)

    R[1, 0] = 2 * (ex * ey + n * ez)
    R[1, 1] = 2 * (n * n + ey * ey) - 1
    R[1, 2] = 2 * (ey * ez - n * ex)

    R[2, 0] = 2 * (ex * ez - n * ey)
    R[2, 1] = 2 * (ey * ez + n * ex)
    R[2, 2] = 2 * (n * n + ez * ez) - 1

    return R;


def rot2quat(R, shape="wxyz"):
    """
    Transforms a rotation matrix to a quaternion.
    """

    q = [None] * 4

    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qwh
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S

    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
      S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
      q[0] = (R[2, 1] - R[1, 2]) / S
      q[1] = 0.25 * S
      q[2] = (R[0, 1] + R[1, 0]) / S
      q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
      S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
      q[0] = (R[0, 2] - R[2, 0]) / S
      q[1] = (R[0, 1] + R[1, 0]) / S
      q[2] = 0.25 * S
      q[3] = (R[1, 2] + R[2, 1]) / S
    else:
      S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
      q[0] = (R[1, 0] - R[0, 1]) / S
      q[1] = (R[0, 2] + R[2, 0]) / S
      q[2] = (R[1, 2] + R[2, 1]) / S
      q[3] = 0.25 * S

    return q / linalg.norm(q);


def get_homogeneous_transformation(pose):
    """
    Returns a homogeneous transformation given a pose [position, quaternion]
    """
    M = matlib.zeros((4, 4))
    p = pose[0:3]
    R = quat2rot(pose[3:7])
    for i in range(0, 3):
        M[i, 3] = p[i]
        for j in range(0, 3):
            M[i, j] = R[i, j]
    M[3, 3] = 1
    return M


def get_pose_from_homog(M):
    """
    Returns a pose [position, quaternion] from a homogeneous matrix
    """
    p = [None] * 3
    R = matlib.eye(3)

    for i in range(0, 3):
        p[i] = M[i, 3]
        for j in range(0, 3):
            R[i, j] = M[i, j]

    q = rot2quat(R)
    return np.concatenate((p, q))


def skew_symmetric(vector):
    output = np.zeros((3, 3))
    output[0, 1] = -vector[2]
    output[0, 2] =  vector[1]
    output[1, 0] =  vector[2]
    output[1, 2] = -vector[0]
    output[2, 0] = -vector[1]
    output[2, 1] =  vector[0]
    return output


def screw_transformation(position, orientation):
    output = np.zeros((6, 6))
    output[0:3, 0:3] = orientation
    output[3:6, 3:6] = orientation
    output[3:6, 0:3] = np.matmul(skew_symmetric(position), orientation)
    return output


def rotation_6x6(orientation):
    output = np.zeros((6, 6))
    output[0:3, 0:3] = orientation
    output[3:6, 3:6] = orientation
    return output


def rot2angleaxis(R):
    angle = np.arccos((np.trace(R) - 1) / 2)
    if angle == 0:
        logger.warn('Angle is zero (the rotation identity)')
        axis = None
    else:
        axis = (1 / (2 * np.sin(angle))) * np.array([R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]])
        if np.linalg.norm(axis) == 0:
            logger.warn('Axis is zero (the rotation is around an axis exactly at pi)')
            axis = None
        else:
            axis = axis / np.linalg.norm(axis)
    return angle, axis


def angleaxis2rot(angle, axis):
    c = math.cos(angle)
    s = math.sin(angle)
    v = 1 - c
    kx = axis[0]
    ky = axis[1]
    kz = axis[2]

    R = np.eye(3)
    R[0, 0] = pow(kx, 2) * v + c
    R[0, 1] = kx * ky * v - kz * s
    R[0, 2] = kx * kz * v + ky * s

    R[1, 0] = kx * ky * v + kz * s
    R[1, 1] = pow(ky, 2) * v + c
    R[1, 2] = ky * kz * v - kx * s

    R[2, 0] = kx * kz * v - ky * s
    R[2, 1] = ky * kz * v + kx * s
    R[2, 2] = pow(kz, 2) * v + c

    return R


def transform_points(points, pos, quat, inv=False):
    '''Points are w.r.t. {A}. pos and quat is the frame {A} w.r.t {B}. Returns the list of points experssed w.r.t.
    {B}.'''
    assert points.shape[1] == 3
    matrix = np.eye(4)
    matrix[0:3, 3] = pos
    matrix[0:3, 0:3] = quat.rotation_matrix()
    if inv:
        matrix = np.linalg.inv(matrix)

    transformed_points = np.transpose(np.matmul(matrix, np.transpose(
        np.concatenate((points, np.ones((points.shape[0], 1))), axis=1))))[:, :3]
    return transformed_points


def transform_poses(poses, target_frame, target_inv=False):
    """
    Poses in Nx7 vectors with position & quaternion w.r.t. {A}.
    target_frame {B} 7x1 vector the target frame to express poses (w.r.t. again {A}
    """
    matrix = np.eye(4)
    matrix[0:3, 3] = target_frame[0:3]
    matrix[0:3, 0:3] = Quaternion.from_vector(target_frame[3:7]).rotation_matrix()
    transformed_poses = np.zeros((poses.shape[0], 7))
    matrix = np.linalg.inv(matrix)
    if target_inv:
        matrix = np.linalg.inv(matrix)

    for i in range(poses.shape[0]):
        matrix2 = np.eye(4)
        matrix2[0:3, 3] = poses[i, 0:3]
        matrix2[0:3, 0:3] = Quaternion.from_vector(poses[i, 3:7]).rotation_matrix()
        transformed = np.matmul(matrix, matrix2)
        transformed_poses[i, 0:3] = transformed[0:3, 3]
        transformed_poses[i, 3:7] = Quaternion.from_rotation_matrix(transformed[0:3, 0:3]).as_vector()
    return transformed_poses

