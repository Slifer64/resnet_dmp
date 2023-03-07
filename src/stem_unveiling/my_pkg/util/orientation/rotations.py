#!/usr/bin/env python3

import numpy as np
import numpy.linalg as linalg
import math


def rot_x(theta):
  rot = np.zeros((3, 3))
  rot[0, 0] = 1
  rot[0, 1] = 0
  rot[0, 2] = 0

  rot[1, 0] = 0
  rot[1, 1] = math.cos(theta)
  rot[1, 2] = -math.sin(theta)

  rot[2, 0] = 0
  rot[2, 1] = math.sin(theta)
  rot[2, 2] = math.cos(theta)

  return rot


def rot_y(theta):
  rot = np.zeros((3, 3))
  rot[0, 0] = math.cos(theta)
  rot[0, 1] = 0
  rot[0, 2] = math.sin(theta)

  rot[1, 0] = 0
  rot[1, 1] = 1
  rot[1, 2] = 0

  rot[2, 0] = -math.sin(theta)
  rot[2, 1] = 0
  rot[2, 2] = math.cos(theta)

  return rot


def rot_z(theta):
  rot = np.zeros((3, 3))
  rot[0, 0] = math.cos(theta)
  rot[0, 1] = -math.sin(theta)
  rot[0, 2] = 0

  rot[1, 0] = math.sin(theta)
  rot[1, 1] = math.cos(theta)
  rot[1, 2] = 0

  rot[2, 0] = 0
  rot[2, 1] = 0
  rot[2, 2] = 1

  return rot


def rotation_is_valid(R, eps=1e-8):
    # Columns should be unit
    for i in range(3):
        error = abs(np.linalg.norm(R[:, i]) - 1)
        if  error > eps:
            raise ValueError('Column ' + str(i) + ' of rotation matrix is not unit (error = ' + str(error) + ') precision: ' + str(eps) + ')')

    # Check that the columns are orthogonal
    if abs(np.dot(R[:, 0], R[:, 1])) > eps:
        raise ValueError('Column 0 and 1 of rotation matrix are not orthogonal (precision: ' + str(eps) + ')')
    if abs(np.dot(R[:, 0], R[:, 2])) > eps:
        raise ValueError('Column 0 and 2 of rotation matrix are not orthogonal (precision: ' + str(eps) + ')')
    if abs(np.dot(R[:, 2], R[:, 1])) > eps:
        raise ValueError('Column 2 and 1 of rotation matrix are not orthogonal (precision: ' + str(eps) + ')')

    # Rotation is right handed
    if not np.allclose(np.cross(R[:, 0], R[:, 1]), R[:, 2], rtol=0, atol=eps):
        raise ValueError('Rotation is not right handed (cross(x, y) != z for precision: ' + str(eps) + ')')
    if not np.allclose(np.cross(R[:, 2], R[:, 0]), R[:, 1], rtol=0, atol=eps):
        raise ValueError('Rotation is not right handed (cross(z, x) != y for precision: ' + str(eps) + ')')
    if not np.allclose(np.cross(R[:, 1], R[:, 2]), R[:, 0], rtol=0, atol=eps):
        raise ValueError('Rotation is not right handed (cross(y, z) != x for precision: ' + str(eps) + ')')

    return True
