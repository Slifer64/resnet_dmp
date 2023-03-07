#!/usr/bin/env python3

import numpy as np
import numpy.linalg as linalg

from .quaternion import Quaternion


class Affine3:
    def __init__(self):
        self.linear = np.eye(3)
        self.translation = np.zeros(3)

    @classmethod
    def from_matrix(cls, matrix):
        assert isinstance(matrix, np.ndarray)
        result = cls()
        result.translation = matrix[:3, 3]
        result.linear = matrix[0:3, 0:3]
        return result

    @classmethod
    def from_vec_quat(cls, pos, quat):
        assert isinstance(pos, np.ndarray)
        assert isinstance(quat, Quaternion)
        result = cls()
        result.translation = pos.copy()
        result.linear = quat.rotation_matrix()
        return result

    def matrix(self):
        matrix = np.eye(4)
        matrix[0:3, 0:3] = self.linear
        matrix[0:3, 3] = self.translation
        return matrix

    def quat(self):
        return Quaternion.from_rotation_matrix(self.linear)

    def __copy__(self):
        result = Affine3()
        result.linear = self.linear.copy()
        result.translation = self.translation.copy()
        return result

    def copy(self):
        return self.__copy__()

    def inv(self):
        return Affine3.from_matrix(np.linalg.inv(self.matrix()))

    def __mul__(self, other):
        return self.__rmul__(other)


    def __rmul__(self, other):
        return Affine3.from_matrix(np.matmul(self.matrix(), other.matrix()))

    def __str__(self):
        return self.matrix().__str__()
