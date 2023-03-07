#!/usr/bin/env python3

import numpy as np
import numpy.linalg as linalg
import math
from typing import List, Tuple

class Quaternion:

    # ============== Constructors ================

    def __init__(self, w=1., x=0., y=0., z=0.):
        """
        Constructs a quaternion. The quaternion is not normalized.

        Arguments:
        w -- float, the scalar part
        x -- float, the element along the x-axis of the vector part
        y -- float, the element along the y-axis of the vector part
        z -- float, the element along the z-axis of the vector part
        """
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_vector(cls, vector, convention='wxyz'):
        """
        Constructs a quaternion from a vector.

        Arguments:
        vector -- vector with 4 elements (must implement operator '[i]')
        convention -- 'wxyz' or 'xyzw' corresponding to the order of the elements in 'vector'

        Returns:
        A Quaterion object.
        """
        if convention == 'wxyz':
            return cls(w=vector[0], x=vector[1], y=vector[2], z=vector[3])
        elif convention == 'xyzw':
            return cls(w=vector[3], x=vector[0], y=vector[1], z=vector[2])
        else:
            raise ValueError('Order is not supported.')

    @classmethod
    def from_rotation_matrix(cls, R):
        """
        Constructs a quaternion from a rotation matrix.

        Arguments:
        R -- rotation matrix as 3x3 array-like object (must implement operator '[i,j]')

        Returns:
        A Quaterion object.
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

        result = q / linalg.norm(q)
        return cls(w=result[0], x=result[1], y=result[2], z=result[3])

    @classmethod
    def from_roll_pitch_yaw(cls, x):
        """
        Constructs a quaternion from roll-pitch-yaw angles.

        Arguments:
        x -- stores the roll-pitch-yaw angles (must implement operator '[i]')

        Returns:
        A Quaterion object.
        """
        roll = x[0]
        pitch = x[1]
        yaw = x[2]

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        return cls(w=cr * cp * cy + sr * sp * sy,
                   x=sr * cp * cy - cr * sp * sy,
                   y=cr * sp * cy + sr * cp * sy,
                   z=cr * cp * sy - sr * sp * cy)

    @classmethod
    def from_tait_bryan(self, angles, convention='z1y2x3'):
        # see https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
        c1 = np.cos(angles[0])
        c2 = np.cos(angles[1])
        c3 = np.cos(angles[2])
        s1 = np.sin(angles[0])
        s2 = np.sin(angles[1])
        s3 = np.sin(angles[2])
        if convention == 'z1y2x3':
            rot = np.array([[c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
                            [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
                            [-s2, c2 * s3, c2 * s3]])
        else:
            raise RuntimeError('Quaternion class: Convention is not supported.')

        return self.from_rotation_matrix(rot)

    # ============== Misc ================

    def __getitem__(self, item):
        """
        Applies the [] operator to the quaternion, treating it as an array [w, x, y, z].

        Arguments:
        item -- int between [0, 3] or any other valid slice

        Returns:
        The element at position 'item' or the corresponding slice.
        """

        quat = np.array([self.w, self.x, self.y, self.z])
        return quat[item]

    def __copy__(self):
        """
        Returns a deep copy of 'self'.
        """
        return Quaternion(w=self.w, x=self.x, y=self.y, z=self.z)

    def copy(self):
        return self.__copy__()

    def __call__(self, convention='wxyz') -> np.array:
        """
        See @self.as_vector
        """

        return self.as_vector(convention)

    def as_vector(self, convention='wxyz') -> np.array:
        """
        Returns the quaternion as an np.array.

        Arguments:
        convention -- 'wxyz' or xyzw'

        Returns:
        The quaternion as an np.array in the format defined by 'convention'.
        """
        if convention == 'wxyz':
            return np.array([self.w, self.x, self.y, self.z])
        elif convention == 'xyzw':
            return np.array([self.x, self.y, self.z, self.w])
        else:
            raise RuntimeError

    def vec(self) -> np.array:
        """
        Returns: np.array(3), the vector part of the quaternion.
        """
        return np.array([self.x, self.y, self.z])

    def __str__(self):
        return '%.3f' % self.w + " + " + '%.3f' % self.x + "i +" + '%.3f' % self.y + "j + " + '%.3f' % self.z + "k"

    # ============== Mathematical operations ================

    def mul(self, quat2: 'Quaternion'):
        """
        Implements the quaternion product between 'self' and another quaternion.

        Arguments:
        quat2 -- Quaternion

        Returns:
        Quaternion, the quaternion product of 'self' and 'quat2'
        """

        assert isinstance(quat2, Quaternion), "'second' must be a Quaternion"

        w1, v1 = self.w, self.vec()
        w2, v2 = quat2.w, quat2.vec()
        w = w1 * w2 - np.dot(v1, v2)
        v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
        return Quaternion(w, *v)

    def inv(self):
        """
        Returns: Quaternion, the quaternion inverse of 'self'.
        """

        temp = pow(self.w, 2) + pow(self.x, 2) + pow(self.y, 2) + pow(self.z, 2)
        return Quaternion(self.w/temp, -self.x/temp, -self.y/temp, -self.z/temp)

    def diff(self, quat2: 'Quaternion'):
        """
        Calculates the quaternion difference, i.e. the product of 'self' with inverse('quat2').

        Arguments:
        quat2 -- Quaternion

        Returns:
        Quaternion, the difference between 'self' and 'quat2'.

        Note: Since quat2 and -quat2 represent the same orientation, the minimum difference between 'self' and 'quat2'
        is calculated.
        """

        assert isinstance(quat2, Quaternion), "'quat2' must be a Quaternion"

        if np.dot(self(), quat2()) < 0:
            quat2 = quat2.copy().negate()

        return self.mul(quat2.inv())

    @staticmethod
    def log(quat: 'Quaternion', zero_tol=1e-16):
        """
        Calculates the quaternion logarithm as 2*log(quat)

        Arguments:
        quat -- A @Quaternion object.
        zero_tol -- Zero tolerance threshold (optional, default=1e-16)

        Returns:
        qlog -- np.array(3), the quaternion logarithm of quat.
        """

        assert isinstance(quat, Quaternion), "'quat' must be a Quaternion"

        v = quat.vec()
        v_norm = linalg.norm(v)

        if v_norm > zero_tol:
            qlog = 2 * math.atan2(v_norm, quat.w) * v / v_norm
        else:
            qlog = np.array([0, 0, 0])

        return qlog

    @staticmethod
    def exp(qlog: np.array, zero_tol=1e-16):
        """
        Calculates the quaternion exponential as exp(2*qlog)

        Arguments:
        qlog -- np.array(3)
        zero_tol -- Zero tolerance threshold (optional, default=1e-16)

        Returns:
        quat -- @Quaternion, the quaternion exponential of qlog.
        """

        norm_qlog = linalg.norm(qlog)
        theta = norm_qlog

        if theta > zero_tol:
            quat = Quaternion(math.cos(theta / 2.), *(math.sin(theta / 2) * qlog / norm_qlog))
        else:
            quat = Quaternion(1, 0, 0, 0)

        return quat

    @staticmethod
    def logDot_to_rotVel(logQ_dot: np.array, quat: 'Quaternion'):
        """
        Calculates the rotational velocity, corresponding to dlog(Q)/dt.

        Arguments:
            logQ_dot -- np.array(3), the 1st order time derivative of the quaternion logarithm of 'Q'
            Q -- Quaternion (must be unit quaternion!)

        Returns:
            rot_vel -- np.array(3), rotational velocity corresponding to dlog(Q)/dt
        """

        assert isinstance(quat, Quaternion), "'quat' must be a Quaternion"

        logQ_dot = logQ_dot.squeeze()  # just to make sure...
        assert logQ_dot.shape[0] == 3, 'logQ_dot must be an np.array(3)'

        # return np.matmul(jacob_rotVel_qLog(Q), logQ_dot)

        w = quat.w
        if (1 - math.fabs(w)) <= 1e-8:
            rot_vel = logQ_dot.copy()
        else:
            v = quat.vec()
            norm_v = linalg.norm(v)
            k = v / norm_v  # axis of rotation
            s_th = norm_v  # sin(theta/2)
            c_th = w  # cos(theta/2)
            th = math.atan2(s_th, c_th)  # theta/2
            Pk_qdot = k * np.dot(k, logQ_dot)  # projection of logQ_dot on k
            rot_vel = Pk_qdot + (logQ_dot - Pk_qdot) * s_th * c_th / th + (s_th ** 2 / th) * np.cross(k, logQ_dot)

        return rot_vel

    @staticmethod
    def logDDot_to_rotAccel(logQ_ddot: np.array, rotVel: np.array, quat: 'Quaternion'):

        assert isinstance(quat, Quaternion), "'quat' must be a Quaternion"

        logQ_ddot = logQ_ddot.squeeze()
        rotVel = rotVel.squeeze()

        w = quat.w
        if (1-math.fabs(w)) <= 1e-8:
            rot_accel = logQ_ddot
        else:
            v = quat.vec()
            norm_v = linalg.norm(v)
            k = v / norm_v # axis of rotation
            s_th = norm_v # sin(theta/2)
            c_th = w     # cos(theta/2)
            th = math.atan2(s_th, c_th) # theta/2

            Pk_rotVel = k*np.dot(k, rotVel) # projection of rotVel on k
            qdot = Pk_rotVel + (rotVel - Pk_rotVel)*th*c_th/s_th - th*np.cross(k, rotVel)

            qdot_bot = qdot - np.dot(k, qdot)*k # projection of qdot on plane normal to k
            k_dot = 0.5*qdot_bot/th
            th2_dot = 0.5*np.dot(k, qdot)

            sc_over_th = (s_th * c_th) / th

            JnDot_qdot = (1 - sc_over_th)*(np.dot(k, qdot)*k_dot + np.dot(k_dot, qdot)*k) + \
                    (s_th**2/th)*np.cross(k_dot, qdot) + \
                    ( (1 - 2*s_th**2)/th - sc_over_th/th )*th2_dot*qdot_bot + \
                    (2*sc_over_th - (s_th/th)**2)*th2_dot*np.cross(k, qdot)

            Pk_qddot = np.dot(k, logQ_ddot)*k # projection of qddot on k
            Jn_qddot = Pk_qddot + (logQ_ddot - Pk_qddot)*sc_over_th + (s_th**2/th)*np.cross(k, logQ_ddot)

            rot_accel = Jn_qddot + JnDot_qdot

        return rot_accel

    def normalize(self) -> 'Quaternion':
        """
        Normalizes 'self' so that it is a unit quaternion.
        """
        q = self.as_vector()
        q = q / np.linalg.norm(q)
        self.w = q[0]
        self.x = q[1]
        self.y = q[2]
        self.z = q[3]

        return self

    def negate(self) -> 'Quaternion':
        """
        Inverts the sign of all elements of 'self'. Notice that the represented orientation doesn't alter.
        """

        self.w, self.x, self.y, self.z = -self.w, -self.x, -self.y, -self.z
        return self

    # ============== Conversions to other orientation representations ================

    def rotation_matrix(self) -> np.array:
        """
        Returns: np.array(3), the rotation matrix of this quaternion.
        """
        n = self.w
        ex = self.x
        ey = self.y
        ez = self.z

        R = np.eye(3)

        R[0, 0] = 2 * (n * n + ex * ex) - 1
        R[0, 1] = 2 * (ex * ey - n * ez)
        R[0, 2] = 2 * (ex * ez + n * ey)

        R[1, 0] = 2 * (ex * ey + n * ez)
        R[1, 1] = 2 * (n * n + ey * ey) - 1
        R[1, 2] = 2 * (ey * ez - n * ex)

        R[2, 0] = 2 * (ex * ez - n * ey)
        R[2, 1] = 2 * (ey * ez + n * ex)
        R[2, 2] = 2 * (n * n + ez * ez) - 1

        return R

    def roll_pitch_yaw(self) -> Tuple[float, float, float]:
        # Calculate q2 using lots of information in the rotation matrix.
        # Rsum = abs( cos(q2) ) is inherently non-negative.
        # R20 = -sin(q2) may be negative, zero, or positive.
        R = self.rotation_matrix()
        R22 = R[2, 2]
        R21 = R[2, 1]
        R10 = R[1, 0]
        R00 = R[0, 0]
        Rsum = np.sqrt((R22 * R22 + R21 * R21 + R10 * R10 + R00 * R00) / 2)
        R20 = R[2, 0]
        q2 = np.arctan2(-R20, Rsum)

        e0 = self.w
        e1 = self.x
        e2 = self.y
        e3 = self.z
        yA = e1 + e3
        xA = e0 - e2
        yB = e3 - e1
        xB = e0 + e2
        epsilon = 1e-10
        isSingularA = (np.abs(yA) <= epsilon) and (np.abs(xA) <= epsilon)
        isSingularB = (np.abs(yB) <= epsilon) and (np.abs(xB) <= epsilon)
        if isSingularA:
            zA = 0.0
        else:
            zA = np.arctan2(yA, xA)
        if isSingularB:
            zB = 0.0
        else:
            zB = np.arctan2(yB, xB)
        q1 = zA - zB
        q3 = zA + zB

        # If necessary, modify angles q1 and/or q3 to be between -pi and pi.
        if q1 > np.pi:
            q1 = q1 - 2 * np.pi
        if q1 < -np.pi:
            q1 = q1 + 2 * np.pi
        if q3 > np.pi:
            q3 = q3 - 2 * np.pi
        if q3 < -np.pi:
            q3 = q3 + 2 * np.pi

        return (q1, q2, q3)


quatLog = Quaternion.log
quatExp = Quaternion.exp
quatLogDot_to_rotVel = Quaternion.logDot_to_rotVel


def jacob_rotVel_qLog(quat: Quaternion, zero_tol=1e-8):
    """
    Calculates the Jacobian that maps 1st order time derivatives of quaternion logarithm to rotational velocity.

    Arguments:
        quat -- Quaternion (must be unit quaternion!)
        zero_tol -- Zero tolerance threshold (optional, default=1e-8)

    Returns:
        np.array(3,3), Jacobian that maps dlog(Q)/dt to rotational velocity
    """

    if (1-math.fabs(quat.w)) < zero_tol:
        return np.eye(3)

    v = quat.vec()
    norm_v = linalg.norm(v)
    k = v / norm_v
    s_th = norm_v
    c_th = quat.w
    th = math.atan2(s_th, c_th)
    kkT = np.outer(k, k)

    skew = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

    return kkT + (s_th*c_th/th)*(np.eye(3) - kkT) + (s_th**2/th)*skew(k)



def qLogDot_to_rotVel_deprecated(logQ_dot: np.array, Q: Quaternion, zero_tol=1e-8) -> np.array:

    JQq = np.zeros((4, 3))
    if (1 - math.fabs(Q.w)) < zero_tol:
        JQq[1:, :] = np.eye(3)
    else:
        w = Q.w
        v = Q.vec()
        norm_v = linalg.norm(v)
        eta = v / norm_v
        s_th = norm_v
        c_th = w
        th = math.atan2(s_th, c_th)
        Eta = np.outer(eta, eta)
        JQq[0, :] = -0.5 * s_th * eta.reshape(1, -1)
        JQq[1:, :] = 0.5 * ((np.eye(3) - Eta) * s_th / th + c_th * Eta)

    rotVel = 2*(Quaternion(*np.matmul(JQq, logQ_dot).squeeze()).mul(Q.inv())[1:])
    return rotVel