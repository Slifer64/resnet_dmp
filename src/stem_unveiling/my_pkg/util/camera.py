import numpy as np
from typing import List, Tuple, Dict, Union
import cv2
from my_pkg.util.orientation.quaternion import Quaternion


class PinholeCameraIntrinsics:
    """
    PinholeCameraIntrinsics class stores intrinsic camera matrix,
    and image height and width.
    """
    def __init__(self, width, height, fx, fy, cx, cy):

        self.width, self.height = width, height
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy

    @classmethod
    def from_params(cls, params):
        width, height = params['width'], params['height']
        fx, fy = params['fx'], params['fy']
        cx, cy = params['cx'], params['cy']
        return cls(width, height, fx, fy, cx, cy)

    def get_intrinsic_matrix(self):
        camera_matrix = np.array(((self.fx, 0, self.cx),
                                  (0, self.fy, self.cy),
                                  (0, 0, 1)))
        return camera_matrix

    def get_focal_length(self):
        return self.fx, self.fy

    def get_principal_point(self):
        return self.cx, self.cy

    def back_project(self, p, z):

        x = (p[0] - self.cx) * z / self.fx
        y = (p[1] - self.cy) * z / self.fy
        return np.array([x, y, z])


class BulletCamera:
    def __init__(self, pos: List, target_pos: List, up_vector: List, pinhole_camera_intrinsics: PinholeCameraIntrinsics,
                 name: str ='bullet_camera'):
        """
        Initializes the camera with the camera instrinsics and the camera position params expressed in the world frame.

        Parameters
        ----------
        pos: list(3), the position (eye) of the camera w.r.t. the world frame
        target_pos: list(3), the target (focus) point of the camera w.r.t. the world frame
        up_vector: list(3), the camera's up-vector w.r.t. the world frame
        pinhole_camera_intrinsics: PinholeCameraIntrinsics, contains the camera intrinsic params
        name
        """

        import pybullet as p
        import math

        self.name = name

        self.pos = np.array(pos)
        self.target_pos = np.array(target_pos)
        self.up_vector = np.array(up_vector)

        # Compute camera pose w.r.t. world frame
        z = self.target_pos - self.pos
        z /= np.linalg.norm(z)
        x = np.cross(-self.up_vector, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R_world_camera = np.eye(3)
        R_world_camera[:, 0] = x
        R_world_camera[:, 1] = y
        R_world_camera[:, 2] = z
        self.cam_pos = self.pos
        self.cam_quat = Quaternion.from_rotation_matrix(R_world_camera)

        # Compute view matrix
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=pos,
                                               cameraTargetPosition=target_pos,
                                               cameraUpVector=up_vector)

        # view_mat = np.array(self.view_matrix).reshape(4, 4).T  # pybullet returns list of 16 on column-major hence the T
        # cam_pose = np.linalg.inv(view_mat)
        # cam_pose[0:3, 1] = -cam_pose[0:3, 1]
        # cam_pose[0:3, 2] = -cam_pose[0:3, 2]
        # self.cam_pos = cam_pose[0:3, 3]
        # self.cam_quat = Quaternion.from_rotation_matrix(cam_pose[0:3, 0:3])

        self.z_near = 0.01
        self.z_far = 5.0
        self.width, self.height = pinhole_camera_intrinsics.width, pinhole_camera_intrinsics.height
        self.fx, self.fy = pinhole_camera_intrinsics.fx, pinhole_camera_intrinsics.fy
        self.cx, self.cy = pinhole_camera_intrinsics.cx, pinhole_camera_intrinsics.cy

        # Compute projection matrix
        fov_h = math.atan(self.height / 2 / self.fy) * 2 / math.pi * 180
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=fov_h, aspect=self.width / self.height,
                                                              nearVal=self.z_near, farVal=self.z_far)

    @classmethod
    def from_pose(cls, pos: List, quat: Quaternion, pinhole_camera_intrinsics: PinholeCameraIntrinsics,
                  name: str ='bullet_camera'):

        rotm = quat.rotation_matrix()
        target_pos = pos + 1.0 * rotm[:, 2] # where the z axis points
        up_vector = -rotm[:, 1]  # the camera's -y axis
        return cls(pos, target_pos, up_vector, pinhole_camera_intrinsics, name)

    def get_pose(self) -> Tuple[np.array, Quaternion]:
        """
        Returns the camera pose w.r.t. the world frame.

        Returns
        -------
        np.array(3), camera position w.r.t. the world frame
        Quaternion, camera orientation w.r.t. the world frame
        """
        return self.cam_pos.copy(), self.cam_quat.copy()

    def get_data(self) -> Tuple[np.array, np.array, np.array]:
        """
        Returns
        -------
        rgb: np.array((h, w, 3), uint8), the RGB image
        depth: np.array((h, w), float32), the depth image, with the distance in meters of each pixel from the camera
        seg: np.array((h, w), int), the segmentation map, with pixel values the body id of each object
        """

        import pybullet as p

        image = p.getCameraImage(self.width, self.height, self.view_matrix, self.projection_matrix,
                                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        height, width = image[1], image[0]
        rgba = np.array(image[2], dtype=np.uint8).reshape(height, width, 4)
        rgb = rgba[:, :, 0:3]
        depth = self.__convert_to_depth_map(np.array(image[3], dtype=np.float32).reshape(height, width))
        seg = np.array(image[4], dtype=np.int).reshape(height, width)
        return rgb, depth, seg

    def get_intrinsics(self) -> Dict:
        """
        Returns
        -------
        dict, contains the came intrinsics: width, height, fx, fy, cx, cy
        """
        return {'width': self.width, 'height': self.height, 'fx': self.fx, 'fy': self.fy, 'cx': self.cx, 'cy': self.cy}

    def __convert_to_depth_map(self, depth_buffer: np.array) -> np.array:
        """
        Converts the depth buffer to depth map.

        Parameters
        ----------
        depth_buffer: np.array()
            The depth buffer as returned from opengl
        """
        depth = self.z_far * self.z_near / (self.z_far - (self.z_far - self.z_near) * depth_buffer)
        return depth


class Rs2Camera:

    def __init__(self, pos: np.array, quat: Quaternion):

        import pyrealsense2 as rs

        assert isinstance(quat, Quaternion), 'quat must be of type Quaternion'

        self.pos = pos.copy()
        self.quat = quat.copy()

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            raise RuntimeError("Couldn't find RGB...")

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        # Get intrinsics
        self.intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

    def __del__(self):
        self.pipeline.stop()

    def get_pose(self) -> Tuple[np.array, Quaternion]:
        """
        Returns the camera pose w.r.t. the world frame.

        Returns
        -------
        np.array(3), camera position w.r.t. the world frame
        Quaternion, camera orientation w.r.t. the world frame
        """
        return self.pos.copy(), self.quat.copy()

    def get_data(self) -> Tuple[np.array, np.array, np.array]:
        """
        Returns
        -------
        rgba: np.array((h, w, 3), uint8), the RGB image
        depth: np.array((h, w), float64), the depth image, with the distance in meters of each pixel from the camera
        seg: np.array((h, w), int), the segmentation map, with pixel values the body id of each object
        """

        # Wait for a coherent pair of frames: depth and color
        while True:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if depth_frame and color_frame:
                break

        # # Grab new intrinsics (may change by decimation)
        # self.intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()) / 1000  # convert mm to meters
        rgb_img = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)

        depth_dim = depth_image.shape
        color_dim = rgb_img.shape
        if depth_dim != color_dim:
            rgb_img = cv2.resize(rgb_img, dsize=(depth_dim[1], depth_dim[0]), interpolation=cv2.INTER_AREA)

        seg_image = np.zeros(rgb_img.shape, dtype=np.int)

        return rgb_img, depth_image, seg_image

    def get_intrinsics(self) -> Dict:
        """
        Returns
        -------
        dict, contains the came intrinsics: width, height, fx, fy, cx, cy
        """
        return {'width': self.intrinsics.width, 'height': self.intrinsics.height,
                'fx': self.intrinsics.fx, 'fy': self.intrinsics.fy,
                'cx': self.intrinsics.ppx, 'cy': self.intrinsics.ppy}


class BagCamera:

    def __init__(self, bag_data_path, random=False, seed=0, ind=[]):

        import os

        self.pos = np.array([0, 0, 0])
        self.quat = Quaternion()
        self.rng = np.random.RandomState()
        self.rng.seed(seed)

        self.bag_data_path = bag_data_path
        self.state_dirs = os.listdir(bag_data_path)
        self.state_dirs.sort()
        self.bag_size = len(self.state_dirs)
        self.random = random

        if ind:
            self.ind = ind
        else:
            self.ind = list(range(self.bag_size))

        self.current_i = 0

        # import shutil
        # new_bag_path = '../logs/real_dataset2/'
        # if os.path.exists(new_bag_path):
        #     shutil.rmtree(new_bag_path)
        # os.makedirs(new_bag_path)
        # for i, s in enumerate(self.state_dirs):
        #     state_path = self.bag_data_path + s
        #     print('Loading state: ' + state_path)
        #     state = BulletState.load(state_path)
        #     state.save_compressed = True
        #     state_path = new_bag_path + 'state_' + str(i).zfill(3)
        #     print('Saving state: ' + state_path)
        #     os.makedirs(state_path)
        #     state.save(state_path)
        # exit()

        # Set intrinsics
        self.intrinsics = {'width': 640, 'height': 480, 'fx': 463, 'fy': 463, 'cx': 320, 'cy': 240}

    def get_pose(self) -> Tuple[np.array, Quaternion]:
        """
        Returns the camera pose w.r.t. the world frame.

        Returns
        -------
        np.array(3), camera position w.r.t. the world frame
        Quaternion, camera orientation w.r.t. the world frame
        """
        return self.pos.copy(), self.quat.copy()

    def get_data(self) -> Tuple[np.array, np.array, np.array]:
        """
        Returns
        -------
        rgba: np.array((h, w, 3), uint8), the RGB image
        depth: np.array((h, w), float32), the depth image, with the distance in meters of each pixel from the camera
        seg: np.array((h, w), int), the segmentation map, with pixel values the body id of each object
        """

        if self.current_i == 0 and self.random:
            self.rng.shuffle(self.ind)

        state_path = self.bag_data_path + self.state_dirs[self.ind[self.current_i]]
        self.current_i = (self.current_i + 1) % len(self.ind)

        print('Loading state: ' + state_path)
        state = BulletState.load(state_path)

        return state['rgb'], state['depth'], state['seg']

    def get_intrinsics(self) -> Dict:
        """
        Returns
        -------
        dict, contains the came intrinsics: width, height, fx, fy, cx, cy
        """
        return self.intrinsics

    def num_of_frames(self):

        return self.bag_size

    def get_frame(self, frame_id):
        state_path = self.bag_data_path + self.state_dirs[frame_id]
        state = BulletState.load(state_path)
        return state['rgb'], state['depth'], state['seg']

    def export_all_images(self, path):

        for i in range(self.num_of_frames()):
            rgb = self.get_frame(i)[0]
            cv2.imwrite(path + '/rgb_' + str(i) + '.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


class TopicCamera:
    def __init__(self, image_topic: str, pos: np.array, quat: Quaternion):

        import rospy
        from sensor_msgs.msg import Image as sensor_msg_Image
        from my_pkg.util.cv_bridge import CvBridge
        import threading
        
        assert isinstance(quat, Quaternion), 'quat must be of type Quaternion'

        self.pos = pos.copy()
        self.quat = quat.copy()

        self.image = None
        self.cv_br = CvBridge()
        self.updated = False
        self.lock = threading.Lock()

        self.intrinsics = {'width': 640, 'height': 480,
                           'fx': 913.9525146484375, 'fy': 912.655029296875,
                           'cx': 633.8968505859375, 'cy': 358.45623779296875}

        rospy.Subscriber(image_topic, sensor_msg_Image, self.callback)

    def read_intrinsics_from_topic(self, topic: str):
        
        import rospy
        from sensor_msgs.msg import CameraInfo as sensor_msg_CamInfo

        def cam_info_callback(got_msg_ptr, msg: sensor_msg_CamInfo):
            self.intrinsics['height'] = msg.height
            self.intrinsics['width'] = msg.width
            K = np.array(msg.K).reshape(3, 3)
            self.intrinsics['fx'] = K[0, 0]
            self.intrinsics['fy'] = K[1, 1]
            self.intrinsics['cx'] = K[0, 2]
            self.intrinsics['cy'] = K[1, 2]
            got_msg_ptr[0] = True

        got_msg = [False]
        sub = rospy.Subscriber(topic, sensor_msg_CamInfo, lambda msg: cam_info_callback(got_msg, msg))
        while not got_msg[0]:
            rospy.rostime.wallsleep(0.1)
        sub.unregister()

    def get_data(self) -> Tuple[np.array, np.array, np.array]:
        """
        Returns
        -------
        rgba: np.array((h, w, 3), uint8), the RGB image
        depth: np.array((h, w), float64), the depth image, with the distance in meters of each pixel from the camera
        seg: np.array((h, w), int), the segmentation map, with pixel values the body id of each object
        """

        rgb_img = self.get_rgb_image()

        return rgb_img, None, None

    def get_pose(self) -> Tuple[np.array, Quaternion]:
        """
        Returns the camera pose w.r.t. the world frame.

        Returns
        -------
        np.array(3), camera position w.r.t. the world frame
        Quaternion, camera orientation w.r.t. the world frame
        """
        return self.pos.copy(), self.quat.copy()
 
    def get_intrinsics(self) -> Dict:
        return self.intrinsics.copy()

    def callback(self, msg):
        # rospy.loginfo('Image received...')
        with self.lock:
            self.image = self.cv_br.imgmsg_to_cv2(msg)
            self.updated = True


    def get_rgb_image(self):

        while not self.updated:
            pass

        with self.lock:
            return self.image
            # return cv2.resize(self.image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            # return cv2.cvtColor(cv2.resize(self.image, dsize=(640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)