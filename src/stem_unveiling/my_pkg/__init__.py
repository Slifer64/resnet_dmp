IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

MP_KERNELS = 8

ROBOT_DESCRIPTION = '../assets/robot_description/robot_description_with_wheels.urdf'
# ROBOT_DESCRIPTION = '../assets/robot_description/robot_description.urdf'

CROP = 240

TARGET_DEPTH = 0.36

TIME_STEP = 1. / 240

LOG_DIR = None
DATASET_DIR = None

get_leaves_mask = None

SHOW_DEBUG_PLOT = False


def index2pxl(index):
    return [index[1], index[0]]

# from bacchus_bullet.util.orientation.quaternion import Quaternion
# from bacchus_bullet.util.orientation.affine3 import Affine3

# from bacchus_bullet.camera import PinholeCameraIntrinsics
# from bacchus_bullet.state import BulletState, CameraState
# from bacchus_bullet.util.info import mkdir, mkdirs, input_warning, print_warning, print_info

# import bacchus_bullet.util.debug as debug

from my_pkg.util.timer import Timer
global_timer = Timer()
tic = global_timer.tic
toc = global_timer.toc
