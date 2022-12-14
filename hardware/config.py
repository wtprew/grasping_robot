import numpy as np
from geometry_msgs.msg import Quaternion
import os


#change these paths to the location of your robot intrinsics and where you would like to save your data
home_directory = os.path.expanduser( '~' )
homedir = os.path.join(home_directory, 'ros_ws/intrinsics')
# DATA COLLECTION
DATAPATH = '~/grasping_robot/results/h5py/'  # path for saving data collection samples

# CONTROL
PRELIFT_HEIGHT = .2
Z_OFFSET = 0.04
Z_MIN = .03
CONTROL_NOISE_COEFFICIENT_ALPHA = 0.9949 # recompute if necessary
CONTROL_NOISE_COEFFICIENT_BETA = 0.

# CAMERA

#camera to robot
CALIBRATION_MATRIX = np.asarray([
        [ 0.03413103, -0.99916265,  0.02256256,  0.20542988],
        [-0.99933083, -0.03441645, -0.0123852 , -0.00566434],
        [ 0.01315135, -0.02212474, -0.99966871,  0.60809941],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]
)

#robot to camera
# CALIBRATION_MATRIX = np.asarray([
#       [ 0.03413103, -0.99933083,  0.01315135, -0.02066941],
#       [-0.99916265, -0.03441645, -0.02212474,  0.21851696],
#       [ 0.02256256, -0.0123852 , -0.99966871,  0.60319277],
#       [ 0.        ,  0.        ,  0.        ,  1.        ]]))


# DEPTH_K = np.array([[476.7152099609375, 0.0, 315.1253662109375],
                    # [0.0, 476.7152099609375, 245.31260681152344],
                    # [0.0, 0.0, 1.0]])
# RGB_K = np.array([[621.4144287109375, 0.0, 303.1456298828125],
                #   [0.0, 621.4144897460938, 238.14071655273438],
                #   [0.0, 0.0, 1.0]])
# DEPTH_TO_RGB_RT = np.matrix([[0.9999925494194031, -0.003690361976623535, 0.0011324022198095918, 0.025699999183416367],
                            #  [0.003694217884913087, 0.9999873042106628, -0.0034221974201500416, 0.0007007527747191489],
                            #  [-0.0011197587009519339, 0.0034263553097844124, 0.9999935030937195, 0.00415800791233778],
                            #  [0., 0., 0., 1.]])

RGB_IMAGE_TOPIC = '/camera/color/image_raw'
DEPTH_IMAGE_TOPIC = '/camera/aligned_depth_to_color/image_raw'
POINTCLOUD_TOPIC = '/camera/depth_registered/points'
RGB_CAMERA_INFO_TOPIC = '/camera/color/camera_info'
DEPTH_CAMERA_INFO_TOPIC = '/camera/depth/camera_info'

MAX_DEPTH = 700.0
PC_BOUNDS = [(0.35, .17),
			(0.35, -.17),
			(0.1, -.17),
			(0.1, .17)]
HEIGHT_BOUNDS = (.01, .17)
PC_BASE_PATH = 'pc_base.npy'

# ARENA BOUNDARIES
END_EFFECTOR_BOUNDS = [(.35,  .17), # Top Left Corner
                        (.35, -.17), # Top Right Corner 
                        (.1, -.17), # Bottom Right Corner
                        (.1,  .17)] # Bottom Left Corner

# CONTROLLER CONSTANTS AND PREPLANNED ROUTINES
PREGRASP_VALUES = [0, 0, 0, 1.5708, 0]
NEUTRAL_VALUES = [0, -1.4839419194602816, 1.4971652489763858, 0, 0]
INITIAL_VALUES = [0, -1.560058461279697, 1.05231082048955, -1.1504855909142309, 0]
RESET_VALUES = [0, -1.3054176504906807, 1.033903051034922, -1.3499030933393643, 0]
PULLED_BACK_VALUES = [0.0, -1.57, 0.7, 1.57, 0.0]
DROP_VALUES = [1.1566215140657734, 0, 0, 1.70, 0]

DOWN_ORIENTATION = Quaternion(x=0, y=-1, z=-0, w=0)

GRIPPER_DROP = [0.031, 0.031]
GRIPPER_OPEN = [0.031, 0.031]
GRIPPER_CLOSED = [0.002, 0.002]

TL_CORNER = [0.2270291566070749, 0.9909515889741242, -0.8574952604280734, 1.552388557340269, 0.0]
TL_CORNER_POSE = [.3, .15]
BL_CORNER = [0.39269908169872414, 0.1641359443037636, 0.7531845668518499, 0.7378447589729934, 0.0]
BL_CORNER_POSE = [.1, .15]
TR_CORNER = [-0.34054373491061235, 1.000155473701438, -0.8544272988523022, 1.4971652489763858, 0.0]
TR_CORNER_POSE = [.3, -.15]
BR_CORNER = [-0.4387185053352934, 0.2561747915769021, 0.5890486225480862, 0.9863496466104673, 0.0]
BR_CORNER_POSE = [.1, -.15]

UPRIGHT_DROP = [0.0, -0.34821363885004053, 0.7470486437003072, 1.2041749184902284, -0.0051132692929521375, 0.003]
UPRIGHT_NEUTRAL = [0.0, -1.1642914180052018, 1.239456476611598, -0.10737865515199488, 0.0051132692929521375, 0.003]
UPRIGHT_RESET = [0.0, -0.4, 0.7915340865489908, -0.3942330624866098, -0.010226538585904275, 0.003]

# RL BOUNDS
BOUNDS_FLOOR = -0.05
WORKSPACE_CENTRE_X = 0.20
BOUNDS_LEFTWALL = 0.17
BOUNDS_RIGHTWALL = -0.17
BOUNDS_FRONTWALL = 0.10
BOUNDS_BACKWALL = 0.35

JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'gripper_joint']
JOINT_MIN = np.array([-3.1, -1.571, -1.571, -1.745, -2.617, 0.003])
JOINT_MAX = np.array([3.1, 1.571, 1.571, 1.745, 2.617, 0.03])