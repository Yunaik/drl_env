import os
import inspect
import copy
import imageio
import datetime
from matplotlib import pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import numpy as np

DefaultCfg = {
    "backend"           : "bullet",
    "gpu"               : False,
    "gui"               : False,
    "core"              : 1,
    "solver"            : "pgs",
    "enlarge"           : 1,
    "sim_ts"            : 1000,
    "ctl_ts"            : 500,
    "cmd_ts"            : 25,
    "is_render"         : False,
    "video_name"        : "base_{}.mp4",
    "motor_vel_limit"   : np.inf,
    # "hip_x_kp"          : 0,
    # "hip_x_kd"          : 0,
    # "hip_y_kp"          : 0,
    # "hip_y_kd"          : 0,
    # "knee_kp"          : 0,
    # "knee_kd"          : 0,
    "cam_dist"          : 2.0,
    "cam_yaw"           : 52,
    "cam_pitch"         : -30,
    "records_sim"       : False,
    "fixed"             : False,
    "urdf_root"            : "../../../urdf",
    "_max_episode_steps" : 125,
    # After args are used for Parallel Environment
    "batch"             : 5,
    "distance"          : 5,
    "init_noise"        : False
}
