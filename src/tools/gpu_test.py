from b3px_gym.b3px_env.parallel import laikago_gym_env_gpu
import copy
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
	"video_name"        : "laikago_{}.mp4",
    "motor_vel_limit"   : np.inf,
    "motor_kp"          : 400,
    "motor_kd"          : 10,
    "cam_dist"          : 2.0,
    "cam_yaw"           : 52,
    "cam_pitch"         : -30,
	"records_sim"       : False,
	"fixed"             : False,
	# "urdf_root"         : "/home/syslot/DevSpace/WALLE/src/pybullet_demo/urdf"
    "urdf_root"			: "../../../urdf",

	# After args are used for Parallel Environment
	"batch"             : 5,
	"distance"          : 5,
}

cfg = copy.deepcopy(DefaultCfg)
cfg['backend'] = 'physx'
cfg['gui'] = True
cfg['gpu'] = True
cfg['solver'] = 'tgs'
cfg['urdf_root'] = '../urdf'
cfg['cam_dist'] = 1000
cfg['core'] = 2
cfg['batch'] = 32

env = laikago_gym_env_gpu.LaikagoB3PxEnvGpu(cfg)

acts = np.array([-0.35, -0.05, -0.5, -0.35, -0.05, -0.5, -0.35, -0.05, -0.5, -0.35, -0.05, -0.5] * cfg['batch']).reshape( cfg['batch'], -1)

for i in range(1000000):
    o, r, d, _ = env.step(acts)
    # o, r, d, _ = env.step([0,0,0] * 1000)
    # print(o)

