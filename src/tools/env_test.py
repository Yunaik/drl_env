import copy
from b3px_gym.b3px_env.parallel import laikago_gym_env_pl


# import rlkit.torch.pytorch_util as ptu
# from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
# from rlkit.launchers.launcher_util import setup_logger
# from rlkit.samplers.data_collector import MdpPathCollector, BatchMdpPathCollector
# from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
# from rlkit.torch.sac.sac import SACTrainer
# from rlkit.torch.networks import FlattenMlp
# from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
#
# import numpy as np
import torch

# import imageio



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
    "motor_vel_limit"   : 100000,
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

def make_env(cfg):
    env = laikago_gym_env_pl.LaikagoB3PxEnvPl_1(cfg)
    return env


# SimEnv Config
cfg = copy.deepcopy(DefaultCfg)
cfg['backend'] = 'physx'
cfg['gui'] = True
cfg['solver'] = 'tgs'
cfg['urdf_root'] = '../urdf'
cfg['cam_dist'] = 1000
cfg['core'] = 4
cfg['batch'] = 32
cfg['gpu'] = True


env = laikago_gym_env_pl.LaikagoB3PxEnvPl_1(cfg)

print(env.observation_space)