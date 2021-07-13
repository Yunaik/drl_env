import os,inspect,argparse, datetime, copy, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import matplotlib.pyplot as plt
from b3px_gym.b3px_env.parallel import parallel_env_mujoco
from b3px_gym.b3px_env.examples.base_cfg import DefaultCfg
from b3px_gym.b3px_env.singleton.cartpole import CartPoleBulletEnv
# from b3px_gym.b3px_env.singleton.anymal import Anymal
from pybullet_envs.gym_pendulum_envs import InvertedPendulumBulletEnv, InvertedPendulumSwingupBulletEnv, InvertedDoublePendulumBulletEnv
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv, AntBulletEnv, AntBulletEnvMJC, AntBulletEnvMJC_physx, HalfCheetahBulletEnv, Walker2DBulletEnv, HopperBulletEnv
from rllab.envs.pybullet.valkyrie_multi_env.anymal import Anymal
# from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
import time
import pickle
import pybullet
from pybullet_utils import bullet_client
import numpy as np
import matplotlib.pyplot as plt



cfg = copy.deepcopy(DefaultCfg)
cfg['backend'] = 'physx' if True else 'bullet'
cfg['gui']          = True
cfg['solver']       = 'tgs'
cfg['urdf_root']    = '../urdf'
cfg['distance']     = 3
cfg['batch']        = 1
cfg['gpu']          = True
cfg['core']         = 10

class_args = {"render":True,"isPhysx":True, "time_step": 0.01}
env = parallel_env_mujoco.ParallelEnv(cfg, robotClass=HopperBulletEnv, class_args=class_args, isMujocoEnv=True)

for _ in range(int(25*20000)):
    actions = [env.robots[0].action_space.sample()]*1
    obs, rewards, dones, _ =  env.step(actions)
