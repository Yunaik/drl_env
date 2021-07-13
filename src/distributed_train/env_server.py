import os,inspect,argparse, datetime, copy, itertools, time
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(currentdir))

# from b3px_gym.b3px_env.singleton import laikago, laikago_gym_env
from b3px_gym.b3px_env.parallel import parallel_env_mujoco

from pybullet_envs.gym_pendulum_envs import InvertedPendulumBulletEnv, InvertedPendulumSwingupBulletEnv, InvertedDoublePendulumBulletEnv
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv, AntBulletEnv, HalfCheetahBulletEnv, Walker2DBulletEnv, HopperBulletEnv
from b3px_gym.b3px_env.singleton.valkyrie_gym_env import Valkyrie
# from rllab.envs.pybullet.valkyrie_multi_env.anymal_parallel import Anymal

from distWrapper import grpcServer, grpcClient

import numpy as np
import copy, argparse
import json


def make_env(cfg):
    with open('./robot.json') as f:
        robot_config = json.load(f)
    time_to_stabilize=0.0
    if robot_config["robot"] == 0:
        robot = HumanoidBulletEnv
        isMujocoEnv=True
        class_args = {"isPhysx": True if cfg['backend'] == 'physx' else False, "time_step": 0.01}
    elif robot_config["robot"] == 1:
        robot = AntBulletEnv
        isMujocoEnv=True
        class_args = {"isPhysx": True if cfg['backend'] == 'physx' else False, "time_step": 0.01}
    elif robot_config["robot"] == 2:
        robot = HalfCheetahBulletEnv
        isMujocoEnv=True
        class_args = {"isPhysx": True if cfg['backend'] == 'physx' else False, "time_step": 0.01}
    elif robot_config["robot"] == 3:
        robot = Walker2DBulletEnv
        isMujocoEnv=True
        class_args = {"isPhysx": True if cfg['backend'] == 'physx' else False, "time_step": 0.01}
    elif robot_config["robot"] == 4:
        robot = HopperBulletEnv
        isMujocoEnv=True
        class_args = {"isPhysx": True if cfg['backend'] == 'physx' else False, "time_step": 0.01}
    elif robot_config["robot"] == 5:
        robot = Valkyrie
        isMujocoEnv=False
        class_args = {"time_step": 0.005, "frame_skip": 8, "margin_in_degree": cfg["margin"], "useFullDOF": cfg["useFullDOF"], "regularise_action": cfg["regularise_action"]}
        time_to_stabilize=3.0
    elif robot_config["robot"] == 6:
        robot = Anymal
        isMujocoEnv=False
        class_args = {"time_step": 0.01, "frame_skip": 4}
        time_to_stabilize=1.0
    try:
        force_duration=cfg["force_duration"]
    except:
        force_duration = 0.0
    
    # class_args = {"isPhysx": True if cfg['backend'] == 'physx' else False, "fixed_base": True, "time_to_stabilize":-1.0}
    env = parallel_env_mujoco.ParallelEnv(cfg, robotClass=robot, class_args=class_args, time_to_stabilize=time_to_stabilize, 
                isMujocoEnv=isMujocoEnv, spawn_height=robot_config["spawn_height"][robot_config["robot"]], force_duration=force_duration)

    # print("class args: ", class_args)
    # env = parallel_env_mujoco.ParallelEnv(cfg, robotClass=HumanoidBulletEnv, class_args=class_args, isMujocoEnv=True, spawn_height=2)
    #env = parallel_env_mujoco.ParallelEnv(cfg, robotClass=AntBulletEnv, class_args=class_args, isMujocoEnv=True)
    # env = parallel_env_mujoco.ParallelEnv(cfg, robotClass=HalfCheetahBulletEnv, class_args=class_args, isMujocoEnv=True)
    #env = parallel_env_mujoco.ParallelEnv(cfg, robotClass=Walker2DBulletEnv, class_args=class_args, isMujocoEnv=True)
    # env = parallel_env_mujoco.ParallelEnv(cfg, robotClass=HopperBulletEnv, class_args=class_args, isMujocoEnv=True)
    return env

def global_seed_reset(seed):
    np.random.seed(seed)

def main(args, bind_addr = '127.0.0.1:18888'):
    global_seed_reset(666666)
    print("Established server with address %s" % bind_addr)
    grpcServer.Serve(bind_addr, make_env)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--addr', type=str, default='127.0.0.1:5000',
                        help='Default Server Listen Path(default:127.0.0.1:5000)')

    args = parser.parse_args()

    main(args, args.addr)

