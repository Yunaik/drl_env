# !/usr/bin/env python3
"""This is an example to train a task with PPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import torch.nn as nn
import sys
sys.path.insert(0, '../')

import spinup.algos.pytorch.ppo.core as core
from spinup.algos.pytorch.ppo.ppo import ppo

import torch
from rlkit.samplers.data_collector import BatchMdpPathCollector, BatchMdpStepCollector, BatchMdpStepCollector_parallel
import os,inspect,argparse, datetime, copy, itertools, time
import collections
import copy
from pybullet_envs.gym_pendulum_envs import InvertedPendulumBulletEnv, InvertedPendulumSwingupBulletEnv, InvertedDoublePendulumBulletEnv
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv, AntBulletEnv, HalfCheetahBulletEnv, Walker2DBulletEnv, HopperBulletEnv
from distWrapper import grpcServer, grpcClient
from b3px_gym.b3px_env.examples.base_cfg import DefaultCfg
from b3px_gym.b3px_env.singleton.valkyrie_gym_env import Valkyrie
# from rllab.envs.pybullet.valkyrie_multi_env.anymal_parallel import Anymal

from shutil import copyfile, copytree

import rlkit.torch.pytorch_util as ptu
import json

with open('../distributed_train/robot.json') as f:
  robot_config = json.load(f)
#0: HumanoidBulletEnv, 1: AntBulletEnv, 2: HalfCheetahBulletEnv, 3: Walker2DBulletEnv, 4: HopperBulletEnv


def train(seed=1, args=None):
    """Train PPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """

    debugging = 0
    time_to_stabilize = 0.
    sample_time = 5.
    squash_output=False
    if robot_config["robot"] == 0:
        print("Dummy env is humanoid")
        dummy_env = HumanoidBulletEnv(client=None) # used to get obs_dim, act_dim
    elif robot_config["robot"] == 1:
        print("Dummy env is ant")
        dummy_env = AntBulletEnv(client=None) # used to get obs_dim, act_dim
    elif robot_config["robot"] == 2:
        print("Dummy env is half_cheetah")
        dummy_env = HalfCheetahBulletEnv(client=None) # used to get obs_dim, act_dim
    elif robot_config["robot"] == 3:
        print("Dummy env is walker")
        dummy_env = Walker2DBulletEnv(client=None) # used to get obs_dim, act_dim
    elif robot_config["robot"] == 4:
        print("Dummy env is hopper")
        dummy_env = HopperBulletEnv(client=None) # used to get obs_dim, act_dim
    elif robot_config["robot"] == 5:
        dummy_env = Valkyrie(client=None, useFullDOF=args.useFullDOF, margin_in_degree=args.margin) # used to get obs_dim, act_dim
        time_to_stabilize=5.0
        sample_time = 15
        squash_output=True
    elif robot_config["robot"] == 6:
        dummy_env = Anymal(client=None)
        time_to_stabilize = 1.0
        sample_time = 10
        squash_output=True
    print("Dummy env created")
    max_path_length = int((sample_time+time_to_stabilize)/(dummy_env.timestep*dummy_env.frame_skip)) if not debugging else 10
    ppo_batch_size = max(max_path_length*args.n_workers*args.batch_size, 1000) # every worker gets batch_size*max_path_length in ideal rollout, at least 1k samples
    ppo_batch_size = min(10000, ppo_batch_size)

    if ppo_batch_size/(args.batch_size*args.n_workers) < max_path_length:
        print("the ppo batch size (%d)/parallel_robots  is smaller than minimum rollouts (%d) till episode ends (where advantage will be calculated). Batch size thus increased to %d" 
        % (ppo_batch_size, max_path_length, max_path_length*args.batch_size*args.n_workers))
        ppo_batch_size = max_path_length*args.batch_size*args.n_workers

    print("Max path length: %d, ppo_batch_size: %d" % (max_path_length, ppo_batch_size))
    # SimEnv Config
    cfg = copy.deepcopy(DefaultCfg)
    cfg['backend'] = 'physx' if args.backend else 'bullet'
    cfg['gui'] = args.gui
    cfg['solver'] = 'tgs'
    cfg['urdf_root'] = '../urdf'
    cfg['cam_dist'] = 1000
    cfg['core'] = args.core
    cfg['batch'] = args.batch_size
    cfg['gpu'] = True
    cfg['enlarge'] = 3
    cfg['addr_eval'] = '127.0.0.1:6000'
    cfg["margin"] = args.margin
    cfg["force_duration"] = args.force_duration
    cfg["useFullDOF"] = args.useFullDOF
    cfg["regularise_action"] = args.regularise_action

    eval_env = grpcClient.GrpcClient(cfg['addr_eval'], dummy_env)
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg['batch'] = 1

    eval_env.connect(eval_cfg)
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    print("Obs dim: ", obs_dim)
    print("Action dim: ", action_dim)
    # Generating n expl_path_collectors
    expl_path_collector_batch = []
    for worker_idx in range(args.n_workers): # if using multiple gpu's edit port number here
        addr_expl = "127.0.0.1:500%d" % worker_idx
        expl_env = grpcClient.GrpcClient(addr_expl, dummy_env)
        expl_env.connect(cfg)
        expl_path_collector = BatchMdpStepCollector(
            expl_env,
            policy=None,
            max_path_length=max_path_length
        )
        expl_path_collector_batch.append(expl_path_collector)

    # expl_path_collector_class = BatchMdpStepCollector_parallel
    # expl_path_collector_args    =  {"batch_collector_list": expl_path_collector_batch, "n_workers": args.n_workers}
    expl_path_collector = BatchMdpStepCollector_parallel(
                                expl_path_collector_batch,
                                n_workers=args.n_workers,
                                max_path_length=max_path_length,
                                num_steps=ppo_batch_size
                                )

    logger_kwargs={"output_dir": "../data/ppo_spinup/"    }
    copyfile('../distributed_train/robot.json', logger_kwargs["output_dir"]+"/robot.json")
    # print("OUTPUT: %s" %logger_kwargs["output_dir"]+"/robot.json")
    ppo(eval_env, sampler=expl_path_collector, 
            actor_critic=core.MLPActorCritic, ac_kwargs={"hidden_sizes":(256,256), "activation":nn.ReLU, "squash_output":squash_output}, seed=0, 
            steps_per_epoch=ppo_batch_size+1, 
            epochs=10000, 
            gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
            vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=max_path_length,
            target_kl=0.01,  save_freq=10, logger_kwargs=logger_kwargs)

ptu.set_gpu_mode(True)

parser = argparse.ArgumentParser(description='PyTorch PPO')
parser.add_argument('--backend', type=bool, default=False,
                        help='BackEnd type(default: False, pybullet, True, PhysX)')
parser.add_argument('-g', '--gui', type=bool, default=False,
                    help='use gui or not ( default is False, not load)')
parser.add_argument('--name', type=str, default = 'half_cheetah')
parser.add_argument('--seed', type=int, default = 0)
parser.add_argument('--tbatch', type=int, default=64)
parser.add_argument('--core', type=int, default=10)
# parser.add_argument('--sample_time', type=float, default=5)

parser.add_argument('--batch_size', type=int, default=10)
# parser.add_argument('--ppo_buffer_size', type=int, default=4000)
parser.add_argument('--n_workers', type=int, default=1)

parser.add_argument('--margin', type=float, default=10)

parser.add_argument('--regularise_action', type=bool, default=True)
parser.add_argument('--useFullDOF', type=bool, default=False)
parser.add_argument('--force_duration', type=float, default=0.0)

args = parser.parse_args()


train(seed=1, args=args)


