import os,inspect,argparse, datetime, copy, itertools, time
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(currentdir))
import time
# from b3px_gym.b3px_env.singleton import laikago, laikago_gym_env
# from b3px_gym.b3px_env.parallel import laikago_gym_env_pl
# from b3px_gym.b3px_env.examples.laikago_b3px_simple_env import DefaultCfg
from b3px_gym.b3px_env.examples.base_cfg import DefaultCfg

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import BatchMdpPathCollector, BatchMdpStepCollector, BatchMdpStepCollector_parallel
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
# from rlkit.torch.networks import FlattenMlp
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchOnlineRLAlgorithm
from optimizer.lookahead import Lookahead

from distWrapper import grpcServer, grpcClient

from pybullet_envs.gym_pendulum_envs import InvertedPendulumBulletEnv, InvertedPendulumSwingupBulletEnv, InvertedDoublePendulumBulletEnv
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv, AntBulletEnv, HalfCheetahBulletEnv, Walker2DBulletEnv, HopperBulletEnv
from rllab.envs.pybullet.valkyrie_multi_env.anymal_parallel import Anymal

import numpy as np
import torch

import imageio
from multiprocessing import Process
import json
from b3px_gym.b3px_env.singleton.valkyrie_gym_env import Valkyrie

#0: HumanoidBulletEnv, 1: AntBulletEnv, 2: HalfCheetahBulletEnv, 3: Walker2DBulletEnv, 4: HopperBulletEnv
# if robot_config["robot"] == 0:
#     robot = HumanoidBulletEnv
# elif robot_config["robot"] == 1:
#     robot = AntBulletEnv
# elif robot_config["robot"] == 2:
#     robot = HalfCheetahBulletEnv
# elif robot_config["robot"] == 3:
#     robot = Walker2DBulletEnv
# elif robot_config["robot"] == 4:
#     robot = HopperBulletEnv

def global_seed_reset(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def experiment(variant, cfg, env):

    eval_env = grpcClient.GrpcClient(cfg['addr_eval'], env)
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg['batch'] = 1

    eval_env.connect(eval_cfg)

    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    print("Obs dim: ", obs_dim)
    print("Act dim: ", action_dim)
    print("Act dim shape: ", eval_env.action_space.shape)
    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M]
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = BatchMdpPathCollector(
        eval_env,
        eval_policy,
    )

    # Generating n expl_path_collectors
    expl_path_collector_batch = []
    for worker_idx in range(variant["n_workers"]): # if using multiple gpu's edit port number here
        addr_expl = "127.0.0.1:500%d" % worker_idx
        expl_env = grpcClient.GrpcClient(addr_expl, env)
        expl_env.connect(cfg)
        expl_path_collector = BatchMdpStepCollector(
            expl_env,
            policy,
            max_path_length=variant['algorithm_kwargs']['max_path_length']
        )
        expl_path_collector_batch.append(expl_path_collector)
    print("All env loaded. Time to get all envs: %.2fs" % (time.time()-start_time))
    expl_path_collector = BatchMdpStepCollector_parallel(
                                expl_path_collector_batch,
                                n_workers=variant["n_workers"],
                            )

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        eval_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )

    algorithm = TorchOnlineRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=expl_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


def main(args):
    debugging = 0
    time_to_stabilize = 0.
    sample_time = 5
    with open('../distributed_train/robot.json') as f:
        robot_config = json.load(f)
    if robot_config["robot"] == 0:
        env = HumanoidBulletEnv(client=None) # used to get obs_dim, act_dim
    elif robot_config["robot"] == 1:
        env = AntBulletEnv(client=None) # used to get obs_dim, act_dim
    elif robot_config["robot"] == 2:
        env = HalfCheetahBulletEnv(client=None) # used to get obs_dim, act_dim
    elif robot_config["robot"] == 3:
        env = Walker2DBulletEnv(client=None) # used to get obs_dim, act_dim
    elif robot_config["robot"] == 4:
        env = HopperBulletEnv(client=None) # used to get obs_dim, act_dim
    elif robot_config["robot"] == 5:
        env = Valkyrie(client=None) # used to get obs_dim, act_dim
        time_to_stabilize = 5.0
        sample_time = 15
    elif robot_config["robot"] == 6:
        env = Anymal(client=None)
        time_to_stabilize = 1.0
        sample_time = 10
    max_path_length = int((sample_time+time_to_stabilize)/(env.timestep*env.frame_skip)) if not debugging else 10
    print("Masx path length: ", max_path_length)
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

    variant = dict(
        seed=args.seed,  # int(time.time()),
        algorithm="SAC",
        version="normal",
        layer_size=256,
        n_workers=args.n_workers,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=10000,
            num_eval_steps_per_epoch=max_path_length,
            num_expl_steps_per_train_loop=max_path_length*10,
            min_num_steps_before_training=max_path_length*100,
            max_path_length=max_path_length,
            batch_size=args.tbatch,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=1E-3,
            qf_lr=1E-3,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        cfg = cfg,
    )
    setup_logger( args.name, variant = variant) 
    experiment(variant, cfg, env)

if __name__ == '__main__':
    ptu.set_gpu_mode(True)

    start_time = time.time()
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--backend', type=bool, default=False,
                           help='BackEnd type(default: False, pybullet, True, PhysX)')
    parser.add_argument('-g', '--gui', type=bool, default=False,
                        help='use gui or not ( default is False, not load)')
    parser.add_argument('--name', type=str, default = 'sac')
    parser.add_argument('--seed', type=int, default = 0 )
    parser.add_argument('--tbatch', type=int, default=256)
    parser.add_argument('--core', type=int, default=10)
    # parser.add_argument('--sample_time', type=float, default=5)

    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--n_workers', type=int, default=10)
    
    parser.add_argument('--margin', type=float, default=10)

    args = parser.parse_args()

    main(args)


