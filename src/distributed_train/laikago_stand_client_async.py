import os,inspect,argparse, datetime, copy, itertools, time
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(currentdir))

from b3px_gym.b3px_env.singleton import laikago, laikago_gym_env
from b3px_gym.b3px_env.parallel import laikago_gym_env_pl
from b3px_gym.b3px_env.examples.laikago_b3px_simple_env import DefaultCfg


import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpStepCollector, BatchMdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchOnlineRLAlgorithm

from distWrapper import grpcServer, grpcClient

import numpy as np
import torch

import imageio
from multiprocessing import Process


def global_seed_reset(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)


def experiment(variant, cfg):

	expl_env = grpcClient.GrpcClient(cfg['addr_expl'])
	eval_env = grpcClient.GrpcClient(cfg['addr_eval'])

	expl_env.connect(cfg)
	eval_env.connect(cfg)

	obs_dim = expl_env.observation_space.low.size
	action_dim = expl_env.action_space.low.size

	M = variant['layer_size']
	qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
	)
	qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
	target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
	target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
	policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M],
    )
	eval_policy = MakeDeterministic(policy)
	eval_path_collector = BatchMdpPathCollector(
		eval_env,
        eval_policy,
    )
	expl_path_collector = MdpStepCollector(
        expl_env,
        policy,
    )
	replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
	trainer = SACTrainer(
        env=expl_env,
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
    # SimEnv Config
    cfg = copy.deepcopy(DefaultCfg)
    cfg['backend'] = 'physx' if args.backend else 'bullet'
    cfg['gui'] = args.gui
    cfg['solver'] = 'tgs'
    cfg['urdf_root'] = '../urdf'
    cfg['cam_dist'] = 1000
    cfg['core'] = 1
    cfg['batch'] = 1
    cfg['gpu'] = False
    cfg['enlarge'] = 5

    cfg['addr_expl'] = '127.0.0.1:5000'
    cfg['addr_eval'] = '127.0.0.1:6000'

    variant = dict(
        seed=666666,  # int(time.time()),
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=30,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=100,
            min_num_steps_before_training=1000,
            max_path_length=125,
            batch_size=256
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
    )
    setup_logger('laikago_pl_large_exp', variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, cfg)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
	parser.add_argument('--backend', type=bool, default=False,
	                       help='BackEnd type(default: False, pybullet, True, PhysX)')
	parser.add_argument('-g', '--gui', type=bool, default=False,
	                    help='use gui or not ( default is False, not load)')
	args = parser.parse_args()

	main(args)


