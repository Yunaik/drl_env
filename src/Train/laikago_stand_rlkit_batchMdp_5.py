import os,inspect,argparse, datetime, copy, itertools
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
from rlkit.samplers.data_collector import MdpPathCollector, BatchMdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import numpy as np
import torch

import imageio

from tensorboardX.writer import SummaryWriter

def make_env(cfg):
    env = laikago_gym_env_pl.LaikagoB3PxEnvPl_1(cfg)
    return env

def global_seed_reset(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)

def show(args):
	cfg = copy.deepcopy(DefaultCfg)
	cfg['backend'] = 'physx' if args.backend else 'bullet'
	cfg['gui'] = True
	cfg['solver'] = 'tgs'
	cfg['urdf_root'] = '../urdf'
	cfg['is_render'] = False


	# Environment
	env = make_env(cfg)

	obs_dim = env.observation_space.low.size
	action_dim = env.action_space.low.size

	M = 256

	policy = TanhGaussianPolicy(
		obs_dim=obs_dim,
		action_dim=action_dim,
		hidden_sizes=[M, M, M],
	)
	# checkpoint = torch.load('/home/syslot/Desktop/rlkit.kpt')
	policy.load_state_dict(torch.load('./policy.chkp'))



	done = False
	state = env.reset()
	imgs = []
	while not done:
		action,_ = policy.get_action(state, deterministic= True)
		state, reward, done, _ = env.step(action)  # Step
		# img = env.render()
		# imgs.append(img)
		if done or env._env_step_counter == 125:
			break

	if cfg['is_render']:
		imageio.mimwrite(cfg['video_name'].format(datetime.datetime.now()), np.asarray(imgs), fps = 50.0)

def experiment(variant, cfg):

	global_seed_reset(variant['seed'])

	expl_env = make_env(cfg)
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
		expl_env,
        eval_policy,
    )
	expl_path_collector = BatchMdpPathCollector(
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

	algorithm = TorchBatchRLAlgorithm(
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
	cfg['core'] = 2
	cfg['batch'] = 5

	variant = dict(
		seed=666666, #int(time.time()),
		algorithm="SAC",
		version="normal",
		layer_size=256,
		replay_buffer_size=int(1E6),
		algorithm_kwargs=dict(
			num_epochs=3000,
			num_eval_steps_per_epoch=5000,
			num_trains_per_train_loop=1000,
			num_expl_steps_per_train_loop=1000,
			min_num_steps_before_training=10000,
			max_path_length=125,
			batch_size=256,
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
	parser.add_argument('--policy', default="Gaussian",
	                    help='algorithm to use: Gaussian | Deterministic')
	parser.add_argument('--eval', type=bool, default=True,
	                    help='Evaluates a policy a policy every 10 episode (default:True)')
	parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
	                    help='discount factor for reward (default: 0.99)')
	parser.add_argument('--tau', type=float, default=0.005, metavar='G',
	                    help='target smoothing coefficient(τ) (default: 0.005)')
	parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
	                    help='learning rate (default: 0.0003)')
	parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
	                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
	parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
	                    help='Temperature parameter α automaically adjusted.')
	parser.add_argument('--seed', type=int, default=456, metavar='N',
	                    help='random seed (default: 456)')
	parser.add_argument('--batch_size', type=int, default=256, metavar='N',
	                    help='batch size (default: 256)')
	parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
	                    help='maximum number of steps (default: 1000000)')
	parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
	                    help='hidden size (default: 256)')
	parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
	                    help='model updates per simulator step (default: 1)')
	parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
	                    help='Steps sampling random actions (default: 10000)')
	parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
	                    help='Value target update per no. of updates per step (default: 1)')
	parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
	                    help='size of replay buffer (default: 10000000)')
	parser.add_argument('--cuda', action="store_true",
	                    help='run on CUDA (default: False)')
	parser.add_argument('--backend', type=bool, default=False,
	                       help='BackEnd type(default: False, pybullet, True, PhysX)')
	parser.add_argument('-g', '--gui', type=bool, default=False,
	                    help='use gui or not ( default is False, not load)')
	args = parser.parse_args()

	main(args)
	# show(args)