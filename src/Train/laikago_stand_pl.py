import os,inspect,argparse, datetime, copy, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)


from src.gym.b3px_env.singleton import laikago
from src.gym.b3px_env.parallel import laikago_gym_env_pl
from src.gym.b3px_env.examples.laikago_b3px_simple_env import DefaultCfg
from src.SAC.model import *
from src.SAC.replay_memory import *
from src.SAC.sac import SAC
import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm import trange

import numpy as np


from tensorboardX.writer import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter



def make_env(cfg):
    env = laikago_gym_env_pl.LaikagoB3PxEnvPl_1(cfg)
    return env


def show(args):
	cfg = copy.deepcopy(DefaultCfg)
	cfg['backend'] = 'physx' if args.backend else 'bullet'
	cfg['gui'] = True
	cfg['solver'] = 'tgs'
	cfg['urdf_root'] = '../urdf'


	# Environment
	env = make_env(cfg)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	env.seed(args.seed)

	agent = SAC(env.observation_space.shape[0], env.action_space, args)
	agent.load_model(actor_path='./models/sac_actor_laikago_', critic_path='./models/sac_critic_laikago_')

	while True:
		done = False
		state = env.reset()
		while not done:
			action = agent.select_action(state)
			state, reward, done, _ = env.step(action)  # Step

def main(args):
	# SimEnv Config
	cfg = copy.deepcopy(DefaultCfg)
	cfg['backend'] = 'physx' if args.backend else 'bullet'
	cfg['gui'] = args.gui
	cfg['solver'] = 'tgs'
	cfg['urdf_root'] = '../urdf'
	cfg['core'] = 8
	cfg['batch'] = 20


	# Environment
	env = make_env(cfg)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	env.seed(args.seed)

	# Agent
	agent = SAC(env.observation_space.shape[1], env.action_space, args)

	# TesnorboardX
	writer = SummaryWriter(
		logdir='runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'laikago',
		                                     args.policy, "autotune" if args.automatic_entropy_tuning else ""))

	# writer = SummaryWriter()
	# Memory
	memory = ReplayMemory(args.replay_size)

	# Training Loop
	total_numsteps = 0
	updates = 0

	episode_reward = np.array([0.] * cfg['batch'])
	episode_steps = np.array([0.] * cfg['batch'])
	done = np.array([False] * cfg['batch'])
	state = env.reset()

	i_episode = 0
	test_thres = 1000

	while True:
		if args.start_steps > total_numsteps:
			action = env.action_space.sample()  # Sample random action
		else:
			action = agent.select_action(state)  # Sample action from policy

		if len(memory) > args.batch_size:
			# Number of updates per step in environment
			for i in range(args.updates_per_step):
				# Update parameters of all the networks
				critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
				                                                                                     args.batch_size,
				                                                                                     updates)

				writer.add_scalar('loss/critic_1', critic_1_loss, updates)
				writer.add_scalar('loss/critic_2', critic_2_loss, updates)
				writer.add_scalar('loss/policy', policy_loss, updates)
				writer.add_scalar('loss/entropy_loss', ent_loss, updates)
				writer.add_scalar('entropy_temprature/alpha', alpha, updates)
				updates += 1

		next_state, reward, done, _ = env.step(action)  # Step
		episode_steps += 1
		total_numsteps += 1
		episode_reward += reward

		# Ignore the "done" signal if it comes from hitting the time horizon.
		# (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
		mask = np.array([0] * cfg['batch'])
		mask[done] = 1

		memory.push_batch(state, action, reward, next_state, mask)  # Append transition to memory

		state = next_state

		if np.any(done):
			idx = np.where(done==True)[0]
			for i in range(len(idx)):
				writer.add_scalar('reward/train', episode_reward[idx[i]], i_episode+i)
			i_episode += len(idx)

			# Reset Env Which has done
			episode_reward[idx] = 0
			episode_steps[idx] = 0
			env.reset(idx)

			print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
	                                                                              episode_steps,
	                                                                              np.round(episode_reward, 2)))
			if i_episode > test_thres and args.eval == True:
				# Reset All Environment And Do Policy Test
				agent.save_model('laikago')
				avg_reward = 0.
				episodes = 10
				idx_episode = 0
				state = env.reset()
				episode_reward = np.array([0.] * cfg['batch'])

				while True:
					action = agent.select_action(state, eval=True)
					next_state, reward, done, _ = env.step(action)

					episode_reward += reward
					state = next_state

					if np.any(done):
						idxs = np.where(done)[0]
						idx_episode += len(idxs)
						env.reset(idxs)

						if idx_episode >= episodes:
							break

				avg_reward = episode_reward.sum() / episodes
				writer.add_scalar('avg_reward/test', avg_reward, i_episode)

				test_thres += 1000

		if total_numsteps > args.num_steps:
			break
	env.close()


def DefaultStand():
	# SimEnv Config
	cfg = copy.deepcopy(DefaultCfg)
	cfg['backend'] = 'physx' if args.backend else 'bullet'
	cfg['gui'] = True
	cfg['solver'] = 'tgs'
	cfg['urdf_root'] = '../urdf'
	cfg['distance'] = 3


	# Environment
	env = make_env(cfg)
	actions = [[-0.035, -0.05, -0.5] * 4]*5
	for i in range(25*10):
		obs, rewards, dones, _ =  env.step(actions)
		print("obs:", obs)
		print("rewards:", rewards)
		print("dones:", dones)

		if np.any(dones == True):
			env.reset(env.uids[dones == True])




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
	parser.add_argument('--parallel', type=int, default = 5, metavar = 'N',
	                    help = 'Parallel Robots in one environment (default: 5)')
	args = parser.parse_args()

	# DefaultStand()

	main(args)