import os,inspect,argparse, datetime, copy, itertools, time
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(currentdir))

# from b3px_gym.b3px_env.singleton import laikago, laikago_gym_env
# from b3px_gym.b3px_env.parallel import laikago_gym_env_pl
from b3px_gym.b3px_env.examples.laikago_b3px_simple_env import DefaultCfg


# import rlkit.torch.pytorch_util as ptu
# from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
# from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpStepCollector, BatchMdpPathCollector, BatchMdpStepCollector, Random_BatchMdpStepCollector
# from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
# from rlkit.torch.sac.sac import SACTrainer
# from rlkit.torch.networks import FlattenMlp
# from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchOnlineRLAlgorithm
# from optimizer.lookahead import Lookahead

from distWrapper import grpcServer, grpcClient

import numpy as np
import torch

# import imageio
# from multiprocessing import Process


def global_seed_reset(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def experiment(cfg):

    expl_env = grpcClient.GrpcClient(cfg['addr_expl'])
    # eval_env = grpcClient.GrpcClient(cfg['addr_eval'])

    expl_env.connect(cfg)
    # eval_env.connect(cfg)

    action = np.zeros((cfg['batch'], 12))
    expl_path_collector = Random_BatchMdpStepCollector(
        expl_env,
        action,
        max_path_length=cfg["epoch_length"]
    )
    start_time = time.time()
    print("Going to collect")
    expl_path_collector.collect_new_steps(
                                        max_path_length=cfg["episode_length"],
                                        num_steps=cfg["epoch_length"])

    paths = expl_path_collector.get_epoch_paths()
    sum_samples = 0
    for path in paths:
        sum_samples += path["observations"].shape[0]
    # print(sum_samples)

    print("Time for %d samples: %.3f, RTF: %.1f" % (sum_samples, time.time()-start_time, (sum_samples/cfg["cmd_ts"])/(time.time()-start_time)))

    # print("Paths: ", paths.keys())

def main(args):
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
    cfg['enlarge'] = args.enlarge
    cfg['epoch_length'] = args.epoch_length
    cfg['episode_length'] = args.episode_length

    cfg['addr_expl'] = '127.0.0.1:6000'
    # cfg['addr_eval'] = '127.0.0.1:6000'

    # variant = dict(
    #     seed=args.seed,  # int(time.time()),
    #     algorithm="SAC",
    #     version="normal",
    #     layer_size=256,
    #     replay_buffer_size=int(1E6),
    #     algorithm_kwargs=dict(
    #         num_epochs=3000,
    #         num_eval_steps_per_epoch=3000,
    #         num_trains_per_train_loop=1000,
    #         num_expl_steps_per_train_loop=1000,
    #         min_num_steps_before_training=10000,
    #         max_path_length=125,
    #         batch_size=args.tbatch,
    #     ),
    #     trainer_kwargs=dict(
    #         discount=0.99,
    #         soft_target_tau=5e-3,
    #         target_update_period=1,
    #         policy_lr=1E-3,
    #         qf_lr=1E-3,
    #         reward_scale=1,
    #         use_automatic_entropy_tuning=True,
    #     ),
    #     cfg = cfg,
    # )
    # setup_logger( args.name, variant = variant) #'laikago_pl_large_exp', variant=variant)
    # # setup_logger('laikago_pl_large_exp', variant=variant)
    # ptu.set_gpu_mode(True)
    experiment(cfg)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--backend', type=bool, default=True,
                           help='BackEnd type(default: False, pybullet, True, PhysX)')
    parser.add_argument('-g', '--gui', type=bool, default=False,
                        help='use gui or not ( default is False, not load)')
    parser.add_argument('--name', type=str, default = 'laikago_report')
    parser.add_argument('--seed', type=int, default = 0 )
    parser.add_argument('--tbatch', type=int, default=64)
    parser.add_argument('--episode_length', type=int, default=250)
    parser.add_argument('--epoch_length', type=int, default=10000)

    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--core', type=int, default=10)
    parser.add_argument('--enlarge', type=int, default=20)
    args = parser.parse_args()

    main(args)


