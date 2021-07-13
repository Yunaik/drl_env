from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger
from b3px_gym.b3px_env.examples.base_cfg import DefaultCfg
from b3px_gym.b3px_env.parallel import parallel_env_mujoco

from pybullet_envs.gym_pendulum_envs import InvertedPendulumBulletEnv, InvertedPendulumSwingupBulletEnv, InvertedDoublePendulumBulletEnv
from pybullet_envs.gym_locomotion_envs import HumanoidBulletEnv, AntBulletEnv, AntBulletEnvMJC, AntBulletEnvMJC_physx, HalfCheetahBulletEnv, Walker2DBulletEnv, HopperBulletEnv
filename = str(uuid.uuid4())
import copy
#torch.cuda.set_device(0)


def make_env():
    cfg = copy.deepcopy(DefaultCfg)

    cfg['gui']          = True
    cfg['backend']      = 'physx' 
    cfg['solver']       = 'tgs'
    cfg['urdf_root']    = '../urdf'
    cfg['distance']     = 3
    cfg['batch']        = 1
    cfg['gpu']          = True
    cfg['core']    = 10
    class_args = {"render":False,"isPhysx":True, "time_step": 0.01}
    env = parallel_env_mujoco.ParallelEnv(cfg, robotClass=HalfCheetahBulletEnv, class_args=class_args, isMujocoEnv=True)
    return env


def simulate_policy(args):
    # class_args = {"render":render,"isPhysx":use_physx_options[0], "time_step": 0.01}

    env = make_env()
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    # env = data['evaluation/env']
    print("Policy loaded")
    #if args.gpu:
    set_gpu_mode(True)
    policy.cuda()
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=False,
            preprocess_obs_for_policy_fn=lambda x: x[0]
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=50*10,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)

