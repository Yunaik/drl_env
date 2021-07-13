import os,inspect,argparse, datetime, copy, itertools, time
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(currentdir))

# from b3px_gym.b3px_env.singleton import laikago, laikago_gym_env
from b3px_gym.b3px_env.parallel import anymal_pl

from distWrapper import grpcServer, grpcClient
import numpy as np
import copy, argparse

def make_env(cfg):
    print("Making env with Cfg: ", cfg)
    env = anymal_pl.AnymalB3PxEnvPl_1(cfg)
    return env

def global_seed_reset(seed):
    np.random.seed(seed)

def main(args, bind_addr = '127.0.0.1:18888'):
    global_seed_reset(666666)
    grpcServer.Serve(bind_addr, make_env)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--addr', type=str, default='127.0.0.1:5000',
                        help='Default Server Listen Path(default:127.0.0.1:5000)')

    args = parser.parse_args()

    main(args, args.addr)

