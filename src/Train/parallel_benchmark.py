import os,inspect,argparse, datetime, copy, itertools
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from b3px_gym.b3px_env.parallel import laikago_gym_env_pl
from b3px_gym.b3px_env.examples.laikago_b3px_simple_env import DefaultCfg

import numpy as np
import time


# use_physx_options       = [True]
# use_physx_options       = [False]
# use_physx_options       = [True, False]
use_physx_options       = [True]
amount_robots_options   = [50]
# amount_robots_options   = [1]
# use_gpu_options         = [True]
use_gpu_options         = [True]
num_cores_options       = [10]
def make_env(cfg):
    env = laikago_gym_env_pl.LaikagoB3PxEnvPl_1(cfg)
    return env

def stand(args, sample_amount=250):
    # SimEnv Config
    cfg = copy.deepcopy(DefaultCfg)
    cfg['backend'] = 'physx' if args.backend else 'bullet'
    cfg['gui']          = True
    cfg['solver']       = 'tgs'
    cfg['urdf_root']    = '../urdf'
    cfg['distance']     = 3
    cfg['batch']        = args.parallel
    cfg['gpu']          = args.useGPU
    cfg['core']    = args.num_cores

    # Environment
    env = make_env(cfg)
    time4step = []
    actions = [[-0.035, -0.05, -0.5] * 4]*args.parallel
    samples = 0
    for _ in range(max(sample_amount//args.parallel, 10)):
        # print("%d/%d" % (_, max(sample_amount//args.parallel, 10)))
        start_step_time = time.time()
        obs, rewards, dones, _ =  env.step(actions)
        samples += obs.shape[0]
        time4step.append(copy.copy(time.time()-start_step_time))
        # print("obs:", obs.shape)
        # print("rewards:", rewards)
        # print("dones:", dones)

        if np.any(dones == True):
            env.reset(env.uids[dones == True])
    print("Gotten %d samples in %.2fms" % (samples, sum(time4step)*1e3))
    time_per_sample = np.mean(np.array(time4step))/obs.shape[0]
    print("Time per sample: %.2fms" % (time_per_sample*1e3))
    return np.mean(np.array(time4step)), time_per_sample

if __name__ == '__main__':
    amount_of_samples = 1000
    counter = 0
    time_benchmark = {}
    for use_physx in use_physx_options:
        _use_gpu_options = use_gpu_options
        _num_cores_options = num_cores_options
        if not use_physx:
            _use_gpu_options = [False]
            _num_cores_options = [1]
        for amount_robots in amount_robots_options:
            for use_gpu in _use_gpu_options:
                for num_cores in _num_cores_options:
                    counter+=1
                    print("Running use_physx: %d, amount of robots: %d, using GPU: %d, num_core: %d" 
                        % (use_physx, amount_robots, use_gpu, num_cores))
                    start_time = time.time()
                    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
                    parser.add_argument('--backend', type=bool, default=use_physx,
                                        help='BackEnd type(default: False, pybullet, True, PhysX)')
                    parser.add_argument('--useGPU', type=bool, default=use_gpu)
                    parser.add_argument('--parallel', type=int, default = amount_robots, metavar = 'N',
                                        help = 'Parallel Robots in one environment (default: 5)')
                    parser.add_argument('--num_cores', type=int, default = num_cores)
                    
                    
                    args = parser.parse_args()
                    
                    time4step, time_per_sample = stand(args)
                    end_time = time.time()-copy.copy(start_time)

                    time_benchmark.update({"use_physx:%d,amount_robots:%d,use_gpu:%d,num_cores:%d"%(use_physx, amount_robots,use_gpu,num_cores): [end_time, time4step, time_per_sample]})
                    print("RTF: %d" % (1./25./time_per_sample))
                    print("=========================================================================================")
    print("Time benchmark")
    for key in time_benchmark.keys():
        # print("%s: %.2f, time for step: %.2fms, time per sample: %.2fms" % (key, time_benchmark[key][0], time_benchmark[key][1]*1e3, time_benchmark[key][2]*1e3))
        # print("RTF: %.2f" % ((1./25)/time_benchmark[key][2]))        
        print("%s: %d" % (key, ((1./25)/time_benchmark[key][2])))