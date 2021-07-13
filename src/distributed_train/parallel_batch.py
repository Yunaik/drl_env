import os,inspect,argparse, datetime, copy, itertools, time
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, os.path.dirname(currentdir))
import json
from b3px_gym.b3px_env.examples.laikago_b3px_simple_env import DefaultCfg
from rlkit.samplers.data_collector import Random_BatchMdpStepCollector
from distWrapper import grpcServer, grpcClient
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime

class robot_env():
    def __init__(self, agent_num=0, args=None, gpu=0):
        # SimEnv Config
        self.running_on_gpu_id = gpu
        self.agent_num = agent_num
        self.cfg = copy.deepcopy(DefaultCfg)
        self.cfg['gui'] = False
        self.cfg['solver'] = 'tgs'
        self.cfg['urdf_root'] = '../urdf'
        self.cfg['cam_dist'] = 1000
        self.cfg['gpu'] = True
        self.cfg['addr_expl'] = '127.0.0.1:%d00%d' % (5+gpu, self.agent_num)
        if args is not None:
            self.cfg['backend'] = 'physx' if args.backend else 'bullet'
            self.cfg['core'] = args.core
            self.cfg['batch'] = args.batch_size
            self.cfg['enlarge'] = args.enlarge
            self.cfg['epoch_length'] = args.epoch_length
            self.cfg['episode_length'] = args.episode_length
        self.gpu_number = gpu


        # laikago for env:

        self.env = grpcClient.GrpcClient(self.cfg['addr_expl'])
        # eval_env = grpcClient.GrpcClient(cfg['addr_eval'])

        self.env.connect(self.cfg)
        # eval_env.connect(cfg)

        action = np.zeros((self.cfg['batch'], 12))
        self.path_collector = Random_BatchMdpStepCollector(
            self.env,
            action,
            max_path_length=self.cfg["epoch_length"]
        )

    def rollout(self):

        print("GPU%d. Agent%d going to collect"%(self.gpu_number, self.agent_num))
        start_time = time.time()
        self.path_collector.collect_new_steps(
                                            max_path_length=self.cfg["episode_length"],
                                            num_steps=self.cfg["epoch_length"])
        print("GPU%d. Agent%d finished collecting"%(self.gpu_number, self.agent_num))

        paths = self.path_collector.get_epoch_paths()
        sum_samples = 0
        for path in paths:
            sum_samples += path["observations"].shape[0]
        time_for_collection = time.time()-start_time
        print("Agent%d, time for %d samples: %.3f, RTF: %.1f" % (self.agent_num, sum_samples, time_for_collection, (sum_samples/self.cfg["cmd_ts"])/(time_for_collection)))
        return sum_samples, time_for_collection

def collect_samples(env):
    return env.rollout()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--backend', type=int, default=1, help='BackEnd type(default: False, pybullet, True, PhysX)')
    parser.add_argument('-g', '--gui', type=bool, default=False, help='use gui or not ( default is False, not load)')
    parser.add_argument('--name', type=str, default = 'laikago_report')
    parser.add_argument('--seed', type=int, default = 0 )
    parser.add_argument('--tbatch', type=int, default=64)
    parser.add_argument('--episode_length', type=int, default=250)
    parser.add_argument('--epoch_length', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--core', type=int, default=24)
    parser.add_argument('--enlarge', type=int, default=10)
    parser.add_argument('--amount_of_env', type=int, default=1)
    parser.add_argument('--counter', type=int, default=0)
    parser.add_argument('--amount_of_gpu', type=int, default=1)
    args = parser.parse_args()
    print("Args: ", args)


    amount_of_env = args.amount_of_env
    try:
        os.mkdir("logs")
    except:
        pass
    file_name = "logs/result_%s.json"%(datetime.now().strftime("%H:%M:%S"))
    result_dict = {
        "backend": args.backend,
        "batch_size": args.batch_size,
        "core": args.core,
        "enlarge": args.enlarge,
        "amount_of_env": amount_of_env,
        "RTF": -1,
        "samples": -1,
        "time": -1,
        "amount_gpu": args.amount_of_gpu
    }

    with open(file_name, 'w') as fp:
        json.dump(result_dict, fp, indent=2)
    
    robot_envs = []
    for gpu in range(args.amount_of_gpu):
        for i in range(amount_of_env):
            robot_envs.append(robot_env(agent_num=i, args=args, gpu=gpu))

    start_time_pool = time.time()
    # Make the Pool of workers
    n_workers = amount_of_env*args.amount_of_gpu
    pool = ThreadPool(n_workers)
    results = pool.map(collect_samples, robot_envs)
    #close the pool and wait for the work to finish
    pool.close()
    print("Pool closed")
    pool.join()
    print("Pool join")
    time_to_pool = time.time()-start_time_pool
    total_samples = sum([samples[0] for samples in results])
    rtf = (total_samples/robot_envs[0].cfg["cmd_ts"])/time_to_pool

    print("Using %d workers to gather %d samples in %.1fs. RTF: %.1f" % (n_workers, total_samples, time_to_pool, rtf))
    
    result_dict = {
        "backend": args.backend,
        "batch_size": args.batch_size,
        "core": args.core,
        "enlarge": args.enlarge,
        "amount_of_env": amount_of_env,
        "RTF": rtf,
        "samples": total_samples,
        "time": time_to_pool,
        "amount_gpu": args.amount_of_gpu
    }

    with open(file_name, 'w') as fp:
        json.dump(result_dict, fp, indent=2)
