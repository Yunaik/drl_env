from __future__ import print_function

import grpc, json, pickle as pkl, numpy as np
import copy
from distWrapper import dist_pb2
from distWrapper import dist_pb2_grpc

from gym.spaces import Box
# from b3px_gym.b3px_env.singleton import laikago
# from b3px_gym.b3px_env.singleton import anymal

class GrpcClient:
    def __init__(self, addr, env=None):
        self._addr = addr
        self.ch = grpc.insecure_channel(self._addr)
        self.stub = dist_pb2_grpc.WALLEStub(self.ch)
        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
        else:
            print("NO ENV GIVEN. MAKING ANYMAL SPACE")
            self.action_space = Box(low=np.array([-1]*12), high=np.array([1]*12))
            observation_high = np.array([np.finfo(np.float32).max] * 21)
            self.observation_space = Box(low=-observation_high, high=observation_high)

    def connect(self, cfg):

        self.batch = cfg['batch']
        self._env_step_counter = np.array([0] * self.batch)
        self.action_space.shape = (self.batch, len(self.action_space.low))
        # print("Self batch: %d" % self.batch)
        cfg_str = json.dumps(cfg)
        req_msg = dist_pb2.EnvCfg(cfg = cfg_str)
        rsp = self.stub.Connect(req_msg)
        return True

    def reset(self, uids = []):
        req = dist_pb2.Uids(ids = uids)
        # print("Req: ", req)
        rsp = self.stub.Reset(req)
        if(len(uids) == 0):
            self._env_step_counter[:] = 0
        else:
            self._env_step_counter[uids] = 0
        return pkl.loads(rsp.obs)

    def get_observations(self, uids = []):
        req = dist_pb2.Uids(ids = uids)
        rsp = self.stub.GetObs(req)
        return pkl.loads(rsp.obs)

    def step(self, acts = None, uids = []):
        req = dist_pb2.Actions()
        # print("Req: ", req)
        if len(uids) != 0:
            req.uids.ids.extend(uids)
        req.actions = acts.dumps()

        rsp = self.stub.Step(req)
        if(len(uids) == 0):
            self._env_step_counter +=1
        else:
            self._env_step_counter[uids] += 1

        n_obs = pkl.loads(rsp.obs)
        r = pkl.loads(rsp.r)
        d = pkl.loads(rsp.d)

        return n_obs, r, d, []

if __name__ == '__main__':
    cfg = copy.deepcopy(DefaultCfg)
    cfg['backend'] = 'physx'
    cfg['gui']          = False
    cfg['solver']       = 'tgs'
    cfg['urdf_root']    = '../urdf'
    cfg['distance']     = 3
    cfg['batch']        = 5
    cfg['gpu']          = 1
    cfg['core']         = 1
    # DefaultCfg = {
    #     "backend": "physx",
    #     "gpu": True,
    #     "gui": False,
    #     "core": 1,
    #     "solver": "pgs",
    #     "enlarge": 1,
    #     "sim_ts": 1000,
    #     "ctl_ts": 500,
    #     "cmd_ts": 25,
    #     "is_render": False,
    #     "video_name": "laikago_{}.mp4",
    #     "motor_kp": 400,
    #     "motor_kd": 10,
    #     "cam_dist": 2.0,
    #     "cam_yaw": 52,
    #     "cam_pitch": -30,
    #     "records_sim": False,
    #     "fixed": False,
    #     # "urdf_root"         : "/home/syslot/DevSpace/WALLE/src/pybullet_demo/urdf"
    #     "urdf_root"		: "../../../urdf",

    #     # After args are used for Parallel Environment
    #     "batch"             : 5,
    #     "distance"          : 5,
    #     "init_noise"        : False
    # }

    env = GrpcClient('127.0.0.1:6000')
    print("Client established")
    rst = env.connect(cfg)
    print("env connect response : ", rst)


    uids = [0,1,2,3,4]
    print("Going to reset")
    rst = env.reset(uids)
    print("env reset response : ", rst)

    rst = env.get_observations(uids)
    print("hi")
    print("env get_obs response : ", rst)


    acts = np.random.rand(len(uids), 12)
    print("OK")
    rst = env.step(uids, acts)
    print("env step response :", rst)
    print("next obs : ", rst[0])
    print("rewards : ", rst[1])
    print("terminal : ", rst[2])



