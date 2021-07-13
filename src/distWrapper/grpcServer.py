from __future__ import print_function

from concurrent import futures
import grpc,time, json
import numpy as np
import pickle as pkl

from distWrapper import dist_pb2
from distWrapper import dist_pb2_grpc


class DummyEnv:
    def __init__(self):
        self.observation_space = (21,1)
        self.action_space = (12,1)

    def Reset(self, uids = None):
        if uids == None:
            uids = [1,2,3,4,5]
        return self.GetObs(uids)


    def GetObs(self, uids=None):
        if uids == None:
            uids = [1, 2, 3, 4, 5]
        return np.random.rand(len(uids), 21)

    def Step(self, uids, acts):
        if uids == None:
            udis = [1,2,3,4,5]

        r = np.random.rand(len(uids), 1)
        d = np.random.choice(a=[False, True], size=(len(uids), 1), p = [0.5, 0.5])

        return self.GetObs(uids),r, d, None

def make_dummy_env(cfg):

    return DummyEnv()


class GrpcServer(dist_pb2_grpc.WALLEServicer):
    def __init__(self, env_fn):
        self._env_fn = env_fn
        self._env = None

    def Connect(self, request, context):
        if self._env == None:

            cfg = json.loads(request.cfg)
            self._env = self._env_fn(cfg)

        response = dist_pb2.EnvInfo()
        response.obs_dim.extend(self._env.observation_space.shape)
        response.act_dim.extend(self._env.action_space.shape)

        return response


    def Reset(self, request, context):
        uids = request.ids

        obs = self._env.reset(uids).dumps()
        # obs = self._env.get_observations().dumps()
        response = dist_pb2.Observation(obs=obs)
        return response


    def GetObs(self, request, context):
        # print(request.ids)
        uids = request.ids
        obs = self._env.get_observations(uids).dumps()
        return dist_pb2.Observation(obs=obs)

    def Step(self, request, context):
        uids = request.uids.ids
        acts = pkl.loads(request.actions)

        obs, r, d, inf = self._env.step(acts, uids)
        obs = obs.dumps()
        r = r.dumps()
        d = d.dumps()
        # inf = inf.dumps()

        rsp = dist_pb2.Records(obs= obs, r= r, d= d)
        return rsp

    def DisConnect(self, request, context):
        self._env.Disconnect()


def Serve(bind_addr = "127.0.0.1:18888", make_env_func =  make_dummy_env):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))

    dist_pb2_grpc.add_WALLEServicer_to_server(GrpcServer(make_env_func), server)

    server.add_insecure_port(bind_addr)
    server.start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("=======================================")
        print("=======================================")
        print("=======================================")
        print("=======================================")
        print("STOPPED THE SERVER")
        server.stop(0)

if __name__ == '__main__':
    Serve()

