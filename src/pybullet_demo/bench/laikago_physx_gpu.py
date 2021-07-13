import pybullet as p
import pybullet_data as pd
import time, psutil, os
import numpy as np
from uuid import uuid1
import rocksdb, json
from multiprocessing import Process
import copy


useMaximalCoordinates = True
needLog = False
# db = rocksdb.DB("fyp.db", rocksdb.Options(create_if_missing=True))

logid = 0

class Laikago:
    def __init__(self, usePhysX, uuid, basePos = [0, 0, 0]):

        self.usePhysX = usePhysX
        self.uuid = uuid

        self.uId = p.loadURDF("laikago/laikago.urdf", basePos, [0, 0.707, 0.707, 0])

        self.nJoints = p.getNumJoints(self.uId)
        self.joints = [p.getJointInfo(self.uId, n) for n in range(self.nJoints)]
        self.uFixJoints = [idx[0] for idx in self.joints if idx[2] != p.JOINT_FIXED]
        self.uFixJL = len(self.uFixJoints)


    def getInfo(self, count):

        # get Info
        pos, ori = p.getBasePositionAndOrientation(self.uId)
        v, ang = p.getBaseVelocity(self.uId)
        joint = p.getJointStates(self.uId, self.uFixJoints)

        jointPos =  [x[0] for x in joint]
        jointV   =  [x[1] for x in joint]
        jointTorq = [x[2] for x in joint]

        info = {
            'Pos': pos,
            'Ori': ori,
            'Vel': v,
            'Ang': ang,

            'JointPos': jointPos,
            'JointVel': jointV,
            'JointTq': jointTorq,
            'uuid': self.uuid
        }
        return info

    def resetArm(self):
        for idx in range(self.nJoints):
            p.resetJointState(self.uId, idx, 0, 0)

    def setArm(self):
        pass


def env_init(bCfg):

    if bCfg['physx']:
        options = "--numCores={}".format(bCfg['core'])
        options += " --solver=tgs" if bCfg['solver'] else " --solver=pgs"
        if bCfg['gpu']:
            options += " --gpu=1"
        if bCfg['enlarge'] != 1:
            options += " --gmem_enlarge={}".format(bCfg['enlarge'])
        print(options)
        p.connect(p.PhysX, options = options)
        if bCfg['gui']:
            p.loadPlugin("eglRendererPlugin")
    elif bCfg['gui']:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    if bCfg['profile']:
        global logid
        logid = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, bCfg['profile_name'])

    p.setAdditionalSearchPath(pd.getDataPath())

    p.setPhysicsEngineParameter(fixedTimeStep = 1. / bCfg['freq'])
    p.setPhysicsEngineParameter(numSolverIterations = bCfg['numSolverIterations'])
    p.setPhysicsEngineParameter(minimumSolverIslandSize = bCfg['minimumSolverIslandSize'])
    p.setPhysicsEngineParameter(contactBreakingThreshold = bCfg['contactBreakingThreshold'])

    p.setGravity(0, 0, -9.8)

    p.loadURDF("plane.urdf", useMaximalCoordinates=True)  # useMaximalCoordinates)

    ins_num = bCfg['instance']
    u = np.floor(np.sqrt(ins_num)) + 1
    dist = bCfg['instance_dist']
    offset = u*dist/2
    return [Laikago(bCfg['physx'], bCfg['uuid'] + str(i), basePos = [ i // u * dist-offset, i % u * dist-offset, 0.47]) for i in range(ins_num)]

def env_deinit():
    global  logid
    p.stopStateLogging(logid)
    p.disconnect()


def benchEngine(benchCfg):
    uuid = str(uuid1())
    benchCfg['uuid'] = uuid
    print(uuid)
    arms = env_init(benchCfg)

    sample_freq = benchCfg['sample_freq']

    count = 0
    max_count = benchCfg['freq'] * benchCfg['sec']
    prevTime = time.time()
    while (count < max_count):
        count += 1
        if count % sample_freq == 0:
            if benchCfg['need_log']:
                logs = [arm.getInfo(count) for arm in  arms]
                # db.put(str.encode(uuid), str.encode(json.dumps(logs)))


        p.stepSimulation()
        # arm.resetArm()

    curTime = time.time()
    print("finished ", uuid)

    env_deinit()

    cpu_usuage = psutil.Process(os.getpid()).cpu_times()

    rst = {
        "time": curTime - prevTime,
        "count": max_count,
        "simRate": benchCfg['sec'] / (curTime - prevTime) * benchCfg['instance'],
        "cpu_user": cpu_usuage.user,
        "cpu_sys":  cpu_usuage.system
    }
    rst.update(benchCfg)
    print(rst)

def do_once(cfg):
    proc = Process(target=benchEngine, args = (cfg,))
    proc.start()
    proc.join()


def ComputePerfBench(defaultBenchCfg):
    # Test 1 bullet * 1
    cfg = copy.deepcopy(defaultBenchCfg)
    do_once(cfg)

    # Test 2 bullet * 10
    cfg = copy.deepcopy(defaultBenchCfg)
    cfg['instance'] = 10
    do_once(cfg)

    # Test 3 bullet * 100
    cfg = copy.deepcopy(defaultBenchCfg)
    cfg['instance'] = 100
    do_once(cfg)

    # Test 4 PhysX cpu * 1
    cfg = copy.deepcopy(defaultBenchCfg)
    cfg['physx'] = True
    do_once(cfg)

    # Test 5 PhysX cpu * 10
    cfg = copy.deepcopy(defaultBenchCfg)
    cfg['physx'] = True
    cfg['instance'] = 10
    do_once(cfg)

    # Test 6 PhysX cpu * 100
    cfg = copy.deepcopy(defaultBenchCfg)
    cfg['physx'] = True
    cfg['instance'] = 100
    do_once(cfg)

    # Test 7 PhysX cpu * 1000
    cfg = copy.deepcopy(defaultBenchCfg)
    cfg['physx'] = True
    cfg['instance'] = 1000
    do_once(cfg)

    # Test 8 PhysX gpu * 1
    cfg = copy.deepcopy(defaultBenchCfg)
    cfg['physx'] = True
    cfg['gpu'] = True
    cfg['instance'] = 1
    do_once(cfg)

    # Test 9 PhysX gpu * 10
    cfg = copy.deepcopy(defaultBenchCfg)
    cfg['physx'] = True
    cfg['gpu'] = True
    cfg['instance'] = 10
    do_once(cfg)

    # Test 10 PhysX gpu * 100
    cfg = copy.deepcopy(defaultBenchCfg)
    cfg['physx'] = True
    cfg['gpu'] = True
    cfg['instance'] = 100
    do_once(cfg)

    # Test 11 PhysX gpu * 1000
    cfg = copy.deepcopy(defaultBenchCfg)
    cfg['physx'] = True
    cfg['gpu'] = True
    cfg['instance'] = 1000
    do_once(cfg)

    print("finished")

    # Test 12 PhysX gpu * 5000
    cfg = copy.deepcopy(defaultBenchCfg)
    cfg['physx'] = True
    cfg['gpu'] = True
    cfg['instance'] = 5000
    do_once(cfg)



def ComputePerfBenchPybullet(defaultBenchCfg):
    # for i in range(1,302,20):
    #     cfg = copy.deepcopy(defaultBenchCfg)
    #     cfg['instance'] = i
    #     do_once(cfg)

    for i in range(350, 1001, 50):
        cfg= copy.deepcopy(defaultBenchCfg)
        cfg['instance'] = i
        do_once(cfg)

def ComputePerfBenchPhysxCpu(defaultBenchCfg):
    defaultBenchCfg['physx'] = True
    # for i in range(1,302,20):
    #     cfg = copy.deepcopy(defaultBenchCfg)
    #     cfg['instance'] = i
    #     do_once(cfg)

    # for i in range(350,502,50):
    #     cfg = copy.deepcopy(defaultBenchCfg)
    #     cfg['instance'] = i
    #     do_once(cfg)

    for i in range(550, 1001, 50):
        cfg = copy.deepcopy(defaultBenchCfg)
        cfg['instance'] = i
        do_once(cfg)


def ComputePerfBenchPhysxGpu(defaultBenchCfg):
    defaultBenchCfg['physx'] = True
    defaultBenchCfg['gpu'] = True
    defaultBenchCfg['sec'] = 3

    # for i in range(1, 101, 10):
    #     cfg = copy.deepcopy(defaultBenchCfg)
    #     cfg['instance'] = i
    #     do_once(cfg)
    #
    # for i in range(200, 1001, 200):
    #     cfg = copy.deepcopy(defaultBenchCfg)
    #     cfg['instance'] = i
    #     do_once(cfg)

    for i in range(1500, 10000, 1000):
        cfg = copy.deepcopy(defaultBenchCfg)
        cfg['instance'] = i
        do_once(cfg)

def SimulationPerf(defaultBenchCfg):

    defaultBenchCfg['need_log'] = True

    #Solver Test

    # Bullet3 Solver
    cfg = defaultBenchCfg.copy()
    do_once(cfg)

    # Physx Pgs Solver
    # cfg = defaultBenchCfg.copy()
    # cfg['solver'] = 0
    # cfg['physx'] = True
    # do_once(cfg)

    # Physx Tgs Solver
    # cfg = defaultBenchCfg.copy()
    # cfg['physx'] = True
    # cfg['solver'] = 1
    # do_once(cfg)

    # # Physx Pgs Solver GPU
    # cfg = defaultBenchCfg
    # cfg['solver'] = 0
    # cfg['physx'] = True
    # cfg['gpu'] = True
    # do_once(cfg)
    #
    #
    # # Physx Tgs Solver GPU
    # cfg = defaultBenchCfg
    # cfg['solver'] = 1
    # cfg['physx'] = True
    # cfg['gpu'] = True
    # do_once(cfg)


if __name__ == '__main__':

    defaultBenchCfg = {
        'physx'  : False,
        'gui'    : False,
        'freq'      : 1000,
        'sec'       : 10,
        'gpu'       : False,
        'core'      : 1,
        'enlarge'   : 50,
        'instance'  : 1,
        'instance_dist': 5,
        'solver'    : 0,        # 0 : pgs, 1 : tgs
        'numSolverIterations': 10,
        'minimumSolverIslandSize':1024,
        'contactBreakingThreshold':0.01,
        'sample_freq': 25,
        'need_log': False,
        'profile': False,
    }

    # ComputePerfBench(defaultBenchCfg)
    # ComputePerfBenchPybullet(defaultBenchCfg)
    # ComputePerfBenchPhysxCpu(defaultBenchCfg)
    ComputePerfBenchPhysxGpu(defaultBenchCfg)

    print("finished")
