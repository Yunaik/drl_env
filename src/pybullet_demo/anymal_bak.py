import pybullet as p
import pybullet_data as pd
import time, psutil, os
import numpy as np


useMaximalCoordinates = True
needLog = True
id = -1
class Anymal:
    def __init__(self, usePhysX, basePos = [0, 0, 0.55]):
        self.usePhysX = usePhysX
        self.anymalId = p.loadURDF("urdf/anymal_bedi_urdf/anymal.urdf")

        self.nJoints = p.getNumJoints(self.anymalId)
        self.joints = [p.getJointInfo(self.anymalId, n) for n in range(self.nJoints)]
        self.uFixJoints = [idx[0] for idx in self.joints if idx[2] != p.JOINT_FIXED]
        self.uFixJL = len(self.uFixJoints)


    def getAnymalInfo(self):

        # get Info
        pos, ori = p.getBasePositionAndOrientation(self.anymalId)
        v, ang = p.getBaseVelocity(self.anymalId)
        joint = p.getJointStates(self.anymalId, self.uFixJoints)

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
            'JointTq': jointTorq
        }
        return info

    def resetAnymal(self):
        for idx in range(self.nJoints):
            p.resetJointState(self.anymalId, idx, 0, 0)

    def setAnymal(self):
        pass


def env_init(usePhysX, useGUI ,freq, enlarge=1, Gpu=-1):

    if usePhysX:
        options = "--numCores=8 --solver=pgs"
        # if Gpu != -1:
        options += " --gpu=0 "
        if enlarge != 1:
            options += " --gmem_enlarge={}".format(enlarge)
        p.connect(p.PhysX, options = options)
        if useGUI:
            p.loadPlugin("eglRendererPlugin")
    elif useGUI:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pd.getDataPath())

    p.setPhysicsEngineParameter(fixedTimeStep=1. / freq, numSolverIterations=10, minimumSolverIslandSize=50)
    p.setPhysicsEngineParameter(contactBreakingThreshold=0.01)

    p.setGravity(0, 0, -9.8)

    # p.loadURDF("plane.urdf", useMaximalCoordinates=True)  # useMaximalCoordinates)

    anymal = Anymal(usePhysX)

    return anymal

def env_deinit():
    # p.stopStateLogging(id)
    p.disconnect()

def benchEngine(usePhysX, useGui, freq, SecThrd, enlarge = 1, sample_freq=25):
    anymal = env_init(usePhysX, useGui, freq, enlarge=enlarge)

    count = 0
    prevTime = time.time()
    curTime = prevTime
    while (1):
        count += 1
        if count % sample_freq == 0:
            if needLog:
                print(anymal.getAnymalInfo())
        if count % 10 == 0:
            curTime = time.time()
            if curTime - prevTime > SecThrd:
                break
            # print("count : ", count)

        curTime = time.time()

        p.stepSimulation()
        # anymal.resetAnymal()

    # env_deinit()

    cpu_usuage = psutil.Process(os.getpid()).cpu_times()

    rst = {
        "Engine": "physx" if usePhysX else "bullet3",
        "Gui": "use" if useGui else "not use",
        "Render": "egl" if usePhysX and useGui else "opengl" if (not usePhysX and useGui) else "none",
        "Time": curTime - prevTime,
        "Count": count,
        "aveCount": count / (curTime - prevTime),
        "cpu_user": cpu_usuage.user,
        "cpu_sys":  cpu_usuage.system
    }

    return rst

if __name__ == '__main__':

    usePhysX = False
    useGui = False
    freq = 1000
    SecThrd = 5
    enlarge = 10


    # rst1 = benchEngine(usePhysX=True, useGui=True, freq=freq, SecThrd=SecThrd, enlarge=enlarge)
    # rst2 = benchEngine(usePhysX=True, useGui=False, freq=freq, SecThrd=SecThrd, enlarge=enlarge)
    rst3 = benchEngine(usePhysX=False, useGui=True, freq=freq, SecThrd=SecThrd, enlarge=enlarge)
    # rst4 = benchEngine(usePhysX=False, useGui=False, freq=freq, SecThrd=SecThrd, enlarge=enlarge)

    # print(rst1)
    # print(rst2)
    print(rst3)
    # print(rst4)
