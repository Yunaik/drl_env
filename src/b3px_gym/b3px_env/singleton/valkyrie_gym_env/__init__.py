__package__ = "valkyrie_gym_env"
import scipy
import os, inspect, time
from pybullet_utils.bullet_client import BulletClient
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
# from rllab.envs.pybullet.valkyrie_multi_env.utils.util import quat_to_rot, rotX, rotY, rotZ
from b3px_gym.b3px_env.parallel.util import quat_to_rot, rotX, rotY, rotZ
import gym
import copy
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as pybullet
import math
from b3px_gym.b3px_env.singleton.valkyrie_gym_env.filter import FilterClass, KalmanFilter, BinaryFilter
from b3px_gym.b3px_env.singleton.valkyrie_gym_env.PD_controller import PDController
from b3px_gym.b3px_env.singleton.valkyrie_gym_env.sensor_signal_process import calCOP

Kp_default = dict([
    ("torsoYaw", 4500),
    ("torsoPitch", 4500),
    ("torsoRoll", 4500),
    ("rightHipYaw", 500),
    ("rightHipRoll", 1000),  # -0.49
    ("rightAnkleRoll", 300),  # -0.71
    ("leftHipYaw", 500),
    ("leftHipRoll", 1000),  # -0.49
    ("leftAnkleRoll", 300),  # -0.71
    ("rightShoulderPitch", 700),
    ("rightShoulderRoll", 1500),
    ("rightShoulderYaw", 200),
    ("rightElbowPitch", 200),
    ("leftShoulderPitch", 700),
    ("leftShoulderRoll", 1500),
    ("leftShoulderYaw", 200),
    ("leftElbowPitch", 200),

    ("leftHipPitch",    1*2000), #2000 # -0.49
    ("leftKneePitch",   1*2000), #2000 # 1.205
    ("leftAnklePitch",  1*3000), #4000 # -0.71
    ("rightHipPitch",   1*2000), #2000 # -0.49
    ("rightKneePitch",  1*2000), #2000 # 1.205
    ("rightAnklePitch", 1*3000), #4000 # -0.71
])

Kd_default = dict([
    ("torsoYaw", 30),
    ("torsoPitch", 30),
    ("torsoRoll", 30),
    ("rightHipYaw", 20),
    ("rightHipRoll", 30),  # -0.49
    ("rightAnkleRoll", 3),  # -0.71
    ("leftHipYaw", 20),
    ("leftHipRoll", 30),  # -0.49
    ("leftAnkleRoll", 3),  # -0.71
    ("rightShoulderPitch", 10),
    ("rightShoulderRoll", 30),
    ("rightShoulderYaw", 2),
    ("rightElbowPitch", 5),
    ("leftShoulderPitch", 10),
    ("leftShoulderRoll", 30),
    ("leftShoulderYaw", 2),
    ("leftElbowPitch", 5),

    ("leftHipPitch",    2.*100), #100 # -0.49
    ("leftKneePitch",   2.*30), #30 # 1.205
    ("leftAnklePitch",  2.*30), #30 # -0.71
    ("rightHipPitch",   2.*100), #100 # -0.49
    ("rightKneePitch",  2.*30), #30 # 1.205
    ("rightAnklePitch", 2.*30), #30 # -0.71
])


class Valkyrie(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    # def __del__(self):
    #     self._p.disconnect()

    def __init__(self,
                 client=None,
                 max_time=16,  # in seconds
                 initial_gap_time=0.01,  # in seconds
                 isEnableSelfCollision=False,
                 renders=True,
                #  PD_freq=500.0,
                #  Physics_freq=1000.0,
                 Kp=Kp_default,
                 Kd=Kd_default,
                 bullet_default_PD=True,
                 logFileName=None,
                 logger=None, links_to_read=None,
                 pos=None,
                 planeId=None,
                 fixed_base=False,
                 time_step=0.01,
                 frame_skip=4,
                 scaling_factor = 1.0,
                 margin_in_degree=10.,
                 useFullDOF=False,
                 regularise_action=False
                 ):
        self.regularise_action = regularise_action
        self.useFullDOF= useFullDOF
        self.timestep = time_step
        self.frame_skip=frame_skip
        self.robot_loaded  = False
        self.client = client
        # print("CLIENT", client)
        self.isPhysx = True if client is not None else False # True if client is not None else False
        # print("isPhysx: ", self.isPhysx)
        # self.getJointInfo = client.getJointInfoPhysX if self.isPhysx else pybullet.getJointInfo

        self.fixed_base = fixed_base
        self.planeID = planeId
        self.pos = pos
        if logger is not None:
            self.logger = logger

        self.links_to_read = links_to_read
        self.frames = []
        self.bullet_default_PD = bullet_default_PD

        self.jointIdx = {}
        self.jointNameIdx = {}
        
        self.jointLowerLimit = []
        self.jointUpperLimit = []
        self.total_mass = 0.0
        # self._p = p
        # self._seed()
        self._envStepCounter = 0
        self._renders = renders
        self._p = self.client if self.isPhysx and client is not None else BulletClient(connection_mode=pybullet.DIRECT)
        if self.useFullDOF:
            self.controlled_joints = [
                                    "rightHipYaw", 
                                    "rightHipRoll", 
                                    "rightHipPitch",
                                    "rightKneePitch",
                                    "rightAnklePitch",
                                    "rightAnkleRoll",
                                    "leftHipYaw", 
                                    "leftHipRoll", 
                                    "leftHipPitch",
                                    "leftKneePitch",
                                    "leftAnklePitch", 
                                    "leftAnkleRoll"
                                    ]
        else:
            self.controlled_joints = [
                                    "rightHipPitch",
                                    "rightKneePitch",
                                    "rightAnklePitch",
                                    "leftHipPitch",
                                    "leftKneePitch",
                                    "leftAnklePitch", ]

        self.nu = len(self.controlled_joints)
        self.r = -1
        self.PD_freq = 1/self.timestep*self.frame_skip
        self.Physics_freq = 1/self.timestep
        self._actionRepeat = int(self.Physics_freq/self.PD_freq)
        self._dt_physics = (1./ self.Physics_freq)
        self._dt_PD = (1. / self.PD_freq)
        self._dt = self._dt_physics # PD control loop timestep
        self._dt_filter = self._dt_PD #filter time step
        self.g = 9.81

        self.joint_states = {}
        self.u_max = dict([("torsoYaw", 190),
                           ("torsoPitch", 150),
                           ("torsoRoll", 150),
                           ("rightShoulderPitch", 190),
                           ("rightShoulderRoll", 190),
                           ("rightShoulderYaw", 65),
                           ("rightElbowPitch", 65),
                           ("rightForearmYaw", 26),
                           ("rightWristRoll", 14),
                           ("rightWristPitch", 14),
                           ("leftShoulderPitch", 190),
                           ("leftShoulderRoll", 190),
                           ("leftShoulderYaw", 65),
                           ("leftElbowPitch", 65),
                           ("leftForearmYaw", 26),
                           ("leftWristRoll", 14),
                           ("leftWristPitch", 14),
                           ("rightHipYaw", 190),
                           ("rightHipRoll", 350),
                           ("rightHipPitch", 350),
                           ("rightKneePitch", 350),
                           ("rightAnklePitch", 205),
                           ("rightAnkleRoll", 205),
                           ("leftHipYaw", 190),
                           ("leftHipRoll", 350),
                           ("leftHipPitch", 350),
                           ("leftKneePitch", 350),
                           ("leftAnklePitch", 205),
                           ("leftAnkleRoll", 205),
                           ("lowerNeckPitch", 50),
                           ("upperNeckPitch", 50),
                           ("neckYaw", 50)])

        self.v_max = dict([("torsoYaw", 5.89),
                           ("torsoPitch", 9),
                           ("torsoRoll", 9),
                           ("rightShoulderPitch", 5.89),
                           ("rightShoulderRoll", 5.89),
                           ("rightShoulderYaw", 11.5),
                           ("rightElbowPitch", 11.5),
                           ("leftShoulderPitch", 5.89),
                           ("leftShoulderRoll", 5.89),
                           ("leftShoulderYaw", 11.5),
                           ("leftElbowPitch", 11.5),
                           ("rightHipYaw", 5.89),
                           ("rightHipRoll", 7),
                           ("rightHipPitch", 6.11),
                           ("rightKneePitch", 6.11),
                           ("rightAnklePitch", 11),
                           ("rightAnkleRoll", 11),
                           ("leftHipYaw", 5.89),
                           ("leftHipRoll", 7),
                           ("leftHipPitch", 6.11),
                           ("leftKneePitch", 6.11),
                           ("leftAnklePitch", 11),
                           ("leftAnkleRoll", 11),
                           ("lowerNeckPitch", 5),
                           ("upperNeckPitch", 5),
                           ("neckYaw", 5)])
        # nominal joint configuration
        # self.q_nom = dict([("torsoYaw", 0.0),
        #                    ("torsoPitch", 0.0),
        #                    ("torsoRoll", 0.0),
        #                    ("lowerNeckPitch", 0.0),
        #                    ("neckYaw", 0.0),
        #                    ("upperNeckPitch", 0.0),
        #                    ("rightShoulderPitch", 0.300196631343),
        #                    ("rightShoulderRoll", 1.25),
        #                    ("rightShoulderYaw", 0.0),
        #                    ("rightElbowPitch", 0.785398163397),
        #                    ("leftShoulderPitch", 0.300196631343),
        #                    ("leftShoulderRoll", -1.25),
        #                    ("leftShoulderYaw", 0.0),
        #                    ("leftElbowPitch", -0.785398163397),
        #                    ("rightHipYaw", 0.0),
        #                    ("rightHipRoll", 0.0),
        #                    ("rightAnkleRoll", 0.0),
        #                    ("leftHipYaw", 0.0),
        #                    ("leftHipRoll", 0.0),
                           
        #                 #    ("rightHipPitch",    -0.49),  # -0.49
        #                 #    ("rightKneePitch",   1.205),  # 1.205
        #                 #    ("rightAnklePitch",  -0.71),  # -0.71
        #                 #    ("leftHipPitch",     -0.49),  # -0.49
        #                 #    ("leftKneePitch",    1.205),  # 1.205
        #                 #    ("leftAnklePitch",   -0.71),  # -0.71

        #                    ("rightHipPitch",    -0.49*0.5),  # -0.49
        #                    ("rightKneePitch",   1.205*0.5),  # 1.205
        #                    ("rightAnklePitch",  -0.71*0.5),  # -0.71
        #                    ("leftHipPitch",     -0.49*0.5),  # -0.49
        #                    ("leftKneePitch",    1.205*0.5),  # 1.205
        #                    ("leftAnklePitch",   -0.71*0.5),  # -0.71

        #                    ("leftAnkleRoll", 0.0)])
        if self.useFullDOF:
            self.q_nom = dict([
                           ("rightHipYaw", 0.0),
                           ("rightHipRoll", -0.1),
                           ("rightHipPitch",    -0.45*1.),  # -0.49
                           ("rightKneePitch",   0.944*1.),  # 1.205
                           ("rightAnklePitch",   -0.527*1.),  # -0.71
                           ("rightAnkleRoll", 0.1),
                           ("leftHipYaw", 0.0),
                           ("leftHipRoll", 0.1),
                           ("leftHipPitch",     -0.45*1.),  # -0.49
                           ("leftKneePitch",    0.944*1.),  # 1.205
                           ("leftAnklePitch",    -0.527*1.),  # -0.71
                           ("leftAnkleRoll", -0.1)
                            ])

        else:
            self.q_nom = dict([
                           ("rightHipPitch",    -0.45*1.),  # -0.49 , -0.49
                           ("rightKneePitch",   0.944*1.),  # 1.205 , 1.205
                           ("rightAnklePitch",  -0.527*1.),  # -0.71 , -0.8
                           ("leftHipPitch",    -0.45*1.),  # -0.49 , -0.49
                           ("leftKneePitch",    0.944*1.),  # 1.205 , 1.205
                           ("leftAnklePitch",  -0.527*1.),  # -0.71 , -0.8
                            ])
        self.q_nom_list = np.array(list(self.q_nom.values()))

        margin = margin_in_degree*3.14/180
        # print("MARGIN IN DEG: %.1f" % margin_in_degree)
        self.margin = margin
        if self.useFullDOF:
            self.joint_limits_low   = { 
                                        "rightHipYaw":     -margin+self.q_nom["rightHipYaw"],
                                        "rightHipRoll":    -margin+self.q_nom["rightHipRoll"],
                                        "rightHipPitch":   -margin+self.q_nom["rightHipPitch"],
                                        "rightKneePitch":  -margin+self.q_nom["rightKneePitch"],
                                        "rightAnklePitch": -margin+self.q_nom["rightAnklePitch"], 
                                        "rightAnkleRoll":  -margin+self.q_nom["rightAnkleRoll"], 
                                        "leftHipYaw":      -margin+self.q_nom["leftHipYaw"],
                                        "leftHipRoll":     -margin+self.q_nom["leftHipRoll"],
                                        "leftHipPitch":    -margin+self.q_nom["leftHipPitch"],
                                        "leftKneePitch":   -margin+self.q_nom["leftKneePitch"],
                                        "leftAnklePitch":  -margin+self.q_nom["leftAnklePitch"], 
                                        "leftAnkleRoll":   -margin+self.q_nom["leftAnkleRoll"] }
            self.joint_limits_high   = {
                                        "rightHipYaw":     +margin+self.q_nom["rightHipYaw"],
                                        "rightHipRoll":    +margin+self.q_nom["rightHipRoll"],
                                        "rightHipPitch":   +margin+self.q_nom["rightHipPitch"],
                                        "rightKneePitch":  +margin+self.q_nom["rightKneePitch"],
                                        "rightAnklePitch": +margin+self.q_nom["rightAnklePitch"], 
                                        "rightAnkleRoll":  +margin+self.q_nom["rightAnkleRoll"], 
                                        "leftHipYaw":      +margin+self.q_nom["leftHipYaw"],
                                        "leftHipRoll":     +margin+self.q_nom["leftHipRoll"],
                                        "leftHipPitch":    +margin+self.q_nom["leftHipPitch"],
                                        "leftKneePitch":   +margin+self.q_nom["leftKneePitch"],
                                        "leftAnklePitch":  +margin+self.q_nom["leftAnklePitch"], 
                                        "leftAnkleRoll":   +margin+self.q_nom["leftAnkleRoll"]}
        else:
            self.joint_limits_low   = { 
                                        "rightHipPitch":   -margin+self.q_nom["rightHipPitch"],
                                        "rightKneePitch":  -margin+self.q_nom["rightKneePitch"],
                                        "rightAnklePitch": -margin+self.q_nom["rightAnklePitch"], 
                                        "leftHipPitch":    -margin+self.q_nom["leftHipPitch"],
                                        "leftKneePitch":   -margin+self.q_nom["leftKneePitch"],
                                        "leftAnklePitch":  -margin+self.q_nom["leftAnklePitch"], 
                                        }
            self.joint_limits_high   = {
                                        "rightHipPitch":   +margin+self.q_nom["rightHipPitch"],
                                        "rightKneePitch":  +margin+self.q_nom["rightKneePitch"],
                                        "rightAnklePitch": +margin+self.q_nom["rightAnklePitch"], 
                                        "leftHipPitch":    +margin+self.q_nom["leftHipPitch"],
                                        "leftKneePitch":   +margin+self.q_nom["leftKneePitch"],
                                        "leftAnklePitch":  +margin+self.q_nom["leftAnklePitch"], 
                                        }

                

        #     self.joint_limits_low   = { 
        #                                 "rightHipPitch":   -margin,
        #                                 "rightKneePitch":  -margin,
        #                                 "rightAnklePitch": -margin, 
        #                                 "leftHipPitch":    -margin,
        #                                 "leftKneePitch":   -margin,
        #                                 "leftAnklePitch":  -margin, 
        #                                 }
        #     self.joint_limits_high   = {
        #                                 "rightHipPitch":   +margin,
        #                                 "rightKneePitch":  +margin,
        #                                 "rightAnklePitch": +margin, 
        #                                 "leftHipPitch":    +margin,
        #                                 "leftKneePitch":   +margin,
        #                                 "leftAnklePitch":  +margin, 
        #                                 }


        # self.Kp = Kp
        self.Kd = Kd

        # self.controllable_joints = ["torsoYaw", "torsoPitch", "torsoRoll", "lowerNeckPitch", "neckYaw",
        #                             "upperNeckPitch", "rightShoulderPitch", "rightShoulderRoll", "rightShoulderYaw",
        #                             "rightElbowPitch", "leftShoulderPitch", "leftShoulderRoll", "leftShoulderYaw",
        #                             "leftElbowPitch", "rightHipYaw", "rightHipRoll", "rightHipPitch", "rightKneePitch",
        #                             "rightAnklePitch", "rightAnkleRoll", "leftHipYaw", "leftHipRoll", "leftHipPitch",
        #                             "leftKneePitch", "leftAnklePitch", "leftAnkleRoll"]

        # self.uncontrolled_joints = [a for a in self.controllable_joints if a not in self.controlled_joints]

        # copy pasted for physx. Don't change urdf
        self.linkCOMPos = {}
        self.linkMass = {}
        if self.isPhysx:
            self.base_pos_nom = self.pos
            # self.base_pos_nom[2] = 1.18
        else:
            self.base_pos_nom = np.array([0, 0, 1.175])  # 1.175 straight #1.025 bend
        self.base_orn_nom = np.array([0, 0, 0, 1])  # x,y,z,w
        self.plane_pos_nom = np.array([0.,0.,0.])
        self.plane_orn_nom = np.array([0.,0.,0.,1.])


        self._setupSimulation()

        self._actionDim = len(self.controlled_joints)
        observationDim = 9+self._actionDim

        observation_high = np.array([np.finfo(np.float32).max] * observationDim)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self._observationDim = observationDim

        low = []
        high = []
        # print("Controlled joints: ", self.controlled_joints)
        for joint in self.controlled_joints:
            low.append(self.joint_limits_low[joint])
            high.append(self.joint_limits_high[joint])
        self.action_space = spaces.Box(np.array(low), np.array(high))
        # print("Action space: ", self.action_space.low)
        self.getLinkMass()
        # print("observationDim", self._observationDim, "actionDim", self._actionDim)
    def getLinkMass(self):
        self.total_mass = 0
        info = self._p.getDynamicsInfo(self.r, -1)  # for base link
        self.linkMass.update({"base": info[0]})
        # self.linkMass.update({"pelvisBase": info[0]})
        self.total_mass += info[0]
        for key, value in self.jointIdx.items():
            info = self._p.getDynamicsInfo(self.r, value)
            self.linkMass.update({key: info[0]})
            self.total_mass += info[0]
    def step(self, action):
        return 0, 0, self._observation, 0

    def render(self, mode='human', close=False, distance=3, yaw=0, pitch=-30, roll=0, ):
        # p.addUserDebugLine(self.COM_pos + np.array([0, 0, 2]), self.COM_pos + np.array([0, 0, -2]), [1, 0, 0], 5,
        #                    0.1)  # TODO rendering to draw COM
        # p.addUserDebugLine(self.support_polygon_center[0] + np.array([0, 0, 2]),
        #                    self.support_polygon_center[0] + np.array([0, 0, -2]), [0, 1, 0], 5,
        #                    0.1)  # TODO rendering to draw support polygon
        # p.addUserDebugLine(self.support_polygon_center[0] + np.array([2, 0, 0]),
        #                    self.support_polygon_center[0] + np.array([-2, 0, 0]), [0, 1, 0], 5,
        #                    0.1)  # TODO rendering to draw support polygon

        width = 1600
        height = 900
        base_pos, base_quat = self._p.getBasePositionAndOrientation(self.r)
        base_orn = self._p.getEulerFromQuaternion(base_quat)

        # yaw = base_orn[2]*180/math.pi
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=2,
        )
        #
        # view_matrix = p.computeViewMatrix(
        #     cameraTargetPosition=base_pos,
        #     cameraEyePosition=np.array(base_pos)+np.array([3,-3,2]),
        #     cameraUpVector=np.array(base_pos)+np.array([0,0,1]),
        # )

        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width)/height,
            nearVal=0.1,
            farVal=100.0,
        )

        # start_time = time.time()
        (_, _, px, _, _) = self._p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self._p.ER_TINY_RENDERER
        )
        #ER_TINY_RENDERER ER_BULLET_HARDWARE_OPENGL
        rgb_array = np.array(px)
        rgb_array = rgb_array[:,:,:3]
        # print("Time it took to get getCameraImage: %.5fs" % (time.time()-start_time))
        return rgb_array
    def reset(self):
        return self._reset()
    def _reset(self, Kp=Kp_default, Kd=Kd_default, base_pos_nom = None, base_orn_nom = None, fixed_base = False, q_nom = None):

        seed=int((time.time()*1e6)%1e9)
        
        np.random.seed(seed=seed)

        self.Kp = Kp
        self.Kd = Kd

        self._setupSimulation(base_pos_nom, base_orn_nom, fixed_base, q_nom)
        self._envStepCounter = 0

        self._observation = self.getExtendedObservation()
        #self._reading = self.getReading()
        return np.array(self._observation)

    def get_observation(self):
        return self.getExtendedObservation()

    def getExtendedObservation(self):
        self._observation = self.getFilteredObservation()  # filtered observation
        return self._observation

    # def _render(self, mode='human', close=False, distance=3, yaw=0, pitch=-30, roll=0, ):
    #     # p.addUserDebugLine(self.COM_pos + np.array([0, 0, 2]), self.COM_pos + np.array([0, 0, -2]), [1, 0, 0], 5,
    #     #                    0.1)  # TODO rendering to draw COM
    #     # p.addUserDebugLine(self.support_polygon_center[0] + np.array([0, 0, 2]),
    #     #                    self.support_polygon_center[0] + np.array([0, 0, -2]), [0, 1, 0], 5,
    #     #                    0.1)  # TODO rendering to draw support polygon
    #     # p.addUserDebugLine(self.support_polygon_center[0] + np.array([2, 0, 0]),
    #     #                    self.support_polygon_center[0] + np.array([-2, 0, 0]), [0, 1, 0], 5,
    #     #                    0.1)  # TODO rendering to draw support polygon

    #     width = 1600
    #     height = 900
    #     base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
    #     base_orn = p.getEulerFromQuaternion(base_quat)

    #     # yaw = base_orn[2]*180/math.pi
    #     view_matrix = p.computeViewMatrixFromYawPitchRoll(
    #         cameraTargetPosition=base_pos,
    #         distance=distance,
    #         yaw=yaw,
    #         pitch=pitch,
    #         roll=roll,
    #         upAxisIndex=2,
    #     )
    #     #
    #     # view_matrix = p.computeViewMatrix(
    #     #     cameraTargetPosition=base_pos,
    #     #     cameraEyePosition=np.array(base_pos)+np.array([3,-3,2]),
    #     #     cameraUpVector=np.array(base_pos)+np.array([0,0,1]),
    #     # )

    #     proj_matrix = p.computeProjectionMatrixFOV(
    #         fov=60,
    #         aspect=float(width)/height,
    #         nearVal=0.1,
    #         farVal=100.0,
    #     )

    #     # start_time = time.time()
    #     (_, _, px, _, _) = p.getCameraImage(
    #         width=width,
    #         height=height,
    #         viewMatrix=view_matrix,
    #         projectionMatrix=proj_matrix,
    #         renderer=p.ER_TINY_RENDERER
    #     )
    #     #ER_TINY_RENDERER ER_BULLET_HARDWARE_OPENGL
    #     rgb_array = np.array(px)
    #     rgb_array = rgb_array[:,:,:3]
    #     # print("Time it took to get getCameraImage: %.5fs" % (time.time()-start_time))
    #     return rgb_array

    # def _termination(self):
    #     return self.checkFall()

    # # TODO create function to log video
    # def _startLoggingVideo(self):
    #     self.logId = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
    #                                      fileName=self._logFileName + '/video.mp4')

    # def _stopLoggingVideo(self):
    #     # p.startStateLogging(self.logId)
    #     p.stopStateLogging(self.logId)

    def getReward(self):
        return self._reward()

    def applyForce(self, force=[0,0,0], linkName="base"):

        if linkName == 'base':
                index = -1
        else:
            index = self.jointIdx[linkName]#
        frame_flag = self._p.LINK_FRAME
        pos = [0,0,0]
        # if linkName is 'base':
        #     pos = self.base_pos
        # else:
        #     link_state = self._p.getLinkState(self.r, self.jointIdx[linkName])
        #     pos = link_state[0]

        self._p.applyExternalForce(self.r, index,
                                forceObj=force,
                                posObj=pos ,#[0, 0.0035, 0],
                                flags=frame_flag)

    def _reward(self):
        x_pos_err = 0 - self.base_pos[0]
        y_pos_err = self.pos[1] - self.base_pos[1]
        z_pos_err = 1.05868 - self.base_pos[2] #1.128
        # print("Self pos: %.2f, %.2f, %.2f " % (x_pos_err, y_pos_err, z_pos_err))
        # print("Base pos: ", self.base_pos)
        # print("VEL: ", self.base_vel_yaw)
        x_vel_err = -self.base_vel_yaw[0]
        y_vel_err = -self.base_vel_yaw[1]
        z_vel_err = -self.base_vel_yaw[2]

        chest_link_state = self._p.getLinkState(self.r, self.jointIdx['torsoRoll'], computeLinkVelocity=1)
        torso_pitch_err = 0-chest_link_state[1][1]
        pelvis_pitch_err = 0-self.base_orn[1]
        torso_roll_err = 0-chest_link_state[1][0]
        pelvis_roll_err = 0-self.base_orn[0]

        alpha = 1e-3#1e-2#1e-1
        x_pos_reward = math.exp(math.log(alpha)*(x_pos_err/0.7)**2) #1.0
        y_pos_reward = math.exp(math.log(alpha)*(y_pos_err/0.7)**2) #1.0
        z_pos_reward = math.exp(math.log(alpha)*(z_pos_err/0.7)**2)  #1.0

        x_vel_reward = math.exp(math.log(alpha)*(x_vel_err/1.0)**2) #4.0
        y_vel_reward = math.exp(math.log(alpha)*(y_vel_err/1.0)**2) #4.0
        z_vel_reward = math.exp(math.log(alpha)*(z_vel_err/1.0)**2)  #2.0

        torso_pitch_reward = math.exp(math.log(alpha)*(torso_pitch_err/1.57)**2)
        pelvis_pitch_reward = math.exp(math.log(alpha)*(pelvis_pitch_err/1.57)**2)
        torso_roll_reward = math.exp(math.log(alpha)*(torso_roll_err/1.57)**2)
        pelvis_roll_reward = math.exp(math.log(alpha)*(pelvis_roll_err/1.57)**2)

        # reward = (
        #             1.0 * x_pos_reward  + 0.0 * y_pos_reward + 6.0 * z_pos_reward\
        #             +1.0 * x_vel_reward + 1.0 * y_vel_reward + 1.0 * z_vel_reward \
        #             +0.0 * torso_pitch_reward + 1.0 * pelvis_pitch_reward \
        #             +0.0 * torso_roll_reward + 1.0 * pelvis_roll_reward \
        #           ) \
        #         * 1 / (1.0 + 0.0 + 6.0 + 1.0 +  1.0 + 1.0 + 1.0 + 1.0 )

        # print(self.total_mass)
        force_targ = -self.total_mass*self.g/2.0
        left_foot_force_err = force_targ-self.leftContactForce[2] # Z contact force
        right_foot_force_err = force_targ-self.rightContactForce[2]
        left_foot_force_reward  = math.exp(math.log(alpha)*(left_foot_force_err/force_targ)**2)
        right_foot_force_reward = math.exp(math.log(alpha)*(right_foot_force_err/force_targ)**2)
        if self.regularise_action:
            velocity_error  = 0
            torque_error    = 0
            for idx, key in enumerate(self.controlled_joints):
                velocity_error    += (self.joint_velocity[idx] / self.v_max[key])**2
                torque_error      += (self.joint_torques[idx]/ self.u_max[key])**2
            # print("Foot contact: ", foot_contact_term, "vel: %.2f, torque: %.2f, power: %.2f" % (velocity_penalty, torque_penalty, power_penalty))
            velocity_error /= len(self.controlled_joints)
            torque_error /= len(self.controlled_joints)

            joint_vel_reward = math.exp(math.log(alpha)*velocity_error)
            joint_torque_reward = math.exp(math.log(alpha)*velocity_error)
        else:
            joint_vel_reward    = 0
            joint_torque_reward = 0

        
        foot_contact_term = 0
        fall_term = 0
        if (self.leftFoot_isInContact and self.rightFoot_isInContact): # both feet lost contact
            foot_contact_term = 2.0#-5  # 1 TODO increase penalty for losing contact with the ground
        elif (self.leftFoot_isInContact or self.rightFoot_isInContact): # both feet lost contact
            foot_contact_term = 0.5#-5  # 1 TODO increase penalty for losing contact with the ground
        # if self.checkFall():
        #     fall_term -= 10

        reward = (
                     1.0 * x_pos_reward + 1.0 * y_pos_reward + 4.0 * z_pos_reward\
                    +1.0 * x_vel_reward + 1.0 * y_vel_reward + 1.0 * z_vel_reward \
                    +0.5 * torso_pitch_reward + 0.5 * pelvis_pitch_reward \
                    # +0.5 * torso_roll_reward + 0.5 * pelvis_roll_reward \
                    +1.0 * left_foot_force_reward + 1.0 * right_foot_force_reward \
                    +1.0 * joint_vel_reward + 1.0 * joint_torque_reward + 2.0* foot_contact_term
                  ) \
                * 10 / (2.0 + 4.0 + 2.0 + 1.0 + 1.0 +  1.0 + 1.0 + 2.0 + 2.0)




        # penalize reward when joint is moving too fast
        # velocity_penalty = 0
        # torque_penalty = 0
        # power_penalty = 0
        # if self.regularise_action:
            # for idx, key in enumerate(self.controlled_joints):
            #     velocity_penalty    -= (self.joint_velocity[idx] / self.v_max[key])**2
            #     torque_penalty      -= 0.1 * abs(self.joint_torques[idx]/ self.u_max[key])
            #     power_penalty       -= 0.1 * abs(self.joint_velocity[idx]) * abs(self.joint_torques[idx])
            # # print("Foot contact: ", foot_contact_term, "vel: %.2f, torque: %.2f, power: %.2f" % (velocity_penalty, torque_penalty, power_penalty))

        

        # reward += foot_contact_term #+fall_term+velocity_penalty+torque_penalty+power_penalty

        reward_term = dict([
            ("x_pos_reward", x_pos_reward),
            ("y_pos_reward", y_pos_reward),
            ("z_pos_reward", z_pos_reward),
            ("x_vel_reward", x_vel_reward),
            ("y_vel_reward", y_vel_reward),
            ("z_vel_reward", z_vel_reward),
            ("torso_pitch_reward", torso_pitch_reward),
            ("pelvis_pitch_reward", pelvis_pitch_reward),
            ("torso_roll_reward", torso_roll_reward),
            ("pelvis_roll_reward", pelvis_roll_reward),
            ("left_foot_force_reward", left_foot_force_reward),
            ("right_foot_force_reward", right_foot_force_reward)
        ])
        # print("Reward: %.4f" % (reward/20))
        # print("Reward: ", reward_term)
        return reward, reward_term

    def resetJointStates(self, base_pos_nom=None, base_orn_nom=None, q_nom=None):
        # if base_pos_nom is None:
        #     base_pos_nom = self.base_pos_nom
        # if base_orn_nom is None:
        #     base_orn_nom = self.base_orn_nom
        if q_nom is None:
            q_nom = self.q_nom
        else:
            #replace nominal joint angle with target joint angle
            temp=dict(self.q_nom)
            for key, value in q_nom.items():
                temp[key] = value
            q_nom = dict(temp)
            self.q_nom = dict(q_nom)

        for jointName in q_nom:
            self._p.resetJointState(self.r,
                              self.jointIdx[jointName],
                              targetValue=q_nom[jointName],
                              targetVelocity=0)
            # print("Reset %s to %.2f" % (jointName, q_nom[jointName]))
        # print(base_orn_nom)
        # self._p.resetBasePositionAndOrientation(self.r, base_pos_nom, base_orn_nom)
        # self._p.resetBaseVelocity(self.r, [0, 0, 0], [0, 0, 0])

    def calcCOM(self):
        self.linkCOMPos.update({"base": np.array(self.base_pos)}) # base position is the COM of the pelvis
        self.com_pos = np.zeros((1, 3))

        for key, value in self.linkMass.items():
            if key != "base":
                info = self._p.getLinkState(self.r, self.jointIdx[key], computeLinkVelocity=0)
                self.linkCOMPos.update({key: info[0]})
            
            # print("WTF: %.3f" % value,  np.array(self.linkCOMPos[key]))
            self.com_pos += np.array(self.linkCOMPos[key]) * value
            # print("KEY: %s, value: %.2f" % (key, value))
        self.com_pos /= self.total_mass
        # self.com_pos /= self.total_mass
        # update global COM position
        # self.com_pos = np.array(sum)
        return self.com_pos
    def _setupSimulation(self, base_pos_nom=None, base_orn_nom=None, fixed_base=False, q_nom=None):
        if base_pos_nom is None:
            base_pos_nom = self.base_pos_nom
        if self.pos is not None:
            base_pos_nom = self.pos
        if base_orn_nom is None:
            base_orn_nom = self.base_orn_nom
        #p.setPhysicsEngineParameter(numSolverIterations=10, erp=0.2) # Physics engine parameter default solver iteration = 50
        # self._setupFilter()

        # p.resetSimulation()

        self._p.setRealTimeSimulation(0)
        self._p.setGravity(0, 0, -self.g) #TODO set gravity
        # print("Time step: ", self._dt)
        self._p.setTimeStep(self._dt)

        self.dir_path = os.path.dirname(os.path.realpath(__file__))

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)
        plane_urdf = self.dir_path + "/plane/plane.urdf"

        if not self.isPhysx:
            # print("HI")
            self.plane = self._p.loadURDF(plane_urdf, basePosition=[0, 0, 0], baseOrientation=[0,0,0,1], useFixedBase=True)
        else:
            # print("PLANE ID: ", self.planeID)
            self.plane = self.planeID
        # print("PLANE: ", self.plane)
        if self.useFullDOF:
            valkyrie_urdf = self.dir_path + "/valkyrie_fixed.urdf"#"/valkyrie_bullet_mass_sims_modified_foot_collision_box_soft_contact.urdf"
        else:
            valkyrie_urdf = self.dir_path + "/valkyrie_reduced_fixed.urdf"#"/valkyrie_bullet_mass_sims_modified_foot_collision_box_soft_contact.urdf"
            # valkyrie_urdf = self.dir_path + "/valkyrie_complex.urdf"#"/valkyrie_bullet_mass_sims_modified_foot_collision_box_soft_contact.urdf"

        if not self.isPhysx:
            self.r = self._p.loadURDF(fileName=valkyrie_urdf,
                            basePosition=base_pos_nom,
                            baseOrientation=base_orn_nom,
                            flags=self._p.URDF_USE_INERTIA_FROM_FILE|self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                            # flags = p.URDF_USE_INERTIA_FROM_FILE,
                            useFixedBase=self.fixed_base,
                            )
        else:
            if self.robot_loaded:
            #     print("ßßßßßßßßßßßßßßßßßßßß")
            #     print(self.base_orn_nom)
            #     print
                self._p.resetBasePositionAndOrientation(self.r, self.base_pos_nom, [0.07265077634217172, -0.050834123935714896, 0.004993265519857639, 0.9960486051594151])
                # self._p.resetBasePositionAndOrientation(self.r, self.base_pos_nom, [0,0,0,1])
                self._p.resetBaseVelocity(self.r, [0, 0, 0], [0, 0, 0])
                # self._p.removeBody(self.r)
            else:
                # print("Load")
                self.robot_loaded = True

                self.r = self._p.loadURDF(fileName=valkyrie_urdf,
                                basePosition=self.base_pos_nom,
                                baseOrientation=self.base_orn_nom,
                                flags=self._p.URDF_USE_INERTIA_FROM_FILE|self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
                                # flags = p.URDF_USE_INERTIA_FROM_FILE,
                                useFixedBase=self.fixed_base,
                                )
        # , flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)  # , flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

        # self.jointIds = [29, 30, 31, 37, 38, 39]
        self.jointIds = []
        # for jointName in self.controlled_joints:
        #     self.jointIds.append(self.jointIdx[jointName])
        # print("Ids: ", self.jointIds)

        for j in range(self._p.getNumJoints(self.r)):
            self._p.changeDynamics(self.r, j, linearDamping=0, angularDamping=0)
            info = self._p.getJointInfo(self.r, j)
            #print(info)
            jointName = info[1].decode("utf-8")
            # print(jointName)
            jointType = info[2]
            if (jointType == self._p.JOINT_REVOLUTE):
                self.jointIds.append(j)
                # print("Appending %s" % jointName)

            self.jointIdx.update({jointName: info[0]})
            self.jointNameIdx.update({info[0]: jointName})

        # print("MARGIN: ", self.margin)
        if self.margin > 90*3.14/180:
            # print("Margin exceeding 90. Setting to read from urdf")

            for joint in self.controlled_joints:
                # print("self joint didx", self.jointIdx)
                info = self._p.getJointInfo(self.r, self.jointIdx[joint])
                self.joint_limits_low.update({joint: (info[8])})
                self.joint_limits_high.update({joint: (info[9])})
        self.resetJointStates(base_pos_nom, base_orn_nom, q_nom)

    def getObservation(self):
        self.base_pos, self.base_quat = self._p.getBasePositionAndOrientation(self.r)
        self.base_vel, self.base_orn_vel = self._p.getBaseVelocity(self.r)
        # print(self.base_pos)
        # for idx in self.jointNameIdx.keys():
        #     print("Val: ", self._p.getLinkState(self.r, idx+1)[0])
        # self.left_foot_pos = self._p.getLinkState(self.r, self.jointIdx["leftAnklePitch"])
        # self.right_foot_pos = self._p.getLinkState(self.r, self.jointIdx["rightAnklePitch"])
        # print("Left: ", self.left_foot_pos[0], ", right: ", self.right_foot_pos[0])


        # ankleRollContact = self._p.getContactPoints(self.r, self.plane, self.jointIdx['leftAnkleRoll'], -1)
        # anklePitchContact = self._p.getContactPoints(self.r, self.plane, self.jointIdx['leftAnklePitch'], -1)
        # ankleRollContact = self._p.getContactPoints(self.r, self.plane, self.jointIdx['rightAnkleRoll'], -1)
        # anklePitchContact = self._p.getContactPoints(self.r, self.plane, self.jointIdx['rightAnklePitch'], -1)
        leftContactInfo     = self._p.getContactPoints(self.r, self.plane, self.jointIdx["leftAnkleRoll"], -1)
        rightContactInfo    = self._p.getContactPoints(self.r, self.plane, self.jointIdx["rightAnkleRoll"], -1)
        # print(leftContactInfo)
        self.leftFoot_isInContact   = (len(leftContactInfo) > 0)
        self.rightFoot_isInContact  = (len(rightContactInfo) > 0)

        # print("LEFT: %d, RIGHT: %d" % (self.leftFoot_isInContact, self.rightFoot_isInContact))
        self.leftContactForce   = [0, 0, 0]
        self.rightContactForce  = [0, 0, 0]
        if self.leftFoot_isInContact:
            for info in leftContactInfo:
                contactNormal = np.array(info[7])  # contact normal of foot pointing towards plane
                contactNormal = -contactNormal  # contact normal of plane pointing towards foot
                contactNormalForce = np.array(info[9])
                F_contact = np.array(contactNormal)*contactNormalForce
                self.leftContactForce += F_contact

        if self.rightFoot_isInContact:
            for info in rightContactInfo:
                contactNormal = np.array(info[7])  # contact normal of foot pointing towards plane
                contactNormal = -contactNormal  # contact normal of plane pointing towards foot
                contactNormalForce = np.array(info[9])
                F_contact = np.array(contactNormal)*contactNormalForce
                self.rightContactForce += F_contact

        # print("LEFT CONTACT FORCE: ", self.leftContactForce)
        # print("RIGHT CONTACT FORCE: ", self.rightContactForce)

        for _id in self.jointIds:
            # print("%s: ID: %d" % (self.jointNameIdx[_id], _id))

            self.joint_states.update({self.jointNameIdx[_id]: self._p.getJointState(self.r, _id)})
            # print("NAME: ", name)
            # print(self.joint_states[name])
        """Observation"""   
        observation = []

        """Yaw adjusted base linear velocity"""
        self.base_orn = self._p.getEulerFromQuaternion(self.base_quat)
        Rz = rotZ(self.base_orn[2])
        self.Rz_i = np.linalg.inv(Rz)
        base_vel = np.array(self.base_vel)
        base_vel.resize(1, 3)
        self.base_vel_yaw = np.transpose(self.Rz_i @ base_vel.transpose())[0] # base velocity in adjusted yaw frame
        # self.base_vel_yaw[0][2] = 0.
        # base_vel_yaw_list = list(copy.copy(self.base_vel_yaw[0][:2]))
        base_vel_yaw_list = list(copy.copy(self.base_vel_yaw))

        observation.extend(copy.copy(base_vel_yaw_list))

        """Calculating gravity vector"""
        invBasePos, invBaseQuat = self._p.invertTransform([0,0,0], self.base_quat) 
        gravity = np.array([0,0,-1]) # in world coordinates
        gravity_quat = self._p.getQuaternionFromEuler([0,0,0])
        gravityPosInBase, gravityQuatInBase = self._p.multiplyTransforms(invBasePos, invBaseQuat, gravity, gravity_quat)
        self.base_gravity_vector = np.array(gravityPosInBase)
        observation.extend(copy.copy(self.base_gravity_vector))
        # print(self.base_gravity_vector)
        """base angular velocity"""
        R = quat_to_rot(self.base_quat)
        self.R_i = np.linalg.inv(R)

        base_orn_vel = np.array(self.base_orn_vel)
        base_orn_vel.resize(1,3)
        base_orn_vel_base = np.transpose(self.R_i @ base_orn_vel.transpose())

        base_orn_vel_base_list = list(copy.copy(base_orn_vel_base[0]))

        observation.extend(copy.copy(base_orn_vel_base_list))
        if self.useFullDOF:
            self.joint_position =  [
                                    self.joint_states["rightHipYaw"][0], 
                                    self.joint_states["rightHipRoll"][0], 
                                    self.joint_states["rightHipPitch"][0], 
                                    self.joint_states["rightKneePitch"][0],
                                    self.joint_states["rightAnklePitch"][0], 
                                    self.joint_states["rightAnkleRoll"][0], 
                                    self.joint_states["leftHipYaw"][0], 
                                    self.joint_states["leftHipRoll"][0], 
                                    self.joint_states["leftHipPitch"][0], 
                                    self.joint_states["leftKneePitch"][0], 
                                    self.joint_states["leftAnklePitch"][0], 
                                    self.joint_states["leftAnkleRoll"][0]
                                    ]  

            self.joint_velocity =  [
                                    self.joint_states["rightHipYaw"][1], 
                                    self.joint_states["rightHipRoll"][1], 
                                    self.joint_states["rightHipPitch"][1], 
                                    self.joint_states["rightKneePitch"][1],
                                    self.joint_states["rightAnklePitch"][1], 
                                    self.joint_states["rightAnkleRoll"][1], 
                                    self.joint_states["leftHipYaw"][1], 
                                    self.joint_states["leftHipRoll"][1], 
                                    self.joint_states["leftHipPitch"][1], 
                                    self.joint_states["leftKneePitch"][1], 
                                    self.joint_states["leftAnklePitch"][1], 
                                    self.joint_states["leftAnkleRoll"][1] 
                                    ]  
            self.joint_torques =  [
                                    self.joint_states["rightHipYaw"][-1], 
                                    self.joint_states["rightHipRoll"][-1], 
                                    self.joint_states["rightHipPitch"][-1], 
                                    self.joint_states["rightKneePitch"][-1],
                                    self.joint_states["rightAnklePitch"][-1], 
                                    self.joint_states["rightAnkleRoll"][-1], 
                                    self.joint_states["leftHipYaw"][-1], 
                                    self.joint_states["leftHipRoll"][-1], 
                                    self.joint_states["leftHipPitch"][-1], 
                                    self.joint_states["leftKneePitch"][-1], 
                                    self.joint_states["leftAnklePitch"][-1], 
                                    self.joint_states["leftAnkleRoll"][-1]
                                    ]  
        else:
            self.joint_position =  [
                                    self.joint_states["rightHipPitch"][0], 
                                    self.joint_states["rightKneePitch"][0],
                                    self.joint_states["rightAnklePitch"][0], 
                                    self.joint_states["leftHipPitch"][0], 
                                    self.joint_states["leftKneePitch"][0], 
                                    self.joint_states["leftAnklePitch"][0], 
                                    ]  

            self.joint_velocity =  [
                                    self.joint_states["rightHipPitch"][1], 
                                    self.joint_states["rightKneePitch"][1],
                                    self.joint_states["rightAnklePitch"][1], 
                                    self.joint_states["leftHipPitch"][1], 
                                    self.joint_states["leftKneePitch"][1], 
                                    self.joint_states["leftAnklePitch"][1], 
                                    ]  
            self.joint_torques =  [
                                    self.joint_states["rightHipPitch"][-1], 
                                    self.joint_states["rightKneePitch"][-1],
                                    self.joint_states["rightAnklePitch"][-1], 
                                    self.joint_states["leftHipPitch"][-1], 
                                    self.joint_states["leftKneePitch"][-1], 
                                    self.joint_states["leftAnklePitch"][-1], 
                                    ]  
        # print("HI")
        observation.extend(copy.copy(self.joint_position))

        """Return observation"""
        observation = np.array(observation)
        return observation

    def getObservationNoise(self):
        state = np.array(self.getObservation())
        state_noise = np.random.normal(state,1.0)
        # print(state-state_noise)
        return state_noise

    # def getJointAnglesDict(self):
    #     joint_angles = dict()
    #     for key in self.controllable_joints:
    #         index = self.jointIdx[key]
    #         joint_state = p.getJointState(self.r, index)
    #         angle = joint_state[0]
    #         joint_angles.update({key:angle})

    #     return joint_angles

    # def getJointAngles(self):
    #     joint_angles = []
    #     for key in self.controllable_joints:
    #         index = self.jointIdx[key]
    #         joint_state = p.getJointState(self.r, index)
    #         angle = joint_state[0]
    #         joint_angles.append(angle)

    #     return np.array(joint_angles)

    def getFilteredObservation(self):
        observation = self.getObservation()
        # observation = self.getObservationNoise()
        # observation_filtered = np.zeros(np.size(observation))

        # for i in range(self.stateNumber):
        #     observation_filtered[i] = self.state_filter_method[i].y[0]

        return observation

    def rotX(self, theta):
        R = np.array([ \
            [1.0, 0.0, 0.0], \
            [0.0, math.cos(theta), -math.sin(theta)], \
            [0.0, math.sin(theta), math.cos(theta)]])
        return R

    def rotY(self, theta):
        R = np.array([ \
            [math.cos(theta), 0.0, math.sin(theta)], \
            [0.0, 1.0, 0.0], \
            [-math.sin(theta), 0.0, math.cos(theta)]])
        return R

    def rotZ(self, theta):
        R = np.array([ \
            [math.cos(theta), -math.sin(theta), 0.0], \
            [math.sin(theta), math.cos(theta), 0.0], \
            [0.0, 0.0, 1.0]])
        return R

    def transform(self, qs):  # transform quaternion into rotation matrix
        qx = qs[0]
        qy = qs[1]
        qz = qs[2]
        qw = qs[3]

        x2 = qx + qx;
        y2 = qy + qy;
        z2 = qz + qz;
        xx = qx * x2;
        yy = qy * y2;
        wx = qw * x2;
        xy = qx * y2;
        yz = qy * z2;
        wy = qw * y2;
        xz = qx * z2;
        zz = qz * z2;
        wz = qw * z2;

        m = np.empty([3, 3])
        m[0, 0] = 1.0 - (yy + zz)
        m[0, 1] = xy - wz
        m[0, 2] = xz + wy
        m[1, 0] = xy + wz
        m[1, 1] = 1.0 - (xx + zz)
        m[1, 2] = yz - wx
        m[2, 0] = xz - wy
        m[2, 1] = yz + wx
        m[2, 2] = 1.0 - (xx + yy)

        return m

    def euler_to_quat(self, roll, pitch, yaw): #rad
        cy = np.cos(yaw*0.5)
        sy = np.sin(yaw*0.5)
        cr = np.cos(roll*0.5)
        sr = np.sin(roll*0.5)
        cp = np.cos(pitch*0.5)
        sp = np.sin(pitch*0.5)

        w = cy*cr*cp+sy*sr*sp
        x = cy*sr*cp-sy*cr*sp
        y = cy*cr*sp+sy*sr*cp
        z = sy*cr*cp-cy*sr*sp

        return [x,y,z,w]

    def checkFall(self):
        fall = False
        # if self.base_pos[2]>1.5:
        #     print("SELF BASE PSO: %.2f" % self.base_pos[2])
        # elif self.base_pos[2] < 0.8:
        #     print("SELF BASE PSO: %.2f" % self.base_pos[2])
        if self.base_pos[2]<=0.8 or self.base_pos[2]>1.4: #TODO check fall criteria
            fall = True
        return fall

    def set_pos(self, action):
        # if self.isPhysx:
        #     Kp = []
        #     Kd = []
        #     max_vel = []
        #     max_force = []
        #     for jointName in self.controlled_joints:
        #         Kp.append(self.Kp[jointName])
        #         Kd.append(self.Kd[jointName])
        #         max_vel.append(self.v_max[jointName])
        #         max_force.append(self.u_max[jointName])
        #         # jointIds.append(self.jointIdx[jointName])
        #     # jointIds = [29, 30, 31, 37, 38, 39] 
        #     # print("action: ", action)
        #     # print("JointIdx: ", (jointIds))
        #     # print("KP: ", Kp)
        #     # print("KD: ", Kd)
        #     # print("max_force: ", max_force)
        #     # joint ids are wrong
        #     self._p.setJointMotorControlArray(self.r, self.jointIds, self._p.POSITION_CONTROL, 
        #         targetPositions=action, positionGains=Kp, velocityGains=Kd, forces=max_force)
        # else:
        # print("Qnom: ", np.array(list(self.q_nom_list)))
        # print("ACtion: ", np.array(action))
        # absolute_joint_pos = np.array(self.q_nom_list)+np.array(action)
        
        joint_states = self._p.getJointStates(self.r, self.jointIds)
        torques = []
        v_max = []
        for n, _id in enumerate(self.jointIds):
            pos = joint_states[n][0]
            vel = joint_states[n][1]
            name = self.jointNameIdx[_id]
            # pos_ref = absolute_joint_pos[n]
            pos_ref = action[n]
            P_u = self.Kp[name] * (pos_ref - pos)
            D_u = self.Kd[name] * (0-vel)
            control_torque = P_u+D_u
            v_max.append(self.v_max[name])
            torques.append(control_torque)
        torques = np.clip(torques, -self.u_max[name], self.u_max[name])
        # print("Vmax: ", v_max)
        # print("torques: ", torques)
        self._p.setJointMotorControlArray(
            self.r, self.jointIds, targetVelocities=np.sign(torques) * v_max,
            forces=np.abs(torques),
            controlMode=self._p.VELOCITY_CONTROL,
        )
    
    def getJacobian(self, q, qdot=None, qddot=None, localPosition=None):
        if qdot == None:
            qdot = [0.]*len(q)
        if qddot == None:
            qddot = [0.]*len(q)
        if localPosition is None:
            localPosition = [0,0,0] # using CoM of link as position
        jacobian_lfoot = self._p.calculateJacobian(self.r, self.jointIdx["leftFoot"], localPosition=localPosition, objPositions=q, objVelocities=qdot, objAccelerations=qddot)
        jacobian_rfoot = self._p.calculateJacobian(self.r, self.jointIdx["rightFoot"], localPosition=localPosition, objPositions=q, objVelocities=qdot, objAccelerations=qddot)
        jacobian_pelvis = self._p.calculateJacobian(self.r, self.jointIdx["pelvis"], localPosition=localPosition, objPositions=q, objVelocities=qdot, objAccelerations=qddot)
        
        stacked_matrix = np.vstack((jacobian_lfoot, jacobian_rfoot))
        stacked_matrix = np.vstack((stacked_matrix, jacobian_pelvis))
        decomposed_matrix = scipy.linalg.lu(stacked_matrix)

        matrix1 = scipy.linalg.null_space(decomposed_matrix)
        matrix2 = scipy.linalg.null_space(stacked_matrix)
        print("Matrix 1: ", matrix1)
        print("Matrix 2: ", matrix2)

        # get random vector in nullspace
        random_vector = np.random.uniform(-1,1, len(q))
        random_vector = matrix1*random_vector

        # normalise vector
        norm = np.linalg.norm(random_vector, ord=1)
        if norm==0:
            norm=np.finfo(v.dtype).eps
        
        random_vector /= norm

        scale = [1.]*len(q)
        random_vector = np.multiply(random_vector, scale)