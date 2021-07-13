import copy
import math
import os
import numpy as np

from pybullet_envs.bullet import motor

INIT_POSITION = [0,0, 0.55]
INIT_POSITION2 = [0,0,1]
# INIT_ORIENTATION=[0, 0.707, 0.707, 0]
INIT_ORIENTATION = [0., 0., 0., 1]


MOTOR_VEL_LIMIT = np.inf

MOTOR_NAMES =['FR_hip_motor_2_chassis_joint',
            'FR_upper_leg_2_hip_motor_joint',
            'FR_lower_leg_2_upper_leg_joint',
            'FL_hip_motor_2_chassis_joint',
            'FL_upper_leg_2_hip_motor_joint',
            'FL_lower_leg_2_upper_leg_joint',
            'RR_hip_motor_2_chassis_joint' ,
            'RR_upper_leg_2_hip_motor_joint',
            'RR_lower_leg_2_upper_leg_joint',
            'RL_hip_motor_2_chassis_joint' ,
            'RL_upper_leg_2_hip_motor_joint',
            'RL_lower_leg_2_upper_leg_joint'
            ]

MOTOR_DICT = {}
REV_MOTOR_DICT = {}
MOTORS_VEL_LIMIT_DICT = None
MOTORS_VEL_LIMIT = None
MOTORS_TORQ_LIMIT_DICT= None
MOTORS_TORQ_LIMIT = None
KP_DICT = None
KD_DICT = None

JOINT_LIMIT_LOWER_DICT = dict([
    ("FR_hip_motor_2_chassis_joint", -0.6),
    ("FR_upper_leg_2_hip_motor_joint", -1.0),
    ("FR_lower_leg_2_upper_leg_joint", -2.3),
    ("FL_hip_motor_2_chassis_joint", -0.6),
    ("FL_upper_leg_2_hip_motor_joint", -1.0),
    ("FL_lower_leg_2_upper_leg_joint", -2.3),
    ("RR_hip_motor_2_chassis_joint", -0.6),
    ("RR_upper_leg_2_hip_motor_joint", -1.0),
    ("RR_lower_leg_2_upper_leg_joint", -2.3),
    ("RL_hip_motor_2_chassis_joint", -0.6),
    ("RL_upper_leg_2_hip_motor_joint", -1.0),
    ("RL_lower_leg_2_upper_leg_joint", -2.3),
])
JOINT_LIMIT_LOWER = np.asarray(JOINT_LIMIT_LOWER_DICT.values())

JOINT_LIMIT_UPPER_DICT = dict([
    ("FR_hip_motor_2_chassis_joint", 0.7),
    ("FR_upper_leg_2_hip_motor_joint", 3.14),
    ("FR_lower_leg_2_upper_leg_joint", 0.0),
    ("FL_hip_motor_2_chassis_joint", 0.7),
    ("FL_upper_leg_2_hip_motor_joint", 3.14),
    ("FL_lower_leg_2_upper_leg_joint", 0.0),
    ("RR_hip_motor_2_chassis_joint", 0.7),
    ("RR_upper_leg_2_hip_motor_joint", 3.14),
    ("RR_lower_leg_2_upper_leg_joint", 0.0),
    ("RL_hip_motor_2_chassis_joint", 0.7),
    ("RL_upper_leg_2_hip_motor_joint", 3.14),
    ("RL_lower_leg_2_upper_leg_joint", 0.0),
])

JOINT_LIMIT_UPPER = np.asarray(JOINT_LIMIT_UPPER_DICT.values())


FEET = {"FL": "FL_lower_leg_2_foot_joint",
        "FR": "FR_lower_leg_2_foot_joint",
        "RL": "RL_lower_leg_2_foot_joint",
        "RR": "RR_lower_leg_2_foot_joint"}

# FEET = {
#     'FL' : 'FL_lower_leg_2_upper_leg_joint',
#     'FR' : 'FR_lower_leg_2_upper_leg_joint',
#     'RL' : 'RL_lower_leg_2_upper_leg_joint',
#     'RR' : 'RR_lower_leg_2_upper_leg_joint'
# }

# Batch Version of Laikago

class BatchLaikago(object):

    def __init__(self, client ,enable_pd = False, kp=400, kd=10, sim_freq = 1000, ctrl_freq = 500, cmd_freq=25, motor_vel_limit = 10, motor_torq_limix = 40,urdf='/laikago.urdf', urdf_root=None, fixed = False, usePhysx = False, plane = -1, pos = None):
        global MOTOR_VEL_LIMIT_DICT, MOTOR_VEL_LIMIT, MOTORS_TORQ_LIMIT_DICT, MOTORS_TORQ_LIMIT, KP_DICT, KD_DICT
        global MOTOR_DICT, REV_MOTOR_DICT

        self._p = client
        self.enable_pd = enable_pd
        self.planeId = plane
        if enable_pd:
            self.kp = kp
            self.kd = kd
            self._motor = motor.MotorModel(torque_control_enabled = False, kp = kp, kd = kd)
        self.sim_freq  = sim_freq
        self.ctrl_freq = ctrl_freq
        self.cmd_freq  = cmd_freq

        MOTOR_VEL_LIMIT_DICT = dict([(MOTOR_NAMES[motor_index], motor_vel_limit) for motor_index in range(len(MOTOR_NAMES))])
        MOTOR_VEL_LIMIT = np.asarray([motor_vel_limit]*len(MOTOR_VEL_LIMIT_DICT))
        MOTORS_TORQ_LIMIT_DICT = dict([(MOTOR_NAMES[motor_index], motor_torq_limix) for motor_index in range(len(MOTOR_NAMES))])
        MOTORS_TORQ_LIMIT = np.asarray([motor_torq_limix] * len(MOTORS_TORQ_LIMIT_DICT))
        KP_DICT = dict([(MOTOR_NAMES[motor_index], kp) for motor_index in range(len(MOTOR_NAMES))])
        KD_DICT = dict([(MOTOR_NAMES[motor_index], kd) for motor_index in range(len(MOTOR_NAMES))])

        # Inner timestamp counter
        self.time = 0

        if pos == None:
            pos = INIT_POSITION2 if fixed else INIT_POSITION

        # Get Uid
        if fixed:
            self.uid = self._p.loadURDF(os.path.join(urdf_root, urdf), pos, INIT_ORIENTATION, useFixedBase=True)
        else:
            self.uid = self._p.loadURDF(os.path.join(urdf_root, urdf), pos, INIT_ORIENTATION)

        self.getJointInfo = self._p.getJointInfoPhysX if usePhysx else self._p.getJointInfo

        # Get joint number
        self.jNum = self._p.getNumJoints(self.uid)
        # Get joint infos
        self.jInfos = [self.getJointInfo(self.uid, n) for n in range(self.jNum)]

        # Get Uncontact Joints

        self.unFootJoints = [joint[0] for joint in self.jInfos if joint[1].decode() not in list(FEET.values())]


        # Get Unfixed Joints ids
        self.unFixJoints = [idx[0] for idx in self.jInfos \
                            if idx[2] == self._p.JOINT_PRISMATIC or idx[2] == self._p.JOINT_REVOLUTE]


        for idx in self.unFixJoints:
            info = self.getJointInfo(self.uid, idx)
            MOTOR_DICT.update({info[1]: info[0]})
            REV_MOTOR_DICT.update({info[0]: info[1]})

        self.unFixJL = len(self.unFixJoints)

        self._observation = []
        self.sim_ts = 1 / sim_freq
        # List all Motors
        self._motors_ids = self.unFixJoints

        # Every ctrl_step, update ctrl info once time
        self.ctrl_step = int(np.ceil(self.sim_freq / self.ctrl_freq))
        # Loop cmd_step simulation times once apply action
        self.cmd_step  = int(np.ceil(self.sim_freq / self.cmd_freq))
        self._motor_vel_limit = motor_vel_limit

        self._p.setPhysicsEngineParameter(fixedTimeStep=self.sim_ts)

        # Remember the init info
        # base info
        pos, ori = self.GetBasePos()
        vel, ang = self.GetBaseVel()

        # joints info
        jss = self.GetJointStates()
        jPos = [x[0] for x in jss]
        jVel = [x[0] for x in jss]
        jTorq = [x[0] for x in jss]

        self.InitInfo = {
            "Pos": pos,
            "Ori": ori,
            "Vel": vel,
            "Ang": ang,
            "JPos": jPos,
            "JVel": jVel,
            "JTorq": jTorq
        }
        # self.InitMotor()
        self.Reset()

    def GetObs(self):
        self._observation = self.GetJointStates()
        return self._observation

    def GetPosObs(self):
        obs = self.GetObs()
        return [s[0] for s in obs]

    def Reset(self):
        self._time = 0
        self._p.resetBasePositionAndOrientation(self.uid, INIT_POSITION, INIT_ORIENTATION)
        self._p.resetBaseVelocity(self.uid, [0, 0, 0], [0, 0, 0])
        self.ResetPos()

    def ResetPos(self):
        """ Reset all lengths pose with init state

        :return: None
        """
        for idx in range(self.unFixJL):
            self._p.resetJointState(self.uid, idx,
                                    self.InitInfo["JPos"][idx],
                                    self.InitInfo["JVel"][idx])

    def Apply_Pos(self, act_cmd, record_joints = False):
        """
        Robot Simulation Env, every cmd exec need to run (sim_freq / cmd_freq) sim_tims steps;

        :param motor_cmd: Action,
        :return:
        """
        # limit motor vel, if surplus, then clip

        if record_joints:
            cur_cmd_js = []
            # cur_cmd_js.append(self.GetPosObs())

        for _ in range(self.cmd_step):

            # print("pos : ", self.GetMotorAng())
            # print("vel : ", self.GetMotroVel())
            # print("torq : ", self.GetMotorTorq())
            if self.time % self.ctrl_step == 0:
                self.SetMotorPos(act_cmd)

            self._p.stepSimulation()
            self.time += 1

            if record_joints:
                cur_cmd_js.append(self.GetPosObs())
                return cur_cmd_js

    def GetJointInfo(self, jIdx):
        return self.jInfos[jIdx]

    def GetJointInfos(self):
        return self.jInfos

    def GetJointState(self, jIdx):
        return self._p.GetJointState(self.uid, jIdx)

    def GetJointStates(self, ids = None):
        jIdxs = ids if ids != None else self.unFixJoints
        return self._p.getJointStates(self.uid, jIdxs)

    def GetMotorAng(self):
        return np.asarray([s[0] for s in self.GetJointStates(self._motors_ids)])

    def GetMotroVel(self):
        return np.asarray([s[1] for s in self.GetJointStates(self._motors_ids)])

    def GetMotorTorq(self):
        return np.asarray([s[3] for s in self.GetJointStates(self._motors_ids)])

    def GetBasePos(self):
        return self._p.getBasePositionAndOrientation(self.uid)

    def GetBaseVel(self):
        return self._p.getBasePositionAndOrientation(self.uid)

    def CheckJointContackPoint(self, jIdx):
        if len(self.GetJContactPoint(jIdx)) >0 :
            return True
        else:
            return False

    def CheckJointsContackGround(self):
        for jIdx in self.unFootJoints:
            if self.CheckJointContackPoint(jIdx):
                return True
        return False

    def CheckBaseContackGround(self):
        if len(self.GetJContactPoint(-1)) > 0:
            return True
        else:
            return False

    def GetJContactPoint(self, jIdx):
        return self._p.getContactPoints(self.uid, self.planeId, jIdx, -1)

    def GetJContactPoints(self):
        return [self.GetJContactPoint(jIdx) for jIdx in self.unFootJoints]

    def InitMotor(self):
        for mid in self._motors_ids:
            self._p.setJointMotorControl2(
                self.uid,
                mid,
                controlMode=self._p.POSITION_CONTROL,
                force=0
            )

    def SetMotorVel(self, mid, torq):
        self._p.setJointMotorControl2(
            self.uid,
            mid,
            targetVelocity=np.sign(torq) * 10,
            controlMode=self._p.VELOCITY_CONTROL,
            force=np.abs(torq)
        )


    def SetMotorVels(self, torq):
        # self._p.setJointMotorControlArray(
        #     self.uid,
        #     jointIndices = self._motors_ids,
        #     targetVelocities = np.asarray([20]*12),
        #     forces = np.asarray([40]*12),
        #     controlMode=self._p.VELOCITY_CONTROL,
        # )
        print("MOTOR_VEL_LIMIT : " , MOTOR_VEL_LIMIT)
        self._p.setJointMotorControlArray(
            self.uid,
            jointIndices=self._motors_ids,
            targetVelocities=np.sign(torq) * MOTOR_VEL_LIMIT,
            forces=np.abs(torq),
            controlMode=self._p.VELOCITY_CONTROL,
            # velocityGains=[0.005]*12,
        )

    def SetMotorPos(self, pos):
        self._p.setJointMotorControlArray(
            self.uid,
            jointIndices    =   self._motors_ids,
            forces          =   MOTORS_TORQ_LIMIT,
            controlMode     =   self._p.POSITION_CONTROL,
            targetPositions =   pos,
            positionGains   =   [self.kp] *self.unFixJL,
            velocityGains   =   [self.kd] *self.unFixJL,
            # velocityGains=[0.005]*12,
        )
