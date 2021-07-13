import copy
import math
import os
import numpy as np
from numpy import pi
import time
# from pybullet_envs.bullet import motor
from rllab.envs.pybullet.valkyrie_multi_env.utils.filter_array import FilterClass
from rllab.envs.pybullet.valkyrie_multi_env.utils.util import quat_to_rot, rotX, rotY, rotZ
from b3px_gym.b3px_env.singleton.anymal_fall_poses import AnymalConfig
from gym.spaces import Box

INIT_POSITION = [0,0, 0.58]
INIT_POSITION2 = [0,0,0.58]
INIT_ORIENTATION = [0., 0., 0., 1]


joint_limit_lower_dict = dict([
    ("FL_HipX_joint",  -30*pi/180.),
    ("FL_HipY_joint",  -90*pi/180.),
    ("FL_Knee_joint",  -120*pi/180.),
    ("FR_HipX_joint",  -0*pi/180.),
    ("FR_HipY_joint",  -90*pi/180.),
    ("FR_Knee_joint",  -120*pi/180.),
    ("HL_HipX_joint",  -30*pi/180.),
    ("HL_HipY_joint",  -70*pi/180.),
    ("HL_Knee_joint",  -0*pi/180.),
    ("HR_HipX_joint",  -0*pi/180.),
    ("HR_HipY_joint",  -70*pi/180.),
    ("HR_Knee_joint",  -0*pi/180.),
])
JOINT_LIMIT_LOWER = np.asarray(list(joint_limit_lower_dict.values()))

joint_limit_upper_dict = dict([
    ("FL_HipX_joint",  0*pi/180.),
    ("FL_HipY_joint",  70*pi/180.),
    ("FL_Knee_joint",  0*pi/180.),
    ("FR_HipX_joint",  30*pi/180.),
    ("FR_HipY_joint",  70*pi/180.),
    ("FR_Knee_joint",  0*pi/180.),
    ("HL_HipX_joint",  0*pi/180.),
    ("HL_HipY_joint",  90*pi/180.),
    ("HL_Knee_joint",  120*pi/180.),
    ("HR_HipX_joint",  30*pi/180.),
    ("HR_HipY_joint",  90*pi/180.),
    ("HR_Knee_joint",  120*pi/180.),
])

JOINT_LIMIT_UPPER = np.asarray(list(joint_limit_upper_dict.values()))

class Anymal(object):

    def __init__(self, client,
                        _motor_hip_x_kp=100,
                        _motor_hip_x_kd=1,
                        _motor_hip_y_kp=100,
                        _motor_hip_y_kd=1,
                        _motor_knee_kp =100,
                        _motor_knee_kd =1,
                        sim_freq = 1000, 
                        ctrl_freq = 500, 
                        cmd_freq=25,
                        motor_torq_limit = 80,
                        urdf='/anymal.urdf', urdf_root=None, 
                        fixed = False, 
                        usePhysx = False, 
                        plane = -1, 
                        pos = None, 
                        max_vel=10,
                        action_bandwidth=10,
                        nominal_base_height=0.5,
                        print_early_termination_reason=False,
                        do_rsi=False):
        # global joint_limit_lower, joint_limit_upper

        # Flags:
        self.do_rsi = do_rsi
        self.print_early_termination_reason = print_early_termination_reason
        self.nominal_base_height = nominal_base_height
        self._p = client

        self.planeId = plane
        self._motor_hip_x_kp = _motor_hip_x_kp
        self._motor_hip_x_kd = _motor_hip_x_kd
        self._motor_hip_y_kp = _motor_hip_y_kp
        self._motor_hip_y_kd = _motor_hip_y_kd
        self._motor_knee_kp = _motor_knee_kp
        self._motor_knee_kd = _motor_knee_kd
        
        self.sim_freq  = sim_freq
        self.ctrl_freq = ctrl_freq
        self.cmd_freq  = cmd_freq
        self.action_bandwidth = action_bandwidth


        self.joint_limit_lower_dict = dict([
            ("FL_HipX_joint",  -30*pi/180.),
            ("FL_HipY_joint",  -90*pi/180.),
            ("FL_Knee_joint",  -120*pi/180.),
            ("FR_HipX_joint",  -0*pi/180.),
            ("FR_HipY_joint",  -90*pi/180.),
            ("FR_Knee_joint",  -120*pi/180.),
            ("HL_HipX_joint",  -30*pi/180.),
            ("HL_HipY_joint",  -70*pi/180.),
            ("HL_Knee_joint",  -0*pi/180.),
            ("HR_HipX_joint",  -0*pi/180.),
            ("HR_HipY_joint",  -70*pi/180.),
            ("HR_Knee_joint",  -0*pi/180.),
        ])
        self.joint_limit_lower = np.asarray(list(self.joint_limit_lower_dict.values()))

        self.joint_limit_upper_dict = dict([
            ("FL_HipX_joint",  0*pi/180.),
            ("FL_HipY_joint",  70*pi/180.),
            ("FL_Knee_joint",  0*pi/180.),
            ("FR_HipX_joint",  30*pi/180.),
            ("FR_HipY_joint",  70*pi/180.),
            ("FR_Knee_joint",  0*pi/180.),
            ("HL_HipX_joint",  0*pi/180.),
            ("HL_HipY_joint",  90*pi/180.),
            ("HL_Knee_joint",  120*pi/180.),
            ("HR_HipX_joint",  30*pi/180.),
            ("HR_HipY_joint",  90*pi/180.),
            ("HR_Knee_joint",  120*pi/180.),
        ])

        self.joint_limit_upper = np.asarray(list(self.joint_limit_upper_dict.values()))


        self.action_space = Box(self.joint_limit_lower, self.joint_limit_upper)
        self.max_torque = motor_torq_limit
        self.joint_limit_torque_array = np.asarray([self.max_torque] * 12)
        self.joint_limit_torque = dict([
                        ("FL_HipX_joint",  self.max_torque),
                        ("FL_HipY_joint",  self.max_torque),
                        ("FL_Knee_joint",  self.max_torque),
                        ("FR_HipX_joint",  self.max_torque), 
                        ("FR_HipY_joint",  self.max_torque),
                        ("FR_Knee_joint",  self.max_torque), 
                        ("HL_HipX_joint",  self.max_torque),
                        ("HL_HipY_joint",  self.max_torque),
                        ("HL_Knee_joint",  self.max_torque),
                        ("HR_HipX_joint",  self.max_torque),
                        ("HR_HipY_joint",  self.max_torque),
                        ("HR_Knee_joint",  self.max_torque),
                    ])
        
        self.v_max = dict([
                        ("FL_HipX_joint",  max_vel),
                        ("FL_HipY_joint",  max_vel),
                        ("FL_Knee_joint",  max_vel),
                        ("FR_HipX_joint",  max_vel),  
                        ("FR_HipY_joint",  max_vel),
                        ("FR_Knee_joint",  max_vel),  
                        ("HL_HipX_joint",  max_vel),
                        ("HL_HipY_joint",  max_vel),
                        ("HL_Knee_joint",  max_vel),
                        ("HR_HipX_joint",  max_vel),
                        ("HR_HipY_joint",  max_vel),
                        ("HR_Knee_joint",  max_vel),
                    ])
        # Inner timestamp counter
        self.time = 0
        self.joint_names = ["FL_HipX_joint",
                            "FL_HipY_joint",
                            "FL_Knee_joint",
                            "FR_HipX_joint",
                            "FR_HipY_joint",
                            "FR_Knee_joint",
                            "HL_HipX_joint",
                            "HL_HipY_joint",
                            "HL_Knee_joint",
                            "HR_HipX_joint",
                            "HR_HipY_joint",
                            "HR_Knee_joint",]

        self.fall_poses = AnymalConfig(fallRecoverySet=0)
        # self.fall_poses = AnymalConfig(fallRecoverySet=1)

        self.key_pos_idx = 2

        if pos == None:
            self.init_pos = INIT_POSITION2 if fixed else INIT_POSITION
        else:
            self.init_pos = pos
            self.init_pos[2] = self.fall_poses.key_pose[self.key_pos_idx%len(self.fall_poses.key_pose)][0][2]
            init_orientation = self.fall_poses.key_pose[self.key_pos_idx%len(self.fall_poses.key_pose)][1]
        # Get Uid
        if fixed:
            self.uid = self._p.loadURDF(os.path.join(urdf_root, urdf), self.init_pos, init_orientation, useFixedBase=True)
        else:
            self.uid = self._p.loadURDF(os.path.join(urdf_root, urdf), self.init_pos, init_orientation)#, flags = self._p.URDF_USE_SELF_COLLISION)

        self.getJointInfo = self._p.getJointInfoPhysX if usePhysx else self._p.getJointInfo

        # Get joint number
        self.jNum = self._p.getNumJoints(self.uid)
        # Get joint infos
        self.jInfos = [self.getJointInfo(self.uid, n) for n in range(self.jNum)]

        # Get Uncontact Joints

        # self.unFootJoints = [joint[0] for joint in self.jInfos if joint[1].decode() not in list(self.feet.values())]


        # Get Unfixed Joints ids
        self.unFixJoints = [idx[0] for idx in self.jInfos \
                            if idx[2] == self._p.JOINT_PRISMATIC or idx[2] == self._p.JOINT_REVOLUTE]


        # for idx in self.unFixJoints:
        #     info = self.getJointInfo(self.uid, idx)
        #     MOTOR_DICT.update({info[1].decode('utf-8'): info[0]})
        #     REV_MOTOR_DICT.update({info[0]: info[1]})

        self.jointIds = []
        self.jointIdx = {}
        self.feetIdx = {"FL_foot_joint": 6,
                        "FR_foot_joint": 12,
                        "HL_foot_joint": 18,
                        "HR_foot_joint": 24 
                        }

        self.jointNameIdx = {}
        if usePhysx:
            for _i, idx in enumerate(self.unFixJoints):
                self._p.changeDynamics(self.uid, idx, linearDamping=0, angularDamping=0)
                info = self._p.getJointInfo(self.uid, idx)
                # print(info)
                jointName = self.joint_names[_i]
                # jointName = info[1].decode("utf-8")
                # print("Joint name: ", jointName)
                # jointType = info[2]
                # if (jointType == self._p.JOINT_PRISMATIC or jointType == JOINT_REVOLUTE):
                self.jointIds.append(idx)

                self.jointIdx.update({jointName: info[0]})
                self.jointNameIdx.update({info[0]: jointName})
        else:
            for j in range(self._p.getNumJoints(self.uid)):
                self._p.changeDynamics(self.uid, j, linearDamping=1, angularDamping=1)
                info = self._p.getJointInfo(self.uid, j)
                #print(info)
                jointName = info[1].decode("utf-8")
                # print(jointName)
                jointType = info[2]
                if (jointType == self._p.JOINT_PRISMATIC or jointType == self._p.JOINT_REVOLUTE):
                    self.jointIds.append(j)

                self.jointIdx.update({jointName: info[0]})
                self.jointNameIdx.update({info[0]: jointName})

        self.unFixJL = len(self.unFixJoints)

        self._observation = []
        self.sim_ts = 1 / sim_freq
        # List all Motors
        self._motors_ids = self.unFixJoints

        # Every ctrl_step, update ctrl info once time
        self.ctrl_step = int(np.ceil(self.sim_freq / self.ctrl_freq))
        # Loop cmd_step simulation times once apply action
        self.cmd_step  = int(np.ceil(self.sim_freq / self.cmd_freq))
        # self._motor_vel_limit = motor_vel_limit

        self._p.setPhysicsEngineParameter(fixedTimeStep=self.sim_ts)

        # Remember the init info
        # base info
        pos, ori = self.GetBasePos()
        vel, ang = self.GetBaseVel()

        # joints info
        jss = self.GetJointStates()
        # jPos = [x[0] for x in jss]
        jPos = [-0.0,     20*pi/180, -40*pi/180,
                -0.0,     20*pi/180, -40*pi/180,
                -0.0,    -20*pi/180,  40*pi/180,  
                -0.0,    -20*pi/180,  40*pi/180]
        jVel = [0 for x in jss]
        jTorq = [0 for x in jss]
        # print("Starting pos: ", jPos)
        self.InitInfo = {
            "Pos": pos,
            "Ori": ori,
            "Vel": vel,
            "Ang": ang,
            "JPos": jPos,
            "JVel": jVel,
            "JTorq": jTorq
        }
        self.InitMotor()

        self._actionDim = 12
        # Setup filter

        filter_order = 1 
        self.action_filter_method = FilterClass(self._actionDim)
        self.action_filter_method.butterworth(1./self.ctrl_freq, self.action_bandwidth, filter_order)  # sample period, cutoff frequency, order
    
        # Run collision filter
        # self.setCollisionFilter()
        self.reset()

    # def setCollisionFilter(self):
    #     # print(self.jointIdx.values())
    #     enableCollision = 0
    #     nonCollisionPairs0 = ['FL_HipX_joint', 'FR_HipX_joint', 'HL_HipX_joint', 'HR_HipX_joint', 
    #                             'FL_HipY_joint', 'FR_HipY_joint', 'HL_HipY_joint', 'HR_HipY_joint',
    #                             'LF_ADAPTER_JOINT','FL_foot_joint','RF_ADAPTER_JOINT','FR_foot_joint',
    #                             'LH_ADAPTER_JOINT','HL_foot_joint','RH_ADAPTER_JOINT','HR_foot_joint',]

    #     _jointIdx = [-1]
    #     # print(MOTOR_DICT.keys())
    #     for name in nonCollisionPairs0:
    #         _jointIdx.append(self.jointIdx[name])

    #     for idx0 in _jointIdx:
    #         for idx1 in _jointIdx:
    #             if idx0 == idx1:
    #                 continue
    #             self._p.setCollisionFilterPair(self.uid, self.uid, idx0, idx1, enableCollision)
    
    def GetObs(self):
        self._observation = self.GetJointStates()
        return self._observation

    def GetPosObs(self):
        obs = self.GetObs()
        return [s[0] for s in obs]

    def reset(self):

        seed=int((time.time()*1e6)%1e9)
        
        np.random.seed(seed=seed)

        self._time = 0
        if self.do_rsi:
            self.key_pos_idx += 1
        else:
            self.key_pos_idx = 0

        self.init_pos[2] = self.fall_poses.key_pose[self.key_pos_idx%len(self.fall_poses.key_pose)][0][2]+0.1
        init_orientation = self.fall_poses.key_pose[self.key_pos_idx%len(self.fall_poses.key_pose)][1]
        self._p.resetBasePositionAndOrientation(self.uid, self.init_pos, init_orientation)
        self._p.resetBaseVelocity(self.uid, [0, 0, 0], [0, 0, 0])
        self.ResetPos()
        self.InitInfo["Pos"] = self.init_pos
        self.InitInfo["Ori"] = init_orientation

    
    def ResetPos(self):
        """ Reset all lengths pose with init state

        :return: None
        """
        jpos = []
        for idx, jId in enumerate(self.jointIds):
            self._p.resetJointState(self.uid, jId,
                                    self.fall_poses.key_pose[self.key_pos_idx%len(self.fall_poses.key_pose)][2][idx],
                                    self.InitInfo["JVel"][idx])
            jpos.append(self.fall_poses.key_pose[self.key_pos_idx%len(self.fall_poses.key_pose)][2][idx])
        self.InitInfo["JPos"] = jpos
        # print("Reset q: ", jpos)

        self.joint_states    = self.GetJointStates(self._motors_ids)
        self.joint_pos       = [s[0] for s in self.joint_states]
        # print("Joint pos: ", self.joint_pos)
            # self._p.resetJointState(self.uid, idx,
            #                         self.InitInfo["JPos"][idx],
            #                         self.InitInfo["JVel"][idx])
                                    
            # self._p.resetJointState(self.uid, idx,# self.InitInfo['JPos'][idx], self.InitInfo['JVel'][idx])
            #                         self.InitInfo["JPos"][idx] + np.random.rand(),
            #                         self.InitInfo["JVel"][idx] + np.random.rand())

    def set_pos(self, act_cmd, clip = True):
        # print("Act comd:", act_cmd)
        if clip:
            act_cmd = np.clip(act_cmd, self.joint_limit_lower, self.joint_limit_upper)
        # print("Clipped comd:", act_cmd)

        self.SetMotorPos(act_cmd)

    def GetJointInfo(self, jIdx):
        return self.jInfos[jIdx]

    def GetJointInfos(self):
        return self.jInfos

    def GetJointState(self, jIdx):
        return self._p.GetJointState(self.uid, jIdx)

    def GetJointStates(self, ids = None):
        jIdxs = ids if ids != None else self.unFixJoints
        return self._p.getJointStates(self.uid, jIdxs)

    # def GetMotorAng(self):
    #     return np.asarray([s[0] for s in self.GetJointStates(self._motors_ids)])

    # def GetMotroVel(self):
    #     return np.asarray([s[1] for s in self.GetJointStates(self._motors_ids)])

    # def GetMotorTorq(self):
    #     return np.asarray([s[3] for s in self.GetJointStates(self._motors_ids)])

    def GetBasePos(self, returnEuler=False):
        pos, quat = self._p.getBasePositionAndOrientation(self.uid)
        if returnEuler:
            return (pos, self._p.getEulerFromQuaternion(quat))
        else:
            return (pos, quat)

    def GetBaseVel(self):
        return self._p.getBaseVelocity(self.uid)


    def CheckJointContackPoint(self, jIdx):
        if len(self.GetJContactPoint(jIdx)) >0 :
            return True
        else:
            return False

    # def CheckJointsContackGround(self):
    #     for jIdx in self.unFootJoints:
    #         if self.CheckJointContackPoint(jIdx):
    #             return True
    #     return False

    def CheckBaseContackGround(self):
        if len(self.GetJContactPoint(-1)) > 0:
            return True
        else:
            return False

    def GetJContactPoint(self, jIdx):
        # print(jIdx)

        # print("idx: ", self._p.getContactPoints(self.uid, self.planeId, jIdx, -1))
        return len(self._p.getContactPoints(self.uid, self.planeId, jIdx, -1)) > 0

    def GetJContactPoints(self):
        # print([jIdx for jIdx in self.feetIdx.values()])
        return [self.GetJContactPoint(jIdx) for jIdx in self.feetIdx.values()]


    def InitMotor(self):
        for mid in self._motors_ids:
            self._p.setJointMotorControl2(
                self.uid,
                mid,
                controlMode=self._p.POSITION_CONTROL,
                force=0
            )

    def SetMotorPos(self, pos):
        kp = [
            self._motor_hip_x_kp, self._motor_hip_y_kp, self._motor_knee_kp, 
            self._motor_hip_x_kp, self._motor_hip_y_kp, self._motor_knee_kp, 
            self._motor_hip_x_kp, self._motor_hip_y_kp, self._motor_knee_kp, 
            self._motor_hip_x_kp, self._motor_hip_y_kp, self._motor_knee_kp, 
        ]
        
        kd = [
            self._motor_hip_x_kd, self._motor_hip_y_kd, self._motor_knee_kd, 
            self._motor_hip_x_kd, self._motor_hip_y_kd, self._motor_knee_kd, 
            self._motor_hip_x_kd, self._motor_hip_y_kd, self._motor_knee_kd, 
            self._motor_hip_x_kd, self._motor_hip_y_kd, self._motor_knee_kd, 
        ]
        
        self._p.setJointMotorControlArray(
            self.uid,
            jointIndices    =   self._motors_ids,
            forces          =   self.joint_limit_torque_array,
            controlMode     =   self._p.POSITION_CONTROL,
            targetPositions =   pos,
            positionGains   =   kp,
            velocityGains   =   kd,
            # velocityGains=[0.005]*12,
        )

    # Torque control for real world. Is a bit buggy
    def set_torque(self, action):
        self.setTorqueControlwithVelocityConstrain(self.getTorqueDict(action))

    def getTorqueDict(self, action):
        torque_dict = {}
        joint_states = {}
        for n in range(self._actionDim):
            joint_states.update({self.jointNameIdx[self.jointIds[n]]: self._p.getJointState(self.uid, self.jointIds[n])})
            name = self.jointNameIdx[self.jointIds[n]]
            joint_state = joint_states[name]
            pos = joint_state[0]
            vel = joint_state[1]

            pos_ref = action[n]
            if "HipX" in name:
                kp = self._motor_hip_x_kp
                kd = self._motor_hip_x_kd
            elif "HipY" in name:
                kp = self._motor_hip_y_kp
                kd = self._motor_hip_y_kd
            elif "Knee" in name:
                kp = self._motor_knee_kp
                kd = self._motor_knee_kd
            else:
                assert 3 == 4, "Name: %s" % name
            
            P_u = kp * (pos_ref - pos)
            D_u = kd * (0-vel)

            control_torque = P_u+D_u
            # print("Name: %s" % name)
            # print("Pos ref: ", pos_ref)
            # print("Pos: ", pos) 
            # print("Vel: ", vel) 
            # print("KP: %.1f, kd: %.1f" %(kp, kd))
            # print("Torque:", control_torque)

            control_torque = np.clip(control_torque, -self.joint_limit_torque[name], self.joint_limit_torque[name])

            torque_dict.update({name: control_torque})
        return torque_dict

    def setTorqueControlwithVelocityConstrain(self, torque_dict): #set control
        # filtered_torque_dict = self.getFilteredTorque(torque_dict)
        filtered_torque_dict = torque_dict
        joint_indices = []
        target_velocities = []
        forces = []
        for jointName, torque in filtered_torque_dict.items():
            joint_indices.append(self.jointIdx[jointName])
            target_velocities.append(np.sign(torque) * self.v_max[jointName])
            
            # print("%s torque: %.1f" % (jointName, np.abs(torque)))
            forces.append(np.abs(torque))
            # print("Max vel:", self.v_max[jointName])

        self.old_torque_dict = torque_dict
        
        self._p.setJointMotorControlArray(
            bodyUniqueId=self.uid, jointIndices=joint_indices, targetVelocities=target_velocities,
            forces=forces,
            controlMode=self._p.VELOCITY_CONTROL,
        )

    def get_observation(self):
        # return the observations, but also read all relevant states for reward
        observation = []
        self.base_pos_vel, self.base_orn_vel = self.GetBaseVel()
        self.base_pos, self.base_quat = self.GetBasePos(returnEuler=False)
        self.base_orn = self._p.getEulerFromQuaternion(self.base_quat)

        self.joint_states    = self.GetJointStates(self._motors_ids)
        self.joint_pos       = [s[0] for s in self.joint_states]
        self.joint_vel       = [s[1] for s in self.joint_states]
        self.joint_torque    = [s[3] for s in self.joint_states] # not that idx 2 is jointReactionForces not jointTorques!
        self.feet_in_contact = self.GetJContactPoints() # doesn't work for physx
        # print("Joint toruqe: ", self.joint_torque)
        # print("Feet in contact: ", sum(self.feet_in_contact))
        # print("Joint pos: ", self.joint_pos)
        """Yaw adjusted base linear velocity"""

        Rz = rotZ(self.base_orn[2])
        self.Rz_i = np.linalg.inv(Rz)
        base_vel = np.array(self.base_pos_vel)
        base_vel.resize(1, 3)
        self.base_vel_yaw = np.transpose(self.Rz_i @ base_vel.transpose())  # base velocity in adjusted yaw frame
        self.base_vel_yaw[0][2] = 0.
        base_vel_yaw_list = list(copy.copy(self.base_vel_yaw[0][:2]))

        """Gravity"""
        invBasePos, invBaseQuat = self._p.invertTransform([0,0,0], self.base_quat) 
        gravity = np.array([0,0,-1]) # in world coordinates
        gravity_quat = self._p.getQuaternionFromEuler([0,0,0])
        #gravity vector in base frame
        gravityPosInBase, gravityQuatInBase = self._p.multiplyTransforms(invBasePos, invBaseQuat, gravity, gravity_quat)
        # print("Gravity: ", gravityPosInBase)
        self.base_gravity_vector = np.array(gravityPosInBase)

        """Append to observation"""
        observation.extend(copy.copy(base_vel_yaw_list))
        observation.extend(copy.copy(gravityPosInBase)) # gravityPosInBase is a list, self.base_gravity_vector is a numpy array
        observation.extend(copy.copy(self.joint_pos))

        observation = np.array(observation)

        return observation

    # Termination stuff

    def checkSelfContact(self):
        
        link = []
        for key,value in self.jointIdx.items():
            link.append(key)

        collision_list = list()
        collision_count = 0
        temp_link = list(link)
        for linkA in link:
            contact = len(self._p.getContactPoints(self.uid, self.uid, -1, self.jointIdx[linkA])) > 0
            if contact:
                collision_count += 1
                collision_list.append(linkA + ';' + 'base_link')
                # print(collision_list[-1])

            temp_link.remove(linkA)

            for linkB in temp_link:
                contact = len(self._p.getContactPoints(self.uid, self.uid, self.jointIdx[linkA], self.jointIdx[linkB])) > 0
                if contact:
                    collision_count += 1
                    collision_list.append(linkA + ';' + linkB)
                    # print(collision_list[-1])
        # print(collision_counter)
        return collision_count, collision_list

    def checkFall(self):
        fall = False
        contact_count, contact_info = self.checkSelfContact()
        if contact_count:
            if self.print_early_termination_reason:
                print("Fell because of self contact: ", contact_info)
            fall = True
            self.is_in_self_collision = True
        else:
            self.is_in_self_collision = False

        return fall

    # Rewards
    def reward(self):
        return self.recovery_reward()

    def recovery_reward(self):
        alpha = 1e-2

        base_x_vel_tar = 0.0
        base_y_vel_tar = 0.0
        base_z_vel_tar = 0.0

        # No penalization when x_vel_tar<COM_vel_yaw to tolerate higher velocity
        base_x_vel_err = np.maximum(base_x_vel_tar - self.base_vel_yaw[0][0], 0.0)
        base_y_vel_err = base_y_vel_tar - self.base_vel_yaw[0][1]
        base_z_vel_err = base_z_vel_tar - self.base_vel_yaw[0][2]

        base_x_vel_reward = math.exp(math.log(alpha)*(base_x_vel_err/1.0)**2)
        base_y_vel_reward = math.exp(math.log(alpha)*(base_y_vel_err/1.0)**2)
        base_z_vel_reward = math.exp(math.log(alpha)*(base_z_vel_err/1.0)**2)

        base_z_pos_tar = self.nominal_base_height
        base_z_pos_err = base_z_pos_tar-self.base_pos[2]
        base_z_pos_err = np.maximum(base_z_pos_err, 0.0) #Do not penalize when robot is higher than base height
        base_z_pos_reward = math.exp(math.log(alpha)*(base_z_pos_err/0.3)**2)
        # print("Noma: %.2f, is: %.2f" % (base_z_pos_tar, self.base_pos[2]))
        # gravity vector error 
        gravity_vector_tar = np.array([0,0,-1])
        base_gravity_vector_err = np.linalg.norm(gravity_vector_tar-self.base_gravity_vector)

        base_gravity_vector_reward = math.exp(math.log(alpha)*(base_gravity_vector_err/1.4)**2)

        joint_vel_err = 0
        joint_torque_err = 0
        for idx, jointId in enumerate(self.jointIds):
            joint_vel_err       += (self.joint_vel[idx] / self.v_max[self.jointNameIdx[jointId]]) ** 2
            joint_torque_err    += (self.joint_torque[idx] / self.joint_limit_torque[self.jointNameIdx[jointId]]) ** 2

        joint_vel_err       = joint_vel_err/len(self.jointIds)
        joint_torque_err    = joint_torque_err/len(self.jointIds)

        joint_vel_reward = math.exp(math.log(alpha)*joint_vel_err)
        joint_torque_reward = math.exp(math.log(alpha)*joint_torque_err)

        if base_gravity_vector_reward < 0.7:
            base_z_pos_reward = 0.
            base_x_vel_reward = 0.
            base_y_vel_reward = 0.
            base_z_vel_reward = 0.

        reward = (
                    1.*base_x_vel_reward \
                    + 1.*base_y_vel_reward \
                    + 1.*base_z_vel_reward \
                    + 5.*base_z_pos_reward \
                    + 10.*base_gravity_vector_reward \
                    + 2.*joint_vel_reward \
                    + 2.*joint_torque_reward \
                    ) \
                * 10. / (1.0+1.0+1.0+5.0+10.0+2.0+2.0)

        foot_contact_term   = 0
        fall_term           = 0
        self_contact_term   = 0
        success_term        = 0
        # print(sum(self.feet_in_contact))
        if sum(self.feet_in_contact): # any feet has contact
            foot_contact_term += 1.
            # print("YES", sum(self.feet_in_contact))
        if not self.is_in_self_collision:
            self_contact_term += 1.0

        reward += foot_contact_term
        reward += fall_term 
        reward += self_contact_term
        # reward currently adds to 12 in total (10 from task + self contact + ground contact)
        reward_term = []
        reward_term = dict([
            ("base_gravity_vector_reward", base_gravity_vector_reward),
            ("base_x_vel_reward", base_x_vel_reward),
            ("base_y_vel_reward", base_y_vel_reward),
            ("base_z_vel_reward", base_z_vel_reward),
            ("base_z_pos_reward", base_z_pos_reward),
            ("foot_contact_term", foot_contact_term),
            ("fall_term", fall_term),
            ("success_term", success_term),
        ])
        return reward/12., reward_term
