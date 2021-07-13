"""This file implements the b3px_gym environment of anymal.

"""

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import math
import time
import gym
from gym.spaces import Box
from gym.utils import seeding
import numpy as np
import pybullet
from pybullet_envs.bullet import bullet_client
from b3px_gym.b3px_env.singleton import anymal
import os
import pybullet_data
import copy

import pkgutil

RENDER_HEIGHT = 960
RENDER_WIDTH = 1024
GRAVITY = [0, 0, -9.81]
THRES_HEIGHT_MIN = 0.3
THRES_HEIGHT_MAX = 0.8

# DefaultCfg = {
#     "backend"           : "bullet",
#     "gpu"               : False,
#     "gui"               : False,
#     "core"              : 1,
#     "solver"            : "pgs",
#     "enlarge"           : 1,
#     "sim_ts"            : 1000,
#     "motor_ts"          : 500,
#     "is_render"         : False,
#     "motor_vel_limit"   : np.inf,
#     "motor_kp"          : 400,
#     "motor_kd"          : 10,
#     "cam_dis"           : 1.0,
#     "cam_yaw"           : 0,
#     "cam_pitch"         : -30,
# }


# class AnymalB3PxEnvPl_1(gym.Env):
class AnymalB3PxEnvPl_1():
    """The b3px_gym environment for the anymal parallel version .

    It simulates the locomotion of a anymal, a quadruped robot. The state space
    include the angles, velocities and torques for all the motors and the action
    space is the desired motor angle for each motor. The reward function is based
    on how far the anymal walks in 1000 steps and penalizes the energy
    expenditure.

    """
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50
    }

    def __init__(self, args, time_to_stabilize=0.):
        gym.logger.set_level(40)
        
        self.confiugre(args)
        self.time_to_stabilize = time_to_stabilize

        self.sim_ts = args['sim_ts']
        self.control_ts = args['ctl_ts']
        self.motor_cmd_ts = args['cmd_ts']


        self.is_render = args['is_render']
        # self._motor_vel_limit = args['motor_vel_limit']
        self._motor_hip_x_kp = args['hip_x_kp']
        self._motor_hip_x_kd = args['hip_x_kd']
        self._motor_hip_y_kp = args['hip_y_kp']
        self._motor_hip_y_kd = args['hip_y_kd']
        self._motor_knee_kp  = args['knee_kp']
        self._motor_knee_kd  = args['knee_kd']
        self._cam_dist = args['cam_dist']
        self._cam_yaw  = args['cam_yaw']
        self._cam_pitch = args['cam_pitch']
        self.backend = args['backend']
        self.gpu    = args['gpu']
        self.gui    = args['gui']
        self.records_sim = args['records_sim']
        self.ts   = 0
        self.batch = args['batch']
        self.distance = args['distance']
        self._max_episode_steps = 125
        self.init_noise = args['init_noise']
        self.sim_time = 0
        self.observation_space = Box(high=9999, low=-9999, shape=(17, 1))

        self._p = None
        # print("Backend: %s" % self.backend)
        if self.backend == 'bullet':
            self._p = bullet_client.BulletClient(
                connection_mode = pybullet.GUI if self.gui else pybullet.DIRECT)
        elif self.backend == 'physx':
            options = "--numCores={}".format(args['core'])
            options += " --solver={}".format(args['solver'])
            if self.gpu :
                options += " --gpu=1"
                if args['enlarge'] != 1:
                    options += " --gmem_enlarge={}".format(args['enlarge'])

            self._p = bullet_client.BulletClient(pybullet.PhysX, options = options)


            if self.gui:
                self._p.loadPlugin("eglRendererPlugin")
            elif args['is_render']:
                egl = pkgutil.get_loader('eglRenderer')
                plugin = self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

        else:
            raise Exception("backend not support!")

        # self.physics_frequency = 1000
        # self.step_frequency = 25
        assert self.sim_ts%self.motor_cmd_ts==0
        self._p.setTimeStep(1. / self.sim_ts)
        # print("1. / self.sim_ts: ", 1. / self.sim_ts)
        self._p.setRealTimeSimulation(0)

        # self._p.setPhysicsEngineParameter(numSolverIterations = 10)
        # self._p.setPhysicsEngineParameter(minimumSolverIslandSize = 1024)
        # self._p.setPhysicsEngineParameter(contactBreakingThreshold = 0.1)

        planeId = self._p.loadURDF(os.path.join(args['urdf_root'], 'plane/plane100.urdf'))
        self._p.changeDynamics(planeId, -1, contactStiffness=1e6, contactDamping=1e3, lateralFriction=2)
        # info = self._p.getDynamicsInfo(planeId, -1)
        # print("PLane: ", info)
        u = np.floor(np.sqrt(self.batch)) + 1
        dist = self.distance
        offset = u * dist / 2

        self.BatchPos = [[i // u * dist - offset, i % u * dist - offset, 0.55] for i in range(self.batch)]

        # Batch Robots ID
        self.anymals = []
        self.uids = np.array(list(range(self.batch))) # 0 is plane
        for pos in self.BatchPos:
            uid = anymal.Anymal(self._p, 
                                       _motor_hip_x_kp = self._motor_hip_x_kp,
                                       _motor_hip_x_kd = self._motor_hip_x_kd,
                                       _motor_hip_y_kp = self._motor_hip_y_kp,
                                       _motor_hip_y_kd = self._motor_hip_y_kd,
                                       _motor_knee_kp  = self._motor_knee_kp,
                                       _motor_knee_kd  = self._motor_knee_kd,
                                       urdf_root= args['urdf_root'],
                                       urdf= 'anymal_bedi_urdf/anymal.urdf', 
                                       usePhysx= True if args['backend'] == 'physx' else False,
                                       fixed = args['fixed'],
                                       plane = planeId,
                                       pos = pos)


            self.anymals.append(uid)

        # Set Action Space Shape with Batch
        self.action_space = Box(self.anymals[0].joint_limit_lower, self.anymals[0].joint_limit_upper)
        self.action_space.shape = (self.batch, len(self.anymals[0].joint_limit_upper))

        self._env_step_counter = np.array([0]* self.batch)

        self._p.setGravity(0, 0, -9.81)

        # set camera direction
        # TODO : Add plugin reset camera size
        # self._p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0,0,0])

        self.reset()


    def confiugre(self, args):
        self.args = args


    def reset(self, uids = []):
        # print("Hello there, ", uids)
        if len(uids) == 0:
            uids = self.uids
        self.sim_time = 0
        # print("Hello there2. Batch: ", self.batch)

        for id in uids:
            # print("id: %d" % id)
            self.anymals[id].Reset()
            self._env_step_counter[id] = 0
        # print("Hello there3")

        if self.args['is_render']:
            self._p.resetDebugVisualizerCamera(self._cam_dist,
                                           self._cam_yaw,
                                           self._cam_pitch,
                                           self.BatchPos[id])
        # print("Obs: ", self.get_observations())
        return self.get_observations()


    def seed(self, seed = None):
        
        pass


    def step(self, actions, uids = []):
        """

        :param action: A list of desired motor angles for all motors ( 12 for anymal)
        :return:
            obs     : The angles, velocities and torques of all motors.
            reward  : The reward for the current state-action pair
            done    : Whether the episode has ended.
            info    : A dict that stores diagnostic information
        """


        self._env_step_counter += 1
        # print(self._env_step_counter, actions)
        if len(uids) == 0:
            uids = self.uids

        for uid in uids:
            # records.append(self.anymals[uid -1].Apply_Pos(actions[uid], self.records_sim))
            # if self.sim_time < self.time_to_stabilize:
            #     self.anymals[uid].Set_Pos(self.anymals[uid].InitInfo["JPos"])
            # else:
            self.anymals[uid].Set_Pos(actions[uid])
            # print("Actions: ", actions[uid])
            # self.anymals[uid].set_torque(actions[uid])
        # self.sim_ts = args['sim_ts']
        # self.control_ts = args['ctl_ts']
        # self.motor_cmd_ts = args['cmd_ts']
        # print("Control ts: ", self.control_ts)
        # print("Conmotor_cmd_tstrol ts: ",self.motor_cmd_ts)
        # print("sim_ts ts: ", self.sim_ts)
        for _ in range(int(np.ceil(self.control_ts/self.motor_cmd_ts))):
            # action = self.getFilteredAction(raw_action)
            # _ = self.getObservation() # filter is running in here. Thus call getObservation()
            # print("Outer: ", int(np.ceil(self.control_ts/self.motor_cmd_ts)))

            for _ in range(int(np.ceil(self.sim_ts/self.control_ts))):
                # print("Inner: ", int(np.ceil(self.sim_ts/self.control_ts)))
                self._p.stepSimulation()
                # pass
                self.sim_time += 1./self.sim_ts
                # print(self.anymals[0].get_observation()[-12:])

        # print("Sim time: %.3f" % self.sim_time)
        # print("==================================")
        obs = copy.deepcopy(self.get_observations())
        # get_observations() returns values and writes them to the variables. Need to call that before reward!
        done = copy.deepcopy(self._terminations(uids))
        
        reward = copy.deepcopy(self._rewards(uids))
        # if self.records_sim:
        #     return (records, reward, done, {})
        # else:
        return (obs, reward, done, {})

    def render(self, mode='rgb_array', close = False):

        if mode != "rgb_array":
            return np.array([])

        # base_pos = self.anymals.GetBasePos()[0]
        view_mtx = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0,0,0],
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)

        proj_mtx = self._p.computeProjectionMatrixFOV(
            fov = 60, aspect = float(RENDER_WIDTH)/ RENDER_HEIGHT,
            nearVal = 0.1, farVal = 100.0
        )

        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            viewMatrix=view_mtx,
            projectionMatrix=proj_mtx,
            # renderer = pybullet.ER_TINY_RENDERER
            # renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def get_motors_vel(self, uids = []):
        if len(uids) == 0:
            uids = self.uids

        MotorVels = []
        for uid in uids:
            MotorVels.append(self.anymals[uid].GetMotroVel())
        return np.asarray(MotorVels)

    def get_motors_troq(self, uids = []):
        if len(uids) == 0:
            uids = self.uids

        MotorTorqs = []
        for uid in uids:
            MotorTorqs.append(self.anymals[uid].GetMotroTorq())

        return np.asarray(MotorTorqs)

    def get_bases_ori(self, uids = []):
        if len(uids) == 0:
            uids = self.uids

        BasePoses = []

        for uid in uids:
            BasePoses.append(self.anymals[uid].GetBasePos())
        return np.asarray(BasePoses)

    def is_fallen(self, uids = []):
        if len(uids) == 0:
            uids = self.uids

        # print(self.anymal.GetBasePos())

        falls = np.array([False] * len(uids))

        for uid in uids:
            self.anymals[uid].checkFall()
        #     pos, ori = self.anymals[uid].GetBasePos()
        #     ori = np.abs(ori) * 180 / np.pi

        #     if pos[2] < THRES_HEIGHT_MIN or pos[2] > THRES_HEIGHT_MAX  \
        #         or self.anymals[uid].CheckJointsContackGround()     \
        #         or self.anymals[uid].CheckBaseContackGround()       \
        #         or np.any(ori > 50.) :
        #         falls[uid] = True
        #         # print("FELL======================")

        return falls

    def _terminations(self, uids = []):
        if len(uids) == 0:
            uids = self.uids

        return self.is_fallen(uids)

    def _rbf(self, v, mean = 0, dim = 1, p = -20):
        v -= mean
        return np.sum(np.exp(p * v * v))/dim

    def _rewards(self, uids = []):
        if len(uids) == 0:
            uids = self.uids

        Rewards = []
        for uid in self.uids:
            # base, _ = self.anymals[uid].GetBasePos()
            # vel, _ = self.anymals[uid].GetBaseVel()
            # Rewards.append(self._rbf(base[2], mean= 0.493, p = -100) + self._rbf(np.asarray(vel), dim=3, p = -100))

            reward, reward_info = self.anymals[uid].reward()
            # if self.sim_time < self.time_to_stabilize:
            #     reward = -99999
            Rewards.append(reward)

        return np.asarray(Rewards)

    def get_observations(self, uids = []):
        # For simple stand task, Only one task needed !
        if len(uids) == 0:
            uids = self.uids

        self._observation = []

        for uid in uids:
            # obs = []
            # _, bOri = self.anymals[uid].GetBasePos()
            # bVel, bAng = self.anymals[uid].GetBaseVel()
            # MotorPos = self.anymals[uid].GetPosObs()
            # obs.extend(bOri)
            # obs.extend(bVel)
            # obs.extend(bAng)
            
            # obs.extend(MotorPos)
            self._observation.append(copy.deepcopy(self.anymals[uid].get_observation()))
            
        self._observation = np.asarray(self._observation)
        return self._observation
