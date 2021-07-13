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
import os
import pybullet_data
import copy

import pkgutil

RENDER_HEIGHT = 960
RENDER_WIDTH = 1024
GRAVITY = [0, 0, -9.81]

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


class ParallelEnv():
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

    def __init__(self, args, robotClass, class_args, time_to_stabilize=0, is2D=False, spawn_height=1.5):
        gym.logger.set_level(40)
        
        self.robotClass = robotClass
        self.configure(args)
        self.time_to_stabilize = time_to_stabilize


        self.is_render = args['is_render']
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
        self._max_episode_steps = args['_max_episode_steps']
        self.init_noise = args['init_noise']
        self.sim_time = 0
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

        # assert self.sim_ts%self.motor_cmd_ts==0
        # self._p.setTimeStep(1. / self.sim_ts)
        # self._p.setRealTimeSimulation(0)

        # self._p.setPhysicsEngineParameter(numSolverIterations = 10)
        # self._p.setPhysicsEngineParameter(minimumSolverIslandSize = 1024)
        # self._p.setPhysicsEngineParameter(contactBreakingThreshold = 0.1)

        self.plane = self._p.loadURDF(os.path.join(args['urdf_root'], 'plane/plane100.urdf'))
        # print(self._p.getDynamicsInfo(planeId, -1))
        self._p.changeDynamics(self.plane, -1, lateralFriction=2)
        # print("HI")
        # robot_id = self._p.loadURDF("/home/kai/physx/bullet3/data/humanoid/humanoid_torso.urdf")
        # # print("HI2")
        # while True:

        #     pass

        u = np.floor(np.sqrt(self.batch)) + 1
        dist = self.distance
        offset = u * dist / 2
        # if is2D:


        # self.BatchPos = [[np.random.uniform(-10,15), i * dist - offset, spawn_height] for i in range(self.batch)]
        # else:
        self.BatchPos = [[i // u * dist - offset,5+ i % u * dist, spawn_height] for i in range(self.batch)]
        #     self.BatchPos = [[i // u * dist - offset, i % u * dist - offset, spawn_height] for i in range(self.batch)]
        # print("Batch pos: ", self.BatchPos)
        # Batch Robots ID
        self.robots = []
        self.uids = np.array(list(range(self.batch))) # 0 is plane
        for pos in self.BatchPos:

            # uid = robotClass(self._p, 
            #                 **class_args)
            uid = robotClass(self._p, pos=pos,planeId=self.plane ,
                            **class_args)
                            # urdf_root= args['urdf_root'],
                            # urdf= 'anymal_bedi_urdf/anymal.urdf', 
                            # usePhysx= True if args['backend'] == 'physx' else False,
                            # fixed = args['fixed'],

            self.robots.append(uid)

        self.timestep = uid.timestep
        self.frame_skip = uid.frame_skip
        # self.control_ts = args['ctl_ts']
        # self.motor_cmd_ts = args['cmd_ts']

        # Set Action Space Shape with Batch
        # self.action_space = Box(self.robots[0].action_space.low, self.robots[0].action_space.high)

        
        # self.action_space.shape = (self.batch, len(self.robots[0].action_space.high))

        self._env_step_counter = np.array([0]* self.batch)

        self._p.setGravity(0, 0, -9.81)

        self.reset()
        self.observation_space = self.robots[0].observation_space
        self.action_space = self.robots[0].action_space

    def configure(self, args):
        self.args = args


    def reset(self, uids = []):
        # self._p.resetSimulation()
        if len(uids) == 0:
            uids = self.uids
        self.sim_time = 0

        for id in uids:
            self.robots[id].reset()
            self._env_step_counter[id] = 0

        print("Timestep uid: %.4f" %self.timestep)
        print("Frameskip uid: %.4f" %self.frame_skip)
        print("Physics engine params parallel: ", self._p.getPhysicsEngineParameters())

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
        # print("Inside actions: ", actions)
        self._env_step_counter += 1
        # print(self._env_step_counter, actions)
        if len(uids) == 0:
            uids = self.uids

        # print("SIm time: %.4f/%.4f" % (self.sim_time, self.time_to_stabilize))
        if self.sim_time < self.time_to_stabilize:
            actions = [self.robots[0].q_nom[idx] for idx in self.robots[0].controlled_joints] * self.batch
        for uid in uids:
            # print("Raw: ", actions[uid])
            clipped_action = np.clip(actions[uid], self.action_space.low, self.action_space.high)
            # print("Clipped action: ", clipped_action)
         
            self.robots[uid].set_pos(clipped_action)
            
        # for _ in range(int(np.ceil(self.control_ts/self.motor_cmd_ts))):
        #     for _ in range(int(np.ceil(self.sim_ts/self.control_ts))):
        for _ in range(self.frame_skip):
            self._p.stepSimulation()
            self.sim_time += self.timestep

        # ground_contact5 = len(self._p.getContactPoints(self.robots[uid].robot.robot_model, self.plane, 4, -1)) > 0
        # ground_contact5 = len(self._p.getContactPoints(self.robots[uid].robot.robot_model, self.plane, 9, -1)) > 0
        # ground_contact5 = len(self._p.getContactPoints(self.robots[uid].robot.robot_model, self.plane, 14, -1)) > 0
        # ground_contact5 = len(self._p.getContactPoints(self.robots[uid].robot.robot_model, self.plane, 19, -1)) > 0

        obs = copy.deepcopy(self.get_observations())
        done = copy.deepcopy(self._terminations(uids))
        reward = copy.deepcopy(self._rewards(uids))
        return (obs, reward, done, {})

    def render(self, mode='rgb_array', close = False):

        if mode != "rgb_array":
            return np.array([])

        # base_pos = self.robots.GetBasePos()[0]
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
            MotorVels.append(self.robots[uid].GetMotroVel())
        return np.asarray(MotorVels)

    def get_motors_troq(self, uids = []):
        if len(uids) == 0:
            uids = self.uids

        MotorTorqs = []
        for uid in uids:
            MotorTorqs.append(self.robots[uid].GetMotroTorq())

        return np.asarray(MotorTorqs)

    def get_bases_ori(self, uids = []):
        if len(uids) == 0:
            uids = self.uids

        BasePoses = []

        for uid in uids:
            BasePoses.append(self.robots[uid].GetBasePos())
        return np.asarray(BasePoses)

    def is_fallen(self, uids = []):
        if len(uids) == 0:
            uids = self.uids

        # print(self.anymal.GetBasePos())

        falls = np.array([False] * len(uids))

        for uid in uids:
            falls[uid] = self.robots[uid].checkFall()
        #     pos, ori = self.robots[uid].GetBasePos()
        #     ori = np.abs(ori) * 180 / np.pi

        #     if pos[2] < THRES_HEIGHT_MIN or pos[2] > THRES_HEIGHT_MAX  \
        #         or self.robots[uid].CheckJointsContackGround()     \
        #         or self.robots[uid].CheckBaseContackGround()       \
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
            # base, _ = self.robots[uid].GetBasePos()
            # vel, _ = self.robots[uid].GetBaseVel()
            # Rewards.append(self._rbf(base[2], mean= 0.493, p = -100) + self._rbf(np.asarray(vel), dim=3, p = -100))

            reward, reward_info = self.robots[uid].getReward()
            if self.sim_time < self.time_to_stabilize:
                reward = -99999
            Rewards.append(reward)

        return np.asarray(Rewards)

    def get_observations(self, uids = []):
        # For simple stand task, Only one task needed !
        if len(uids) == 0:
            uids = self.uids

        self._observation = []
        for uid in uids:
            # obs = []
            # _, bOri = self.robots[uid].GetBasePos()
            # bVel, bAng = self.robots[uid].GetBaseVel()
            # MotorPos = self.robots[uid].GetPosObs()
            # obs.extend(bOri)
            # obs.extend(bVel)
            # obs.extend(bAng)
            
            # obs.extend(MotorPos)
            self._observation.append(copy.deepcopy(self.robots[uid].get_observation()))
            
        self._observation = np.asarray(self._observation)
        return self._observation
