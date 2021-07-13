"""This file implements the b3px_gym environment of laikago.

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
from . import laikago
import os
import pybullet_data

import pkgutil

RENDER_HEIGHT = 720
RENDER_WIDTH = 960
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


class LaikagoB3PxEnv(gym.Env):
    """The b3px_gym environment for the laikago.

    It simulates the locomotion of a laikago, a quadruped robot. The state space
    include the angles, velocities and torques for all the motors and the action
    space is the desired motor angle for each motor. The reward function is based
    on how far the laikago walks in 1000 steps and penalizes the energy
    expenditure.

    """
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50
    }

    def __init__(self, args):
        self.confiugre(args)

        self.sim_ts = args['sim_ts']
        self.control_ts = args['ctl_ts']
        self.motor_cmd_ts = args['cmd_ts']
        self.is_render = args['is_render']
        self._motor_vel_limit = args['motor_vel_limit']
        self._motor_kp = args['motor_kp']
        self._motor_kd = args['motor_kd']
        self._cam_dist = args['cam_dist']
        self._cam_yaw  = args['cam_yaw']
        self._cam_pitch = args['cam_pitch']
        self.backend = args['backend']
        self.gpu    = args['gpu']
        self.gui    = args['gui']
        self.records_sim = args['records_sim']
        self.ts   = 0
        self._max_episode_steps = 125

        self.observation_space = Box(high = np.pi, low = - np.pi, shape = (21, 1))



        self._p = None

        if self.backend == 'bullet':
            self._p = bullet_client.BulletClient(
                connection_mode = pybullet.GUI if self.gui else pybullet.DIRECT)
        elif self.backend == 'physx':
            options = " --numCores={}".format(args['core'])
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


        self._p.setPhysicsEngineParameter(numSolverIterations = 5)
        self._p.setPhysicsEngineParameter(minimumSolverIslandSize = 1024)
        # self._p.setPhysicsEngineParameter(contactBreakingThreshold = 0.1)

        planeId = self._p.loadURDF(os.path.join(args['urdf_root'], 'plane/plane100.urdf'), useMaximalCoordinates=True)
        self.laikago = laikago.Laikago(self._p, True,
                                       kp = self._motor_kp,
                                       kd = self._motor_kd,
                                       urdf_root= args['urdf_root'],
                                       # urdf= 'laikago_description/laikago_foot.urdf', # #'laikago/laikago.urdf', #
                                       urdf= 'laikago_description/laikago_gpu.urdf',
                                       usePhysx= True if args['backend'] == 'physx' else False,
                                       fixed = args['fixed'],
                                       plane = planeId
                                       )


        self.action_space = Box(laikago.JOINT_LIMIT_LOWER, laikago.JOINT_LIMIT_UPPER)
        # self.action_space.shape = (12, 1)
        self._p.setGravity(0, 0, -9.81)

        self.reset()


    def confiugre(self, args):
        self.args = args


    def reset(self):
        self.laikago.Reset()
        self._env_step_counter = 0
        self._last_base_position = [0,0,0]
        if self.args['is_render']:
            self._p.resetDebugVisualizerCamera(self._cam_dist,
                                           self._cam_yaw,
                                           self._cam_pitch,
                                           [0,0,0])

        return self.get_observation()


    def seed(self, seed = None):
        pass


    def step(self, action):
        """

        :param action: A list of desired motor angles for all motors ( 12 for laikago)
        :return:
            obs     : The angles, velocities and torques of all motors.
            reward  : The reward for the current state-action pair
            done    : Whether the episode has ended.
            info    : A dict that stores diagnostic information
        """

        self._env_step_counter += 1
        # records = self.laikago.Apply(action, self.records_sim)
        records = self.laikago.Apply_Pos(action, record_joints = self.records_sim)

        reward = self._reward()
        done = self._termination()

        self.last_obs = self.get_observation()

        if self.records_sim:
            return (records, reward, done, {})
        else:
            return (self.last_obs, reward, done, {})

    def render(self, mode='rgb_array', close = False):

        if mode != "rgb_array":
            return np.array([])

        base_pos = self.laikago.GetBasePos()[0]
        view_mtx = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
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

    def get_motor_vel(self):
        return self.laikago.GetMotroVel()

    def get_motor_troq(self):
        return self.laikago.GetMotroTorq()

    def get_base_ori(self):
        return self.laikago.GetBasePos()

    def is_fallen(self):
        # print(self.laikago.GetBasePos())
        fall = False
        pos, ori = self.laikago.GetBasePos()

        ori = np.abs(ori)

        # Base Height too low
        if pos[2] < THRES_HEIGHT_MIN or pos[2] > THRES_HEIGHT_MAX:
            fall = True
            return fall

        # Check Joint Contact with Ground
        if self.laikago.CheckJointsContackGround():
            fall = True
            return fall

        # Check Base Contact with Ground
        if self.laikago.CheckBaseContackGround():
            fall = True
            return fall

        # Check Base Rotation

        if ori[0] > 50 * np.pi / 180 or ori[1] > 50 * np.pi / 180 or ori[2] > 50 * np.pi:
            fall = True
            return fall

        # Check Too Far away

        # o_pos = self.laikago.InitInfo['Pos']
        # if np.square(o_pos[0] - pos[0]) + np.square(o_pos[1] - o_pos[1]) > 1:
        #     fall = True
        #     return fall

        return fall


    def _termination(self):
        # calulate distence
        # print(self._env_step_counter)
        return self.is_fallen() #or self._max_episode_steps == self._env_step_counter


    def _rbf(self, v, mean = 0, dim = 1, p = -20):
        v -= mean
        return np.sum(np.exp(p * v * v))/dim

    def _reward(self):
        base, _ = self.laikago.GetBasePos()
        vel, _ = self.laikago.GetBaseVel()
        return  self._rbf(base[2], mean = 0.493, p = -100) + self._rbf(np.asarray(vel), dim = 3, p = -100) #+ self._rbf(ori, dim = len(ori), p = -100) # 10 degree inner

    def get_observation(self):
        # For simple stand task, Only one task needed !
        self._observation = []
        # Not use bPos
        _, bOri = self.laikago.GetBasePos()
        bVel, bAng = self.laikago.GetBaseVel()
        MotorPos = self.laikago.GetPosObs()
        # self._observation.extend(bPos)
        self._observation.extend(bOri)
        self._observation.extend(bVel)
        self._observation.extend(bAng)
        self._observation.extend(MotorPos)
        # TODO : add foot contact Info
        self._observation = np.asarray(self._observation)
        return self._observation

    def close(self):
        self._p.disconnect()

