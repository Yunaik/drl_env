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
from b3px_gym.b3px_env.parallel.filter_array import FilterClass
import pkgutil

RENDER_HEIGHT = 1080
RENDER_WIDTH = 1920
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

    def __init__(self, args, robotClass, class_args, time_to_stabilize=0., is2D=False, spawn_height=1.5, isMujocoEnv=True, action_bandwidth=1, filter_action=True, force_duration=0.0, spawn_side_by_side=False):
        gym.logger.set_level(40)
        self.robotClass = robotClass
        self.configure(args)
        self.time_to_stabilize = time_to_stabilize
        self.started_to_perform_action = False 

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
        # self._max_episode_steps = args['_max_episode_steps']
        self.init_noise = args['init_noise']
        self.sim_time = 0
        self.isMujocoEnv = isMujocoEnv
        self._p = None
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


        """Disturbance stuff"""
        self.random_disturbance_time    = True
        self.random_force_magnitude     = True
        self.probability_of_push = 1.
        self.force_duration = force_duration
        self.maxImpulse = 100 # 240 in paper
        if self.random_disturbance_time:
            time1 = np.random.uniform(self.time_to_stabilize+ 2.5, self.time_to_stabilize+4.5) if np.random.random() < self.probability_of_push else 10000
            time2 = np.random.uniform(self.time_to_stabilize+ 5.0, self.time_to_stabilize+7.0) if np.random.random() < self.probability_of_push else 10000
            time3 = np.random.uniform(self.time_to_stabilize+ 7.5, self.time_to_stabilize+9.5) if np.random.random() < self.probability_of_push else 10000
        else:
            time1 = self.time_to_stabilize+2.5 if np.random.random() < self.probability_of_push else 10000
            time2 = self.time_to_stabilize+5.0 if np.random.random() < self.probability_of_push else 10000
            time3 = self.time_to_stabilize+7.5 if np.random.random() < self.probability_of_push else 10000

        force_durations = [self.force_duration for i in range(3)]


        if self.force_duration > 0.:
            force_magnitude_disturbance_x = []


            for duration in force_durations:
                if self.random_force_magnitude:
                    total_impulse_magnitude = np.random.uniform(-self.maxImpulse, self.maxImpulse)
                else:
                    total_impulse_magnitude = -self.maxImpulse
                x_component = total_impulse_magnitude

                x_component /= duration # impulse to force

                force_magnitude_disturbance_x.append(x_component)
        else:
            force_magnitude_disturbance_x = [0, 0, 0]

        self.disturbance_time = [
                                [time1, time1+force_durations[0], force_magnitude_disturbance_x[0]], 
                                [time2, time2+force_durations[1], force_magnitude_disturbance_x[1]], 
                                [time3, time3+force_durations[2], force_magnitude_disturbance_x[2]], 
                                ]
        # print("DISTURBANCE TIME: ", self.disturbance_time)

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
        if spawn_side_by_side:
            self.BatchPos = [[0, i * dist - offset, spawn_height] for i in range(self.batch)]
        else:
            # self.BatchPos = [[i // u * dist - offset, i % u * dist - offset, spawn_height] for i in range(self.batch)]

            self.BatchPos = [[i // u * dist - offset, i % u * dist - offset, spawn_height] for i in range(self.batch)]
        # print("Batch pos: ", self.BatchPos)
        # Batch Robots ID
        self.robots = []
        self.uids = np.array(list(range(self.batch))) # 0 is plane
        for pos in self.BatchPos:
            if len(self.BatchPos) == 1:
                pos[1] = 0
            # uid = robotClass(self._p, 
            #                 **class_ar    gs)
            if isMujocoEnv:
                uid = robotClass(self._p, pos=pos, 
                            **class_args)
            else:
                uid = robotClass(self._p, pos=pos, planeId=self.plane,
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

        try:
            self.observation_space = self.robots[0].robot.observation_space
            self.action_space = self.robots[0].robot.action_space
        except:
            self.observation_space = self.robots[0].observation_space
            self.action_space = self.robots[0].action_space

        # Filter
        self.action_bandwidth = action_bandwidth
        filter_order = 1 
        self.action_filter_methods = []
        for idx in range(self.batch):
            self.action_filter_methods.append(FilterClass(len(self.action_space.low)))
            self.action_filter_methods[idx].butterworth(self.timestep, self.action_bandwidth, filter_order)  # sample period, cutoff frequency, order
        self.filter_action = filter_action 
        if isMujocoEnv:
            self.filter_action = False
        if self.filter_action:
            for _ in range(1000):
                for uid in self.uids:
                    try: # valkyrie
                        filtered_action = self.getFilteredAction([self.robots[0].q_nom[idx] for idx in self.robots[0].controlled_joints], uid)
                        # filtered_action = self.getFilteredAction([0 for idx in self.robots[0].controlled_joints], uid)
                    except: # ANYmal
                        filtered_action = self.getFilteredAction([self.robots[0].joint_config_start[idx] for idx in range(self.robots[0]._actionDim)], uid)


    def getFilteredAction(self, action, idx):
        if self.filter_action:
            # self.unfiltered_action = copy.copy(action)
            filtered_values = self.action_filter_methods[idx].applyFilter(copy.copy(action))
            self.filtered_action = filtered_values
            return self.filtered_action
        else: 
            return action

    def configure(self, args):
        self.args = args


    def reset(self, uids = []):
        self.seed()
        self.started_to_perform_action = False 
        # self._p.resetSimulation()
        if len(uids) == 0:
            uids = self.uids
        self.sim_time = 0

        for id in uids:

            self.robots[id].reset()
            self._env_step_counter[id] = 0

        # print("Timestep uid: %.4f" %self.timestep)
        # print("Frameskip uid: %.4f" %self.frame_skip)
        # print("Physics engine params parallel: ", self._p.getPhysicsEngineParameters())

        if self.args['is_render']:
            self._p.resetDebugVisualizerCamera(self._cam_dist,
                                        self._cam_yaw,
                                        self._cam_pitch,
                                        self.BatchPos[id])
        # print("Obs: ", self.get_observations())
        if self.random_disturbance_time: # best way is to have one random disturbance force per robot 
            time1 = np.random.uniform(self.time_to_stabilize+ 2.5, self.time_to_stabilize+4.5) if np.random.random() < self.probability_of_push else 10000
            time2 = np.random.uniform(self.time_to_stabilize+ 5.0, self.time_to_stabilize+7.0) if np.random.random() < self.probability_of_push else 10000
            time3 = np.random.uniform(self.time_to_stabilize+ 7.5, self.time_to_stabilize+9.5) if np.random.random() < self.probability_of_push else 10000
        else:
            time1 = self.time_to_stabilize+2.5 if np.random.random() < self.probability_of_push else 10000
            time2 = self.time_to_stabilize+5.0 if np.random.random() < self.probability_of_push else 10000
            time3 = self.time_to_stabilize+7.5 if np.random.random() < self.probability_of_push else 10000

        force_durations = [self.force_duration for i in range(3)]

        if self.force_duration > 0.:
            force_magnitude_disturbance_x = []


            for duration in force_durations:
                if self.random_force_magnitude:
                    total_impulse_magnitude = np.random.uniform(-self.maxImpulse, self.maxImpulse)
                else:
                    total_impulse_magnitude = -self.maxImpulse
                x_component = total_impulse_magnitude

                x_component /= duration # impulse to force

                force_magnitude_disturbance_x.append(x_component)
        else:
            force_magnitude_disturbance_x = [0, 0, 0]
            
        self.disturbance_time = [
                                [time1, time1+force_durations[0], force_magnitude_disturbance_x[0]], 
                                [time2, time2+force_durations[1], force_magnitude_disturbance_x[1]], 
                                [time3, time3+force_durations[2], force_magnitude_disturbance_x[2]], 
                                ]
        # print("Reset with disturbance: ", self.disturbance_time)
        return self.get_observations()


    def seed(self, seed = None):

        seed=int((time.time()*1e6)%1e9) if seed is None else seed
        np.random.seed(seed=seed)


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

        if self.sim_time < self.time_to_stabilize:
            try:
                actions =[ [self.robots[0].q_nom[idx] for idx in self.robots[0].controlled_joints] ]* self.batch
                # actions =[ [0]*len(self.robots[0].controlled_joints) ]* self.batch
            except:
                actions = [[self.robots[0].joint_config_start[idx] for idx in range(self.robots[0]._actionDim)]] * self.batch
            # print("ACTION: ", actions)
        else:
            self.started_to_perform_action = True
            

        # for _ in range(int(np.ceil(self.control_ts/self.motor_cmd_ts))):
        #     for _ in range(int(np.ceil(self.sim_ts/self.control_ts))):
        raw_actions = actions
        for _ in range(self.frame_skip):
            for uid in uids:
                # print("HI")
                # print("Action: ", raw_actions[uid])
                # print("High: ", self.action_space.high)
                # print("Low: ", self.action_space.low)
                if not self.isMujocoEnv:
                    assert (np.array(raw_actions[uid])<=self.action_space.high).all() and (np.array(raw_actions[uid])>=self.action_space.low).all(),  "Outside of action space assertion error"
                filtered_action = self.getFilteredAction(raw_actions[uid], uid)
                clipped_action = np.clip(filtered_action, self.action_space.low, self.action_space.high)
                # print("Filtered action: ", clipped_action)
                if self.isMujocoEnv:
                    self.robots[uid].robot.apply_action(clipped_action)
                else:
                    self.robots[uid].set_pos(clipped_action)

                """Disturbance"""
                for dist_time in self.disturbance_time:
                    if (self.sim_time >= dist_time[0]) and (self.sim_time <= dist_time[1]):
                        t = self.sim_time-dist_time[0]
                        a = -4/(self.force_duration**2)
                        b = 4/self.force_duration
                        force = lambda t: (a*t**2+b*t)*dist_time[2]
                        # print("Time: %.3fs. Dist time: . Force: %.2f" % (self.sim_time, force(t)), dist_time)
                        force_vector = [force(t), 0,0]
                        self.robots[uid].applyForce(force_vector)

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

    def render(self, mode='rgb_array', close = False, distance=5, yaw=52, pitch=-30, roll=0):
        if mode != "rgb_array":
            return np.array([])

        # base_pos = self.robots.GetBasePos()[0]

        view_mtx = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0,0,0],
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
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
            # else:
                # print("reward: ", reward)
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
