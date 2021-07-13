# """This file implements the b3px_gym environment of anymal.

# """

# import os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0,parentdir)

# import math
# import time
# import gym
# from gym.spaces import Box
# from gym.utils import seeding
# import numpy as np
# import pybullet
# from pybullet_envs.bullet import bullet_client
# import os
# import pybullet_data
# import copy

# import pkgutil

# RENDER_HEIGHT = 960
# RENDER_WIDTH = 1024
# GRAVITY = [0, 0, -9.81]

# # DefaultCfg = {
# #     "backend"           : "bullet",
# #     "gpu"               : False,
# #     "gui"               : False,
# #     "core"              : 1,
# #     "solver"            : "pgs",
# #     "enlarge"           : 1,
# #     "sim_ts"            : 1000,
# #     "motor_ts"          : 500,
# #     "is_render"         : False,
# #     "motor_vel_limit"   : np.inf,
# #     "motor_kp"          : 400,
# #     "motor_kd"          : 10,
# #     "cam_dis"           : 1.0,
# #     "cam_yaw"           : 0,
# #     "cam_pitch"         : -30,
# # }


# class ParallelEnv():
#     """The b3px_gym environment for the anymal parallel version .

#     It simulates the locomotion of a anymal, a quadruped robot. The state space
#     include the angles, velocities and torques for all the motors and the action
#     space is the desired motor angle for each motor. The reward function is based
#     on how far the anymal walks in 1000 steps and penalizes the energy
#     expenditure.

#     """
#     metadata = {
#         "render.modes": ["human", "rgb_array"],
#         "video.frames_per_second": 50
#     }

#     def __init__(self, args, robotClass, class_args, time_to_stabilize=0., is2D=False):
#         gym.logger.set_level(40)
        
#         self.robotClass = robotClass
#         self.configure(args)
#         self.time_to_stabilize = time_to_stabilize

#         self.sim_ts = args['sim_ts']
#         self.control_ts = args['ctl_ts']
#         self.motor_cmd_ts = args['cmd_ts']


#         self.is_render = args['is_render']
#         self._cam_dist = args['cam_dist']
#         self._cam_yaw  = args['cam_yaw']
#         self._cam_pitch = args['cam_pitch']
#         self.backend = args['backend']
#         self.gpu    = args['gpu']
#         self.gui    = args['gui']
#         self.records_sim = args['records_sim']
#         self.ts   = 0
#         self.batch = args['batch']
#         self.distance = args['distance']
#         self._max_episode_steps = args['_max_episode_steps']
#         self.init_noise = args['init_noise']
#         self.sim_time = 0
#         self.observation_space = Box(high=9999, low=-9999, shape=(17, 1))

#         self._p = None
#         # print("Backend: %s" % self.backend)
#         if self.backend == 'bullet':
#             self._p = bullet_client.BulletClient(
#                 connection_mode = pybullet.GUI if self.gui else pybullet.DIRECT)
#         elif self.backend == 'physx':
#             options = "--numCores={}".format(args['core'])
#             options += " --solver={}".format(args['solver'])
#             if self.gpu :
#                 options += " --gpu=1"
#                 if args['enlarge'] != 1:
#                     options += " --gmem_enlarge={}".format(args['enlarge'])

#             self._p = bullet_client.BulletClient(pybullet.PhysX, options = options)


#             if self.gui:
#                 self._p.loadPlugin("eglRendererPlugin")
#             elif args['is_render']:
#                 egl = pkgutil.get_loader('eglRenderer')
#                 plugin = self._p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

#         else:
#             raise Exception("backend not support!")

#         assert self.sim_ts%self.motor_cmd_ts==0
#         self._p.setTimeStep(1. / self.sim_ts)
#         self._p.setRealTimeSimulation(0)

#         # self._p.setPhysicsEngineParameter(numSolverIterations = 10)
#         # self._p.setPhysicsEngineParameter(minimumSolverIslandSize = 1024)
#         # self._p.setPhysicsEngineParameter(contactBreakingThreshold = 0.1)

#         planeId = self._p.loadURDF(os.path.join(args['urdf_root'], 'plane/plane100.urdf'))
#         self._p.changeDynamics(planeId, -1, contactStiffness=1e6, contactDamping=1e3, lateralFriction=2)

#         u = np.floor(np.sqrt(self.batch)) + 1
#         dist = self.distance
#         offset = u * dist / 2
#         if is2D:
#             self.BatchPos = [[0, i * dist - offset, 1.] for i in range(self.batch)]
#         else:
#             self.BatchPos = [[i // u * dist - offset, i % u * dist - offset, 0.55] for i in range(self.batch)]
#         print("Batch pos: ", self.BatchPos)
#         # Batch Robots ID
#         self.robots = []
#         self.uids = np.array(list(range(self.batch))) # 0 is plane
#         for pos in self.BatchPos:
#             print("class_args", class_args)
#             uid = robotClass(self._p,
#                             pos = pos,
#                             plane = planeId,
#                             **class_args)
#                             # urdf_root= args['urdf_root'],
#                             # urdf= 'anymal_bedi_urdf/anymal.urdf', 
#                             # usePhysx= True if args['backend'] == 'physx' else False,
#                             # fixed = args['fixed'],

#             self.robots.append(uid)

#         # Set Action Space Shape with Batch
#         self.action_space = Box(self.robots[0].action_space.low, self.robots[0].action_space.high)

        
#         self.action_space.shape = (self.batch, len(self.robots[0].action_space.high))

#         self._env_step_counter = np.array([0]* self.batch)

#         self._p.setGravity(0, 0, -9.81)

#         self.reset()


#     def configure(self, args):
#         self.args = args


#     def reset(self, uids = []):
#         if len(uids) == 0:
#             uids = self.uids
#         self.sim_time = 0

#         for id in uids:
#             self.robots[id].reset()
#             self._env_step_counter[id] = 0

#         if self.args['is_render']:
#             self._p.resetDebugVisualizerCamera(self._cam_dist,
#                                         self._cam_yaw,
#                                         self._cam_pitch,
#                                         self.BatchPos[id])
#         # print("Obs: ", self.get_observations())
#         return self.get_observations()


#     def seed(self, seed = None):
        
#         pass


#     def step(self, actions, uids = []):
#         """

#         :param action: A list of desired motor angles for all motors ( 12 for anymal)
#         :return:
#             obs     : The angles, velocities and torques of all motors.
#             reward  : The reward for the current state-action pair
#             done    : Whether the episode has ended.
#             info    : A dict that stores diagnostic information
#         """

#         self._env_step_counter += 1
#         # print(self._env_step_counter, actions)
#         if len(uids) == 0:
#             uids = self.uids

#         for uid in uids:
#             self.robots[uid].set_pos(actions[uid])
#         for _ in range(int(np.ceil(self.control_ts/self.motor_cmd_ts))):
#             for _ in range(int(np.ceil(self.sim_ts/self.control_ts))):
#                 self._p.stepSimulation()
#                 self.sim_time += 1./self.sim_ts

#         obs = copy.deepcopy(self.get_observations())
#         done = copy.deepcopy(self._terminations(uids))
#         reward = copy.deepcopy(self._rewards(uids))
#         return (obs, reward, done, {})

#     def render(self, mode='rgb_array', close = False):

#         if mode != "rgb_array":
#             return np.array([])

#         # base_pos = self.robots.GetBasePos()[0]
#         view_mtx = self._p.computeViewMatrixFromYawPitchRoll(
#             cameraTargetPosition=[0,0,0],
#             distance=self._cam_dist,
#             yaw=self._cam_yaw,
#             pitch=self._cam_pitch,
#             roll=0,
#             upAxisIndex=2)

#         proj_mtx = self._p.computeProjectionMatrixFOV(
#             fov = 60, aspect = float(RENDER_WIDTH)/ RENDER_HEIGHT,
#             nearVal = 0.1, farVal = 100.0
#         )

#         (_, _, px, _, _) = self._p.getCameraImage(
#             width=RENDER_WIDTH,
#             height=RENDER_HEIGHT,
#             viewMatrix=view_mtx,
#             projectionMatrix=proj_mtx,
#             # renderer = pybullet.ER_TINY_RENDERER
#             # renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
#         )
#         rgb_array = np.array(px)
#         rgb_array = rgb_array[:, :, :3]
#         return rgb_array

#     def get_motors_vel(self, uids = []):
#         if len(uids) == 0:
#             uids = self.uids

#         MotorVels = []
#         for uid in uids:
#             MotorVels.append(self.robots[uid].GetMotroVel())
#         return np.asarray(MotorVels)

#     def get_motors_troq(self, uids = []):
#         if len(uids) == 0:
#             uids = self.uids

#         MotorTorqs = []
#         for uid in uids:
#             MotorTorqs.append(self.robots[uid].GetMotroTorq())

#         return np.asarray(MotorTorqs)

#     def get_bases_ori(self, uids = []):
#         if len(uids) == 0:
#             uids = self.uids

#         BasePoses = []

#         for uid in uids:
#             BasePoses.append(self.robots[uid].GetBasePos())
#         return np.asarray(BasePoses)

#     def is_fallen(self, uids = []):
#         if len(uids) == 0:
#             uids = self.uids

#         # print(self.anymal.GetBasePos())

#         falls = np.array([False] * len(uids))

#         for uid in uids:
#             falls[uid] = self.robots[uid].checkFall()
#         #     pos, ori = self.robots[uid].GetBasePos()
#         #     ori = np.abs(ori) * 180 / np.pi

#         #     if pos[2] < THRES_HEIGHT_MIN or pos[2] > THRES_HEIGHT_MAX  \
#         #         or self.robots[uid].CheckJointsContackGround()     \
#         #         or self.robots[uid].CheckBaseContackGround()       \
#         #         or np.any(ori > 50.) :
#         #         falls[uid] = True
#         #         # print("FELL======================")

#         return falls

#     def _terminations(self, uids = []):
#         if len(uids) == 0:
#             uids = self.uids

#         return self.is_fallen(uids)

#     def _rbf(self, v, mean = 0, dim = 1, p = -20):
#         v -= mean
#         return np.sum(np.exp(p * v * v))/dim

#     def _rewards(self, uids = []):
#         if len(uids) == 0:
#             uids = self.uids

#         Rewards = []
#         for uid in self.uids:
#             # base, _ = self.robots[uid].GetBasePos()
#             # vel, _ = self.robots[uid].GetBaseVel()
#             # Rewards.append(self._rbf(base[2], mean= 0.493, p = -100) + self._rbf(np.asarray(vel), dim=3, p = -100))

#             reward, reward_info = self.robots[uid].reward()
#             # if self.sim_time < self.time_to_stabilize:
#             #     reward = -99999
#             Rewards.append(reward)

#         return np.asarray(Rewards)

#     def get_observations(self, uids = []):
#         # For simple stand task, Only one task needed !
#         if len(uids) == 0:
#             uids = self.uids

#         self._observation = []
#         for uid in uids:
#             # obs = []
#             # _, bOri = self.robots[uid].GetBasePos()
#             # bVel, bAng = self.robots[uid].GetBaseVel()
#             # MotorPos = self.robots[uid].GetPosObs()
#             # obs.extend(bOri)
#             # obs.extend(bVel)
#             # obs.extend(bAng)
            
#             # obs.extend(MotorPos)
#             self._observation.append(copy.deepcopy(self.robots[uid].get_observation()))
            
#         self._observation = np.asarray(self._observation)
#         return self._observation
