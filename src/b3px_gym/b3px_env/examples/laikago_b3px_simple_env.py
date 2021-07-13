import os
import inspect
import copy
import imageio
import datetime
from matplotlib import pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import numpy as np
from b3px_env.singleton import laikago_gym_env

DefaultCfg = {
    "backend"           : "bullet",
    "gpu"               : False,
    "gui"               : False,
    "core"              : 1,
    "solver"            : "pgs",
    "enlarge"           : 1,
    "sim_ts"            : 1000,
    "ctl_ts"            : 500,
	"cmd_ts"            : 25,
    "is_render"         : False,
	"video_name"        : "laikago_{}.mp4",
    "motor_vel_limit"   : np.inf,
    "motor_kp"          : 400,
    "motor_kd"          : 10,
    "cam_dist"          : 2.0,
    "cam_yaw"           : 52,
    "cam_pitch"         : -30,
	"records_sim"       : False,
	"fixed"             : False,
	# "urdf_root"         : "/home/syslot/DevSpace/WALLE/src/pybullet_demo/urdf"
    "urdf_root"			: "../../../urdf",

	# After args are used for Parallel Environment
	"batch"             : 5,
	"distance"          : 5,
	"init_noise"        : False
}

jointName = [
			'FL_hip_motor_2_chassis_joint',
            'FL_upper_leg_2_hip_motor_joint',
            'FL_lower_leg_2_upper_leg_joint',
            'FR_hip_motor_2_chassis_joint',
            'FR_upper_leg_2_hip_motor_joint',
            'FR_lower_leg_2_upper_leg_joint',
            'RL_hip_motor_2_chassis_joint',
            'RL_upper_leg_2_hip_motor_joint',
            'RL_lower_leg_2_upper_leg_joint',
            'RR_hip_motor_2_chassis_joint',
            'RR_upper_leg_2_hip_motor_joint',
            'RR_lower_leg_2_upper_leg_joint'
]
jointName = [
	'hip',
	'upper',
	'lower'
]

title = [
	"FL_LEG", "FR_LEG", "RL_LEG" , "RR_LEG"
]

def StandExample(cfg):
	"""An example of laikago standing and squatting on the floor.

	To validate the accurate motor model we command the robot and sit and stand up
	periodically in both simulation and experiment. We compare the measured motor
	trajectories, torques and gains.
	"""



	environment = laikago_gym_env.LaikagoB3PxEnv(cfg)
	steps = int(25 * 3000)

	actions_and_observations = []

	imgs = []
	for step_counter in range(steps):
		# Matches the internal timestep.
		time_step = cfg['cmd_ts'] / cfg['sim_ts']
		t = step_counter * time_step
		current_row = []
		current_row.append(t)

		action = [-0.035, -0.05, -0.5] * 4
		# action = [0, 0, 0] * 4
		current_row.append(action)
		# current_row.extend(action)

		if cfg['is_render']:
			img = environment.render()
			imgs.append(img)

		observation, _, done, _ = environment.step(action)
		current_row.append(observation)
		# current_row.extend(observation)
		actions_and_observations.append(current_row)
		# if done:
		# 	print("fall down, reset")
		#
		# 	break
			# environment.reset()

	if cfg['is_render']:
		imageio.mimwrite(cfg['video_name'].format(datetime.datetime.now()), np.asarray(imgs), fps = 50.0)
	environment.reset()
	return actions_and_observations


def PosPlot():

	cfg = copy.deepcopy(DefaultCfg)

	# cfg['gui'] = True
	# cfg['backend'] = 'physx'
	cfg['motor_kp'] = 10
	cfg['motor_kd'] = 0.1
	# cfg['gpu'] = True
	# cfg['fixed'] = True
	cfg['solver']='tgs'
	cfg['records_sim'] = True
	# cfg['is_render'] =True

	acts_and_obs = StandExample(cfg)

	if cfg['records_sim']:
		# t = np.asarray([row[0] for row in acts_and_obs])
		actions = np.asarray([row[1] for _ in range(40) for row in acts_and_obs])
		pos = []
		for row in acts_and_obs:
			for p in row[2]:
				pos.append(p)
		pos = np.asarray(pos)
		t = np.arange(len(actions))
	else:
		# extract t
		t = np.asarray([row[0] for row in acts_and_obs])

		# extract actions

		actions = np.asarray([row[1:13] for row in acts_and_obs]) /np.pi * 180.

		# extract position
		pos = np.asarray([row[13:] for row in acts_and_obs]) / np.pi* 180.


	plt.figure(figsize=(15, 20))
	for i in range(12):
		if i % 3 == 0:
			plt.subplot(12,1, i//3 + 1)
			plt.title(title[i%3])
		plt.plot(t, pos[:, i], label = jointName[i%3])
		# plt.plot(t, actions[:, i], label = jointName[i%3] + "_ref")

	plt.legend()
	plt.show()


	# print(acts_and_obs)


def PosPlotSingle():


	cfg = copy.deepcopy(DefaultCfg)

	cfg['gui'] = True
	cfg['backend'] = 'physx'
	cfg['motor_kp'] = 400 #10000 * 180 / np.pi
	cfg['motor_kd'] = 10
	cfg['gpu'] = True
	# cfg['fixed'] = True
	cfg['solver'] = 'tgs'
	cfg['records_sim'] = True
	# cfg['ctl_ts'] = 1000
	# cfg['cmd_ts'] = 1000
	# cfg['is_render'] =True

	acts_and_obs = StandExample(cfg)

	if cfg['records_sim']:
		# t = np.asarray([row[0] for row in acts_and_obs])
		actions = np.asarray([row[1] for _ in range(40) for row in acts_and_obs]) /np.pi * 180.
		pos = []
		for row in acts_and_obs:
			for p in row[2]:
				pos.append(p)
		pos = np.asarray(pos) /np.pi * 180.
		t = np.arange(len(actions))
	else:
		# extract t
		t = np.asarray([row[0] for row in acts_and_obs])

		# extract actions

		actions = np.asarray([row[1:13] for row in acts_and_obs]) / np.pi * 180.

		# extract position
		pos = np.asarray([row[13:] for row in acts_and_obs]) / np.pi * 180.


	plt.figure(figsize=(40, 25))
	# plt.plot(t, pos[:,1], linewidth = .5)
	max = 3
	for i in range(max):
		plt.subplot(max,1, i+1)
		# plt.title(title[i%3])
		plt.plot(t, pos[:, i], linewidth=1)
		plt.plot(t, actions[:, i], label = jointName[i%3] + "_ref")

	plt.legend()
	plt.show()




if __name__  == '__main__':
	# StandExample()

	PosPlotSingle()
	# PosPlot()
