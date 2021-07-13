import gym, gym.spaces, gym.utils, gym.utils.seeding
import pybullet as p
import numpy as np

import matplotlib.pylab as plt

class b3pxBaseEnv(gym.Env):

	def __init__(self, robot, render = False):
		self.sence = None
		self.robot = robot
		self.isRender = render
		self.seed()
		self.camera = Camera()

		self._cam_dist = 3
		self._cam_yaw = 0
		self._cam_pitch = -30
		self._render_width = 320
		self._render_height = 240

		self.action_space = robot.action_space
		self.observation_space = robot.observation_space

		self.done = None
		self.reward  = 0

		self.robot.scene = self.sence

		s = self.robot.reset(self._p)

		return s

	def configure(self, args):
		self.args = args

	def seed(self, seed = None):
		self.np_random, seed = gym.utils.seeding.np_random(seed)
		self.robot.np_random = self.np_random

	def reset(self, ids):
		pass

	def render(self, model = 'laikago', close = False):
		pass

	def close(self):
		pass


class Camera:
	def __init__(self):
		pass

	def move_and_look_at(self, i, j, k, x, y, z):
		lookat = [x,y,z]
		distance = 10
		yaw = 10
		self._p.resetDebugVisualizerCamera(distance, yaw, -20, lookat)