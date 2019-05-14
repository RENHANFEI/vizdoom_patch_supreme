import os
import random
from collections import Counter

import cv2
import numpy as np
from numpy import pi
from matplotlib import pyplot as plt

from src.target_calculator import TargetCalculator

class PatchBot(object):
	"""docstring for PatchBot"""
	def __init__(self, resolution=(320,240), memory=5):
		super(PatchBot, self).__init__()
		self.resolution = resolution
		self.memory = 5
		self.view_point = np.array([resolution[1], resolution[0] / 2])
		self.actions = []
		self.diff = 0.05
		self.center_diff_ratio = 1.2
		# TURN_LEFT, TURN_RIGHT, MOVE_FORWARD
#        self.action_dict = {'left':[1, 0, 0], 'right':[0, 1, 0], 'forward':
#            [0, 0, 1], 'left_forward':[1, 0, 1], 'right_forward':[0, 1, 1]}
		self.action_dict = {'left': 0, 'right': 1, 'forward': 2, 'left_forward': 3, 'right_forward': 4}
		self.target_calculator = TargetCalculator()
		self.right_weight = 0
		self.counter = Counter()
		self.save_dir = 'bots/15'
		self.suffix = '.jpg'
		self.idx = 0

	def update(self, frame, count=False, record=False):
		if self.actions:
			return self.action_dict[self.actions.pop()]

		target, size = self.target_calculator.get_target_nearest_test(frame)

		if count:
			self.counter.update([size])

		if record:
			plt.figure()
			plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			if target:
				plt.scatter([target[1]], target[0], marker='o', color='', 
					edgecolors='r', s=1024, linewidths=2)
			plt.xticks([]),plt.yticks([])
			plt.savefig(os.path.join(self.save_dir, str(self.idx) + self.suffix))
			plt.close()
			self.idx += 1

		if target:
			forward = target - self.view_point
			direction = np.arctan2(-forward[0], forward[1])
			if direction < pi * self.diff:
				action = 'right'
			elif direction < pi * (0.5 - self.diff * self.center_diff_ratio):
				action = 'right_forward'
			elif direction < pi * (0.5 + self.diff * self.center_diff_ratio):
				action = 'forward'
			elif direction < pi * (1 - self.diff):
				action = 'left_forward'
			else:
				action = 'left'
		else:
			# no health kit in view
			if random.random() < self.right_weight:
				action = 'right_forward'
			else:
				action = 'left_forward'

		self.actions = [action] * self.memory

		return self.action_dict[action]


	def renew(self):
		self.actions = []

