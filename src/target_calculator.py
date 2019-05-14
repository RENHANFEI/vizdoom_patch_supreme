import cv2
import numpy as np
from src.patch_extractor import PatchExtractor

from matplotlib import pyplot as plt

class TargetCalculator(object):
	"""docstring for TargetCalculator"""
	def __init__(self, thres=0.6, thres_dict={4:0.615, 7:0.600, 14:0.595}):
		super(TargetCalculator, self).__init__()
		self.thres = thres
		self.patch_extractor = PatchExtractor()
		self.thres_dict = thres_dict

	def get_target_nearest(self, frame):
		max_responses = self.patch_extractor.get_max_responses(frame)
		# key: patch_size, value: (max_response, (max_response_x, max_response_y))
		target = None
		target_y = 0
		for size, value in max_responses.items():
			response, coor = value
			y, x = coor
			# if response > self.thres_dict[size] and y > target_y:
			if response > self.thres and y > target_y:
				target = coor
				target_y = coor[0]

		return target

	def get_target_nearest_test(self, frame):
		max_responses = self.patch_extractor.get_max_responses(frame)
		# key: patch_size, value: (max_response, (max_response_x, max_response_y))
		target = None
		target_y = 0
		det_size, det_response = 0, 0
		for size, value in max_responses.items():
			response, coor = value
			y, x = coor
			if response > self.thres_dict[size] and y > target_y:
			# if response > self.thres and y > target_y:
				target = coor
				target_y = coor[0]
				det_size = size
				det_response = response

		# print(det_size, det_response)
		return target, det_size

	def plot_target(self, frame, target):
		plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		if target:
			plt.scatter([target[1]], target[0], marker='o', color='', 
				edgecolors='r', s=1024, linewidths=2)
		plt.show()

	def get_targets(self, frame):
		max_responses = self.patch_extractor.get_max_responses(frame)
		# key: patch_size, value: (max_response, (max_response_x, max_response_y))
		targets = []
		for size, value in max_responses.items():
			response, coor = value
			y, x = coor
			if response > self.thres and y > frame.shape[0] / 2:
				targets.append(coor)

		# print(det_size, det_response)
		return targets

	def plot_targets(self, frame, targets):
		plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		for target in targets:
			plt.scatter([target[1]], target[0], marker='o', color='', 
				edgecolors='r', s=1024, linewidths=2)
		plt.show()



	''' tried but not used methods '''

	def get_target_mean(self, frame):
		responses = np.array(self.patch_extractor.get_responses(frame))
		# key: patch_size, value: (max_response, (max_response_x, max_response_y))
		mean_response = responses.mean(axis=0)
		target = None
		max_response = mean_response.max()
		if max_response > self.thres:
			target = np.unravel_index(mean_response.argmax(), mean_response.shape)
		return target

	def get_target_max(self, frame):
		max_responses = self.patch_extractor.get_max_responses(frame)
		# key: patch_size, value: (max_response, (max_response_x, max_response_y))
		max_response = self.thres
		target = None
		for size, value in max_responses.items():
			response, coor = value
			if response > max_response:
				target = coor
		return target

	def get_target_weighted(self, frame):
		max_responses = self.patch_extractor.get_max_responses(frame)
		# key: patch_size, value: (max_response, (max_response_x, max_response_y))
		max_weighted_response = 0
		target = None
		size_sum = sum(max_responses.keys())
		patch_num = len(max_responses)
		for size, value in max_responses.items():
			# larger patch size means nearer, larger weight
			response, coor = value
			if response > self.thres_dict[size]:
				weight = (size * patch_num / size_sum) ** 0.2
				weighted_response = response * weight
				if weighted_response > max_weighted_response:
					target = coor
		return target
		