import os
import cv2
import numpy as np
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

class PatchExtractor(object):
	"""docstring for PatchExtractor"""
	def __init__(self, patch_dir='patches', suffix='.png'):
		super(PatchExtractor, self).__init__()
		self.patches = []
		self.patch_dir = patch_dir
		# self.patch_scales = [4, 7, 8, 9, 12, 14, 15, 17, 19, 21, 25, 55]
		# to accelerate, use most useful patches (verified in tests)
		self.patch_scales = [4, 7, 14]
		# self.patch_scales = [7]
		self.suffix = suffix
		self.load_patches()

	def load_patches(self):
		for scale in self.patch_scales:
			patch = cv2.imread(os.path.join(self.patch_dir, str(scale) + self.suffix))
			patch = patch
			self.patches.append(patch)

	def get_responses(self, frame):
		fimgs = []
		for patch in self.patches:
			fimg = self.ssd(frame, patch)
			fimgs.append(fimg)
		return fimgs

	def get_max_responses(self, frame):
		fimgs = self.get_responses(frame)
		responses = {}
		for i, fimg in enumerate(fimgs):
			max_response = fimg.max()
			coor = np.unravel_index(fimg.argmax(), fimg.shape)
			responses[self.patch_scales[i]] = (max_response, coor)
		return responses

	def plot_response(self, frame):
		fimgs = self.get_responses(frame)
		rows = 1
		cols = int((len(self.patch_scales) + rows - 1) / rows)
		plt.figure(num='Patch Match Responses')
		for i, fimg in enumerate(fimgs):
			plt.subplot(rows, cols, i + 1)
			# plt.title(str(self.patch_scales[i]))
			# plt.imshow(fimg, cmap='gray', vmin=0, vmax=1)
			plt.imshow(fimg, cmap='gray', vmin=0)
		
		plt.show()


	def ssd(self, image, patch, stride=3):
		w, h, _ = patch.shape
		w_padding = int((w - 1) / 2)
		h_padding = int((h - 1) / 2)
		result = np.zeros_like(image[:, :, 0], dtype=np.float32)
		image = cv2.copyMakeBorder(image, w_padding, w_padding,
			h_padding, h_padding, cv2.BORDER_CONSTANT, value=[0,0,0])
		y, x = result.shape
		y_margin = 0.05
		x_margin = 0.03
		# narrow down the region for patch search
		for i in range(int(y / 2), int(y * (1 - y_margin))):
			for j in range(int(x * x_margin), int(x * (1 - x_margin))):
		# for i in range(y):
		# 	for j in range(x):
				# result[i, j] = 1 - np.linalg.norm(image[i:i+w, j:j+h, :] 
				# 	- patch) / w / h / 3 / 255
				result[i, j] = 1 - (((image[i:i+w, j:j+h, :] - patch) 
					** 2).sum() / w / h / 3 / 255) ** 0.5
		return result


		