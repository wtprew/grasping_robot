import numpy as np

import torch
import torch.utils.data

import random


class GraspDatasetBase(torch.utils.data.Dataset):
	"""
	An abstract dataset for training GG-CNNs in a common format.
	"""
	def __init__(self, ids=None, output_size=300, include_depth=True, include_rgb=False, random_rotate=False,
				 random_zoom=False, input_only=False):
		"""
		:param output_size: Image output size in pixels (square)
		:param include_depth: Whether depth image is included
		:param include_rgb: Whether RGB image is included
		:param random_rotate: Whether random rotations are applied
		:param random_zoom: Whether random zooms are applied
		:param input_only: Whether to return only the network input (no labels)
		"""
		if ids is not None:
			self.ids = np.array(ids)
		else:
			self.ids = None
		self.output_size = output_size
		self.random_rotate = random_rotate
		self.random_zoom = random_zoom
		self.input_only = input_only
		self.include_depth = include_depth
		self.include_rgb = include_rgb

		self.grasp_files = []

		if include_depth is False and include_rgb is False:
			raise ValueError('At least one of Depth or RGB must be specified.')

	@staticmethod
	def numpy_to_torch(s):
		if len(s.shape) == 2:
			return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
		else:
			return torch.from_numpy(s.astype(np.float32))

	def get_gtbb(self, idx, rot=0, zoom=1.0):
		raise NotImplementedError()

	def get_depth(self, idx, rot=0, zoom=1.0):
		raise NotImplementedError()

	def get_rgb(self, idx, rot=0, zoom=1.0):
		raise NotImplementedError()

	def get_sal(self, idx, rot=0, zoom=1.0):
		raise NotImplementedError()

	def __getitem__(self, idx):
		if self.random_rotate:
			rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
			rot = random.choice(rotations)
		else:
			rot = 0.0

		if self.random_zoom:
			zoom_factor = np.random.uniform(0.5, 1.0)
		else:
			zoom_factor = 1.0

		# Load the RGB image
		rgb_img = self.get_rgb(idx, rot, zoom_factor)

		# Load the depth image or saliency image        
		if self.include_depth:
			depth_img = self.get_depth(idx, rot, zoom_factor)
		else:
			depth_img = self.get_sal(idx, rot, zoom_factor)
		
		# Load the grasps
		bbs = self.get_gtbb(idx, rot, zoom_factor)

		pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
		width_img = np.clip(width_img, 0.0, 150.0)/150.0

		# print('Depth', depth_img.shape, np.amin(depth_img), np.amax(depth_img))
		depth = self.numpy_to_torch(depth_img)
		# print('Depth', depth.shape, torch.min(depth), torch.max(depth))
		x = self.numpy_to_torch(rgb_img)
		# print('RGB', x.shape, torch.min(x), torch.max(x))

		pos = self.numpy_to_torch(pos_img)
		cos = self.numpy_to_torch(np.cos(2*ang_img))
		sin = self.numpy_to_torch(np.sin(2*ang_img))
		width = self.numpy_to_torch(width_img)

		return x, (pos, cos, sin, width, depth), idx, rot, zoom_factor

	def __len__(self):
		return len(self.grasp_files)
