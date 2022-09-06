import os
import glob
import cv2
import numpy as np

from .grasp_data_mask import GraspDatasetBase
from utils.dataset_processing import grasp, image


class JacquardDataset(GraspDatasetBase):
	"""
	Dataset wrapper for the Jacquard dataset.
	"""
	def __init__(self, file_path, **kwargs):
		"""
		:param file_path: Jacquard Dataset directory.
		:param start: If splitting the dataset, start at this fraction [0,1]
		:param end: If splitting the dataset, finish at this fraction
		:param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
		:param kwargs: kwargs for GraspDatasetBase
		"""
		super(JacquardDataset, self).__init__(**kwargs)

		graspf = glob.glob(os.path.join(file_path, '*', '*', '*_grasps.txt'))
		graspf.sort()
		l = len(graspf)

		if l == 0:
			raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
		
		if self.ids is not None:
			graspf = [graspf[i] for i in self.ids]
		else:
			graspf = list(graspf)
		
		depthf = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in graspf]
		rgbf = [f.replace('perfect_depth.tiff', 'RGB.png') for f in depthf]
		maskf = [f.replace('grasps.txt', 'mask.png') for f in graspf]

		self.grasp_files = graspf
		self.depth_files = depthf
		self.rgb_files = rgbf
		self.mask_files = maskf

		if self.include_depth:
			self.depth_background = self.depth_background
			self.depth_max = self.depth_background.max()
		if self.include_rgb:
			self.rgb_background = self.rgb_background

	def get_gtbb(self, idx, rot=0, zoom=1.0):
		gtbbs = grasp.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
		c = self.output_size//2
		gtbbs.rotate(rot, (c, c))
		gtbbs.zoom(zoom, (c, c))
		return gtbbs

	def get_depth(self, idx, rot=0, zoom=1.0):
		depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
		scale = self.depth_max - depth_img.img.max()
		depth_img.img += scale # scale data to new scene

		depth_img.rotate(rot)
		depth_img.zoom(zoom)
		depth_img.resize((self.output_size, self.output_size))
		
		return depth_img.img

	def get_rgb(self, idx, rot=0, zoom=1.0, transpose=True):
		rgb_img = image.Image.from_file(self.rgb_files[idx])

		rgb_img.rotate(rot)
		rgb_img.zoom(zoom)
		rgb_img.resize((self.output_size, self.output_size))
		if transpose:
			# rgb_img.normalise()
			rgb_img.img = rgb_img.img.transpose((2, 0, 1))
		return rgb_img.img

	def get_mask(self, idx, rot=0, zoom=1.0):
		mask = cv2.imread(self.mask_files[idx], 0)
		img_mask = mask > 0
		img_mask = image.Image(img_mask)

		img_mask.rotate(rot)
		img_mask.zoom(zoom)
		img_mask.resize((self.output_size, self.output_size))
		return img_mask.img

	def get_back_depth(self, depth_background, inpaint=True):
		depth_back = image.DepthImage.from_tiff(depth_background)
		height = depth_back.shape[0]
		width = depth_back.shape[1]

		left = (width - self.output_size) // 2
		top = (height - self.output_size) // 2
		right = (width + self.output_size) // 2
		bottom = (height + self.output_size) // 2

		bottom_right = (bottom, right)
		top_left = (top, left)

		depth_back.crop(bottom_right=bottom_right, top_left=top_left)
		if inpaint:
			depth_back.inpaint()
		# print('depth_back', depth_img.min(), depth_img.max())
		# depth_back.resize((self.output_size, self.output_size))
		return depth_back.img

	def get_back_rgb(self, rgb_background, transpose=True):
		rgb_back = cv2.imread(rgb_background)
		rgb_back = image.Image(rgb_back)

		height = rgb_back.shape[0]
		width = rgb_back.shape[1]

		left = (width - self.output_size) // 2
		top = (height - self.output_size) // 2
		right = (width + self.output_size) // 2
		bottom = (height + self.output_size) // 2

		bottom_right = (bottom, right)
		top_left = (top, left)

		rgb_back.crop(bottom_right=bottom_right, top_left=top_left)
		# rgb_back.resize((self.output_size, self.output_size))
		if transpose:
			rgb_back.img = rgb_back.img.transpose((2, 0, 1))
		return rgb_back.img

	def get_jname(self, idx):
		return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])