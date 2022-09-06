import os

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, random_split

import depth_image_encoding

def gt(row):
	s_threshold = 0.2
	ssim_threshold = 0.96
	if row.ssim == np.nan:
		if row.status > s_threshold:
			return 1
		else:
			return 0
	else:
		if (row.ssim < ssim_threshold) or (row.status > s_threshold):
			return 1
		else:
			return 0

class graspdataset(Dataset):
	def __init__(self, df, root, train=True):
		self.df = df
		self.root = root
		self.train = train

	#Perform the same random crop on each image
	def transform(self, initial, depthinitial, current, depthcurrent):
		if self.train == True:
			# initial = TF.center_crop(initial, 512)
			# current = TF.center_crop(current, 512)
			# i, j, h ,w = transforms.RandomCrop.get_params(initial, output_size=(472, 472))
			# initial = TF.crop(initial, i, j, h, w)
			# current = TF.crop(current, i, j, h, w)
			initial = initial[-472:,-472:, :]
			depthinitial = depthinitial[-472:,-472:, :]
			initiallab = cv2.cvtColor(initial, cv2.COLOR_BGR2LAB)
			initialfloat = depth_image_encoding.ImageToFloatArray(depthinitial)
			# initialfloat = cv2.cvtColor(depthinitial, cv2.COLOR_BGR2GRAY)
			initiallab[:,:,0] = initialfloat
			current = current[-472:,-472:, :]
			depthcurrent = depthcurrent[-472:,-472:, :]
			currentlab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB)
			depthfloat = depth_image_encoding.ImageToFloatArray(depthcurrent)
			# depthfloat = cv2.cvtColor(depthcurrent, cv2.COLOR_BGR2GRAY)
			currentlab[:,:,0] = depthfloat
			initial = TF.to_tensor(initiallab)
			current = TF.to_tensor(currentlab)
		elif self.train == False:
			# initial = TF.center_crop(initial, 472)
			# current = TF.center_crop(current, 472)
			initial = initial[-472:,-472:, :]
			depthinitial = depthinitial[-472:,-472:, :]
			initiallab = cv2.cvtColor(initial, cv2.COLOR_BGR2LAB)
			initialfloat = depth_image_encoding.ImageToFloatArray(depthinitial)
			initiallab[:,:,0] = initialfloat
			current = current[-472:,-472:, :]
			depthcurrent = depthcurrent[-472:,-472:, :]
			currentlab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB)
			depthfloat = depth_image_encoding.ImageToFloatArray(depthcurrent)
			currentlab[:,:,0] = depthfloat
			initial = TF.to_tensor(initiallab)
			current = TF.to_tensor(currentlab)
		return initial, current

	def __getitem__(self, idx):
		root = self.root
		df = self.df
		initialimage = os.path.join(root, df.iloc[idx, 3])
		initialimage = cv2.imread(initialimage)
		initialdepth = os.path.join(root, df.iloc[idx, 4])
		initialdepth = cv2.imread(initialdepth)
		stepimage = os.path.join(root, df.iloc[idx, 5])
		stepimage = cv2.imread(stepimage)
		stepdepth = os.path.join(root, df.iloc[idx, 6])
		stepdepth = cv2.imread(stepdepth)
		initialimage, stepimage = self.transform(initialimage, initialdepth, stepimage, stepdepth)
		g = df.iloc[idx, -1]
		ground_truth = torch.Tensor([g])

		paramsx = df.iloc[idx, 26]
		paramsy = df.iloc[idx, 27]
		paramsz = df.iloc[idx, 28]
		rotation = df.iloc[idx, 29]
		cos = df.iloc[idx, 11]
		param_numpy = np.array((paramsx, paramsy, paramsz, rotation, cos), dtype=np.float32)
		params_tensor = torch.from_numpy(param_numpy)

		sample = {"ground": ground_truth, "initial": initialimage, "current": stepimage, "params": params_tensor}
		return sample

	def __len__(self):
		return len(self.df)
