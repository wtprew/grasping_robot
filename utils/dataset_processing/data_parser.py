from __future__ import division

import csv
import os
import sys
import time

import cv2
import h5py
import numpy as np

from PIL import Image
from .depth_image_encoding import FloatArrayToRgbImage

file_extension = '.png'
depth_file_extension = '.tiff'

def main(save=False):
	fileList = []
	# counter = 0
	paths = ['/data/h5py/'] # list data directories here
	save_dir = '/data/'
	for path in paths:
		for f in os.listdir(path):
			if f[-5:] == '.hdf5':
				fileList.append(path + f)

	print('Processing %d samples' % len(fileList))
	
	for i in range(len(fileList)):
		print(i, fileList[i])
		with h5py.File(fileList[i], 'r') as f:
			# try:
			timestamp = f['timestamp'][()]
			save_dir_path = os.path.join(save_dir, str(timestamp))
			
			try:
				os.mkdir(save_dir_path)
			except:
				pass

			#images
			initial = f['initial_img'][()]
			# initial_depth_rgb = FloatArrayToRgbImage(initial[:,:,-1])
			step = f['pregrasp'][()]
			gripper = f['gripper'][()]
			post_grasp = f['post_grasp'][()]
			post_drop = f['post_drop'][()]

			if save:
				cv2.imwrite(os.path.join(save_dir_path, 'initial' + str(file_extension)), cv2.cvtColor(initial[:,:,:3].astype(np.uint8), cv2.COLOR_RGB2BGR))
				cv2.imwrite(os.path.join(save_dir_path, 'inital_depth' + str(depth_file_extension)), (((initial[:,:,-1].astype(np.float64))/(initial[:,:,-1].max()))*255.0).astype(np.uint8))
				cv2.imwrite(os.path.join(save_dir_path, 'step' + str(file_extension)), cv2.cvtColor(step[:,:,:3], cv2.COLOR_RGB2BGR))
				cv2.imwrite(os.path.join(save_dir_path, 'step_depth' + str(depth_file_extension)), step[:,:,-1:])
				cv2.imwrite(os.path.join(save_dir_path, 'gripper' + str(file_extension)), cv2.cvtColor(gripper[:,:,:3], cv2.COLOR_RGB2BGR))
				cv2.imwrite(os.path.join(save_dir_path, 'gripper_depth' + str(depth_file_extension)), gripper[:,:,-1:])
				cv2.imwrite(os.path.join(save_dir_path, 'post_grasp' + str(file_extension)), cv2.cvtColor(post_grasp[:,:,:3], cv2.COLOR_RGB2BGR))
				cv2.imwrite(os.path.join(save_dir_path, 'post_grasp_depth' + str(depth_file_extension)), post_grasp[:,:,-1:])
				cv2.imwrite(os.path.join(save_dir_path, 'post_drop' + str(file_extension)), cv2.cvtColor(post_drop[:,:,:3], cv2.COLOR_RGB2BGR))
				cv2.imwrite(os.path.join(save_dir_path, 'post_drop_depth' + str(depth_file_extension)), post_drop[:,:,-1:])

			#poses
			pregrasp_pose = f['pregrasp_pose'][()]
			gripper_pose = f['reached_pose'][()]
			
			#chosen parameters
			parameters = f['parameters'][()]

			#joints
			pregrasp_joints = f['pregrasp_joints'][()]
			gripper_joints = f['gripper_joints'][()]
			post_grasp_pose = f['post_grasp_joints'][()]
			post_drop_pose = f['post_drop_joints'][()]

			grasp = np.concatenate([gripper_pose[:3], [gripper_joints[4]]], axis=0)
			angle = gripper_joints[0] + gripper_joints[4]

			#success of 
			# success = f['success'][()]
			width = f['gripper_status'][()]
			error = f['gripper_error'][()]

			with open(os.path.join(save_dir_path, 'data.csv'), mode='w') as c:
				writer = csv.writer(c)
				writer.writerow(['directory path', save_dir_path])
				# writer.writerow(['success', success.to_list()])
				writer.writerow(['gripper_status', width[0]])
				writer.writerow(['gripper_error', error])
				writer.writerow(['params', parameters])
				writer.writerow(['angle', angle])
				writer.writerow(['grasp', grasp])
				writer.writerow(['initial', os.path.join(save_dir_path, 'initial.jpg'), os.path.join(save_dir_path, 'initial_depth.jpg')])
				writer.writerow(['step', os.path.join(save_dir_path, 'step.jpg'), os.path.join(save_dir_path, 'step_depth.jpg')])
				writer.writerow(['pregrasp_pose', pregrasp_pose])
				writer.writerow(['pregrasp_joints', pregrasp_joints])
				writer.writerow(['gripper', os.path.join(save_dir_path, 'gripper.jpg'), os.path.join(save_dir_path, 'gripper_depth.jpg')])
				writer.writerow(['reached_pose', pregrasp_pose])
				writer.writerow(['gripper_joints', pregrasp_joints])
				writer.writerow(['post_grasp', os.path.join(save_dir_path, 'post_grasp.jpg'), os.path.join(save_dir_path, 'post_grasp_depth.jpg')])
				writer.writerow(['post_grasp_joints', post_grasp_pose])
				writer.writerow(['post_drop', os.path.join(save_dir_path, 'post_drop.jpg'), os.path.join(save_dir_path, 'post_drop_depth.jpg')])
				writer.writerow(['post_drop_joints', post_drop_pose])
				c.close()
			f.close()

			# except:
			# 	print('Skipping')
			# 	continue

if __name__ == "__main__":
	main(save=True)
