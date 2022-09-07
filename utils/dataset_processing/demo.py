from http.server import ThreadingHTTPServer
import imp
import os
import time
import traceback
import threading

import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from utils.dataset_processing.evaluation import plot_output

from ..visualisation.plot import plot_grasp, plot_output
from .camera_data import CameraData
from .grasp import detect_grasps
from .post_process import post_process_output
from models.unet import UNet
from models.grconvnet_bins import GenerativeResnet

class GraspGenerator:
	def __init__(self, description, results, saved_model_path, cam_id, output_size, object_id, use_rgb=True, use_depth=False, start=0, attempts=10, use_width=True, width_scaling=150, bins=3, visualise=False):

		self.description = description
		self.object = object_id
		self.save_folder = os.path.join(results, self.description, self.object)
		if not os.path.exists(self.save_folder):
			os.makedirs(self.save_folder)
		self.filepath = os.path.join(self.save_folder, self.object)+'.hdf5'
		if not os.path.exists(self.filepath):
			self.savefile = h5py.File(self.filepath, 'w')
		else:
			self.savefile = h5py.File(self.filepath, 'r+')

		self.saved_model_path = saved_model_path
		self.model = None
		self.device = None
		self.start = start
		self.attempts = attempts

		self.camera = RealSenseCamera(device_id=cam_id)
		
		self.use_rgb = use_rgb
		self.use_depth = use_depth
		self.use_width = use_width
		self.width_scaling = width_scaling
		self.cam_data = CameraData(output_size=output_size, include_depth=use_depth, include_rgb=use_rgb)
		# Connect to camera
		self.camera.connect()
		self.cam_depth_scale = self.camera.scale

		self.z_offset = 0.02

		self.initial = None
		self.post_grasp = None
		self.threshold = 0.9
		self.handle_as_ggcnn = False
		self.bins=bins

		#change datapath to location of saved intrinsic data
		datapath = os.path.join(os.path.expanduser( '~' ), 'ros_ws/intrinsics')
		self.cm = np.load(os.path.join(datapath, 'camera_pose.npy'))

		# self.inv_cm = np.linalg.inv(self.cm)
		self.grasp_request = os.path.join(datapath, "grasp_request.npy")
		self.grasp_available = os.path.join(datapath, "grasp_available.npy")
		self.grasp_pose = os.path.join(datapath, "grasp_pose.npy")
		self.grasp_success = os.path.join(datapath, "grasp_success.npy")
		self.grasp_closure = os.path.join(datapath, "grasp_closure.npy")

		np.save(self.grasp_available, 0)
		np.save(self.grasp_request, 1)

		self.visualise =  visualise
		self.fig = plt.figure()
		self.gs = gridspec.GridSpec(3, self.bins+3)
		self.gs.tight_layout(self.fig)

		plt.ion()

	def get_input(self):
		keystrk=input('Press a key \n')
		# thread doesn't continue until key is pressed
		print('You pressed: ', keystrk)
		self.grasp_flag=False
		print('flag is now:', self.grasp_flag)

	def load_model(self):
		print('Loading model... ')		
		self.model = torch.load(self.saved_model_path)
		# Get the compute device
		self.device = get_device(force_cpu=False)
		self.handle_as_ggcnn = True

	def load_model_dict(self, network):
		print('Loading {} Model... '.format(str(network)))
		if network == 'unet':
			self.model = UNet(4, self.bins)
		elif network == 'gr':
			self.model = GenerativeResnet(input_channels=4, output_channels=self.bins)
		else:
			self.model = None
		self.model.load_state_dict(torch.load(self.saved_model_path)['model_state_dict'])
		self.device = get_device(force_cpu=False)
		self.model.to(self.device)
		self.model.eval()

	def generate(self, attempt=0):
		print('Generating grasps...')
		grasps = []
		self.model.eval()
		camera_check = 0 # test to make sure camera is on
		while not camera_check:
			try:
				while True:
					# print('Getting Image bundle') # Get RGB-D image from camera
					image_bundle = self.camera.get_image_bundle()
					rgb = image_bundle['rgb']
					depth = image_bundle['aligned_depth']
					x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

					# print('Model prediction: ') # Predict the grasp pose using the saved model
					with torch.no_grad():
						xc = x.to(self.device)
						pred = self.model.predict(xc)

					plt.clf()

					q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'], gauss=self.handle_as_ggcnn, width_scaling=self.width_scaling)
					if self.use_width:
						grasps = detect_grasps(q_img, ang_img, width_img, handle_as_ggcnn=self.handle_as_ggcnn)
					else:
						grasps = detect_grasps(q_img, ang_img, handle_as_ggcnn=self.handle_as_ggcnn)

					grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
					
					grasp_ax = self.fig.add_subplot(self.gs[0:3, 0:3])
					grasp_ax.set_title('Grasp')
					grasp_ax.axis('off')

					grasp_ax.imshow(self.cam_data.get_rgb(rgb, False))

					for g in grasps:
						g.plot(grasp_ax)

					# depth_ax = self.fig.add_subplot(self.gs[0,2])
					# depth_ax.set_title('Depth')
					# depth_ax.axis('off')
					# depth_ax.imshow(np.squeeze(self.cam_data.get_depth(depth)), cmap='gray')
					
					if self.bins > 1:
						for i in range(0, self.bins):
							q_ax = self.fig.add_subplot(self.gs[0, i+3])
							# q_ax.set_title('Q')
							q_ax.axis('off')
							q_ax.imshow(q_img[i], cmap='Greys', vmin=0, vmax=1)

						for i in range(0, self.bins):
							ang_ax = self.fig.add_subplot(self.gs[1, i+3])
							# ang_ax.set_title('Angle')
							ang_ax.axis('off')
							ang_ax.imshow(ang_img[i], cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)

						for i in range(0, self.bins):
							w_ax = self.fig.add_subplot(self.gs[2,i+3])
							# w_ax.set_title('Width')
							w_ax.axis('off')
							w_ax.imshow(width_img[i], cmap='Greys', vmin=0, vmax=150)
					else:
							q_ax = self.fig.add_subplot(self.gs[0, 3])
							q_ax.set_title('Q')
							q_ax.axis('off')
							q_ax.imshow(q_img, cmap='Greys', vmin=0, vmax=1)

							ang_ax = self.fig.add_subplot(self.gs[1, 3])
							ang_ax.set_title('Angle')
							ang_ax.axis('off')
							ang_ax.imshow(ang_img, cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)

							w_ax = self.fig.add_subplot(self.gs[2, 3])
							w_ax.set_title('Width')
							w_ax.axis('off')
							w_ax.imshow(width_img, cmap='Greys', vmin=0, vmax=150)

					plt.pause(0.1)
					self.fig.canvas.draw()
				
					time.sleep(0.1)
			except KeyboardInterrupt:
				print('Press Ctrl-C to attempt grasp')
				pass

			self.initial = self.cam_data.get_cropped(rgb=rgb)

			# Get grasp position from model output
			x = grasps[0].center[1] + self.cam_data.top_left[1]
			y = grasps[0].center[0] + self.cam_data.top_left[0]
			depth_point = depth[y, x]
			depth_point = self.camera.pixel_to_point(x, y, depth_point) # get x,y,z camera coordinate value from point cloud frame
			pos_x, pos_y, pos_z = depth_point[0], depth_point[1], depth_point[2]

			# If no point cloud data found
			if pos_z == 0:
				print('No point cloud data', pos_x, pos_y, pos_z)
				time.sleep(1.0)
				continue
			
			camera_check = 1

		target = np.asarray([pos_x, pos_y, pos_z])
		print('target pixel: ', x, y, 'target point: ', target)
		target.shape = (3, 1)

		# Convert Z-out camera frame to Z-up robot frame
		camera2robot = self.cm
		target_position = np.dot(camera2robot[0:3, 0:3], target) + camera2robot[0:3, 3:] #add camera translation
		target_position[2] = target_position[2] - self.z_offset # Apply offset from end-effector to gripper
		target_position = target_position[0:3, 0]

		# Convert camera to robot angle
		angle = np.asarray([0, 0, grasps[0].angle])
		# need to rotate planar angle by rotation axis
		angle.shape = (3, 1)
		target_angle = np.dot(camera2robot[0:3, 0:3], angle)

		# Convert pixel length to real world gripper length
		mmpixel = 0.8 #mm per pixel
		mpixel = mmpixel/1000 # meters per pixel
		width = np.asarray([0.031])
		max_width = np.asarray([0.031])
		
		if self.use_width:
			pixel_length = grasps[0].length
			print("length of grasp in pixels:", pixel_length)
			width = np.asarray(pixel_length * mpixel)	
		if width > max_width or width < 0.0075: # if width is above max threshold or below min threshold
			width = max_width

		# Concatenate grasp pose with grasp angle
		grasp_pose = np.append(target_position, target_angle[2])
		grasp_pose = np.append(grasp_pose, width)
		print('grasp_pose: ', grasp_pose)

		np.save(self.grasp_pose, grasp_pose)
		self.grasp_flag = 1


	def run(self):
		attempt = 0
		while attempt < self.attempts:
			try:
				if np.load(self.grasp_request, allow_pickle=True):
					time.sleep(0.1)
					print('Grasp attempt: ', attempt)
					self.generate(attempt=attempt)
					np.save(self.grasp_request, 0)
					np.save(self.grasp_available, 1)
					while not np.load(self.grasp_request, allow_pickle=True, fix_imports=True):
						time.sleep(0.1)

					attempt +=1
									
					plt.clf()
				else:
					time.sleep(0.1)
			except OSError as err:
				# import ipdb; ipdb.set_trace()
				print("OS error: {0}".format(err))
				pass
				
		print('Total objects grasped: ', attempt)
