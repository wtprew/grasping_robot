import os
import time
import traceback

# from cv2 import 

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from skimage.metrics import structural_similarity as ssim
from utils.dataset_processing.evaluation import plot_output

from ..visualisation.plot import plot_grasp, plot_output
from .camera_data import CameraData
from .grasp import detect_grasps
from .post_process import post_process_output
from models.unet import UNet
from models.grconvnet_bins import GenerativeResnet
class GraspGenerator:
	def __init__(self, description, results, saved_model_path, cam_id, output_size, object_id, use_rgb=True, use_depth=False, start=0, attempts=1, use_width=True, width_scaling=150, bins=3, visualise=False):

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

		datapath = os.path.join('/media/will/research/ros_ws/intrinsics')
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

	def load_model(self):
		print('Loading model... ')		
		self.model = torch.load(self.saved_model_path)
		# Get the compute device
		self.device = get_device(force_cpu=False)
		self.handle_as_ggcnn = True

	def load_model_dict(self, network):
		print('Loading {} Model... '.format(str(network)))
		if network == 'unet':
			self.model = UNet(4, 3)
		elif network == 'gr':
			self.model = GenerativeResnet(input_channels=4, output_channels=3)
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
			while len(grasps) == 0:
				# print('Getting Image bundle') # Get RGB-D image from camera
				image_bundle = self.camera.get_image_bundle()
				rgb = image_bundle['rgb']
				depth = image_bundle['aligned_depth']
				x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

				print('Model prediction: ') # Predict the grasp pose using the saved model
				with torch.no_grad():
					xc = x.to(self.device)
					pred = self.model.predict(xc)
		
				q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'], gauss=self.handle_as_ggcnn, width_scaling=self.width_scaling)
				if self.use_width:
					grasps = detect_grasps(q_img, ang_img, width_img, handle_as_ggcnn=self.handle_as_ggcnn)
				else:
					grasps = detect_grasps(q_img, ang_img, handle_as_ggcnn=self.handle_as_ggcnn)

				if len(grasps) == 0:
					fig = plt.figure(figsize=(10, 10))
					ax = fig.add.subplot(111)
					ax.imshow(q_img[0])
					ax.set_title('Probabilities')
					ax.axis('off')
					fig.canvas.draw()
			else:
				time.sleep(0.1)
			
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

		self.savefile.create_dataset(str(attempt)+'/rgb', data=rgb_img)
		self.savefile.create_dataset(str(attempt)+'/depth', data=depth_img)
		self.savefile.create_dataset(str(attempt)+'/cos', data=pred['cos'].cpu().numpy())
		self.savefile.create_dataset(str(attempt)+'/sin', data=pred['sin'].cpu().numpy())
		self.savefile.create_dataset(str(attempt)+'/quality', data=q_img)
		self.savefile.create_dataset(str(attempt)+'/probability', data=q_img.max())
		self.savefile.create_dataset(str(attempt)+'/angle', data=ang_img)
		self.savefile.create_dataset(str(attempt)+'/width', data=width_img)
		self.savefile.create_dataset(str(attempt)+'/target', data=target)
		self.savefile.create_dataset(str(attempt)+'/grasp_centre', data=np.asarray([grasps[0].center[1], grasps[0].center[0]]))
		self.savefile.create_dataset(str(attempt)+'/grasp_angle', data=np.asarray(grasps[0].angle))
		self.savefile.create_dataset(str(attempt)+'/grasp_length', data=np.asarray(grasps[0].length))
		self.savefile.create_dataset(str(attempt)+'/grasp_width', data=np.asarray(grasps[0].width))
		self.savefile.create_dataset(str(attempt)+'/grasp_pose', data=grasp_pose)

		if self.visualise:
			plot_grasp(rgb_img=self.cam_data.get_rgb(rgb, norm=False), grasps=grasps,
						grasp_q_img=q_img, grasp_angle_img=ang_img, grasp_width_img=width_img,
						save=True, save_folder=self.save_folder, attempt=attempt)
			plot_output(q_img, ang_img, width_img, save=True, save_folder=self.save_folder, attempt=attempt)

	def run(self):
		attempt = self.start
		successful = 0
		while attempt < self.attempts:
			try:
				if np.load(self.grasp_request, allow_pickle=True):
					time.sleep(0.1)
					try:
						group = self.savefile.create_group(str(attempt))
					except:
						del self.savefile[str(attempt)]
						group = self.savefile.create_group(str(attempt))
					print('Grasp attempt: ', attempt)
					self.generate(attempt=attempt)
					np.save(self.grasp_request, 0)
					np.save(self.grasp_available, 1)
					while not np.load(self.grasp_request, allow_pickle=True, fix_imports=True):
						time.sleep(0.1)
					success = np.load(self.grasp_success, allow_pickle=True, fix_imports=True)
					closure = np.load(self.grasp_closure, allow_pickle=True, fix_imports=True)

					# Get image after grasp attempt
					image_bundle = self.camera.get_image_bundle()
					rgb = image_bundle['rgb']
					self.post_grasp = self.cam_data.get_cropped(rgb=rgb)

					difference = ssim(self.initial, self.post_grasp, multichannel=True)
					
					group.create_dataset(str(attempt)+'/post_grasp', data=self.post_grasp)
					group.create_dataset(str(attempt)+'/ssim', data=np.asarray(difference))
					dset = group.create_dataset(str(attempt)+'/grasp_success', data=success)
					group.create_dataset(str(attempt)+'/grasp_closure', data=closure)
					
					if success or (difference < self.threshold) and (closure > 0.0001):
						print(f'Successful grasp, ssim: {difference:.2f} closure: {closure:.4f}')
						att = 0
					else:
						print(f'Unsuccessful grasp, ssim: {difference:.2f} closure: {closure:.4f}')
						att = int(input('Reason for failed grasp: \n 1: Wrong position \n 2: Wrong angle \n 3: Wrong width \n 4: Wrong height \n 5: Gripper collision \n 6: Out of bounds \n 7: Object Difficulty \n'))
						if att == 6:
							print('Out of bounds, repeating experiment')
							plt.close('all')
							continue
					if att == 0:
						successful += 1
					dset.attrs.create('result', data=att)

					attempt += 1
					plt.close('all')
				else:
					time.sleep(0.1)
			except OSError as err:
				# import ipdb; ipdb.set_trace()
				print("OS error: {0}".format(err))
				pass
				
		print('Total objects grasped correctly: ', successful, '/', self.attempts-self.start)
