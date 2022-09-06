import os
import time
import traceback

import h5py
import math as maths
import cv2
import matplotlib.pyplot as plt
import numpy as np

from hardware.camera import RealSenseCamera
from skimage.metrics import structural_similarity as ssim

from .grasp import Grasp
from ..visualisation.plot import plot_grasp
from .camera_data import CameraData
from .grasp import detect_grasps
from .post_process import post_process_output


class GraspGenerator:
	def __init__(self, description, results, cam_id, object_id, use_rgb=True, use_depth=False, use_canny=True, im_thresh=40, start=0, attempts=1, z_offset=0.02, visualize=False):

		self.description = description
		self.object = object_id
		self.save_folder = os.path.join(results, self.description, self.object)
		self.filepath = os.path.join(self.save_folder, self.object)+'.hdf5'
		if not os.path.exists(self.filepath):
			self.savefile = h5py.File(self.filepath, 'w')
		else:
			self.savefile = h5py.File(self.filepath, 'r+')

		self.start = start
		self.attempts = attempts

		self.camera = RealSenseCamera(device_id=cam_id)
		
		self.use_rgb = use_rgb
		self.use_depth = use_depth
		# self.use_width = use_width
		self.cam_data = CameraData(include_depth=use_depth, include_rgb=use_rgb)

		# Connect to camera
		self.camera.connect()
		self.cam_depth_scale = self.camera.scale

		self.im_thresh = im_thresh
		self.canny = use_canny

		self.z_offset = z_offset

		self.initial = None
		self.post_grasp = None
		self.threshold = 0.9

		datapath = os.path.join('/media/will/research/ros_ws/intrinsics')
		self.cm = np.load(os.path.join(datapath, 'camera_pose.npy'))

		# self.inv_cm = np.linalg.inv(self.cm)
		self.grasp_request = os.path.join(datapath, "grasp_request.npy")
		self.grasp_available = os.path.join(datapath, "grasp_available.npy")
		self.grasp_pose = os.path.join(datapath, "grasp_pose.npy")
		self.grasp_success = os.path.join(datapath, "grasp_success.npy")
		self.grasp_closure = os.path.join(datapath, "grasp_closure.npy")

		if visualize:
			self.fig = plt.figure(figsize=(10, 10))
		else:
			self.fig = None

	def generate(self, attempt=0):
		grasps = []
		time.sleep(2)
		while len(grasps) == 0:
			# Get RGB-D image from camera
			image_bundle = self.camera.get_image_bundle()
			rgb = image_bundle['rgb']
			depth = image_bundle['aligned_depth']
			x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

			rgb_img_raw = self.cam_data.get_cropped(rgb=rgb)

			# Convert image to gray and blur it
			src_gray = cv2.cvtColor(rgb_img_raw, cv2.COLOR_BGR2GRAY)
			# src_gray = cv2.blur(src_gray, (3,3))

			max_thresh = 255
			threshold = self.im_thresh # initial threshold

			if self.canny:
				canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
			else:
				thresh, canny_output = cv2.threshold(src_gray, threshold, max_thresh, cv2.THRESH_BINARY_INV)

			contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			# Get the moments
			mu = [None]*len(contours)
			for i in range(len(contours)):
				mu[i] = cv2.moments(contours[i])

			# Get the mass centers
			mc = [None]*len(contours)
			for i in range(len(contours)):
				# add 1e-5 to avoid division by zero
				mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))

			if self.canny:
				# find largest contour
				big_contour = max(contours, key=cv2.contourArea)
			else:
				#for finding a different contour
				sorteddata = sorted(contours, key=cv2.contourArea, reverse=True)				
				big_contour = sorteddata[1]

			# fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree 
			ellipse = cv2.fitEllipse(big_contour)
			(xc,yc), (d1,d2), angle = ellipse

			# draw vertical line
			# compute major radius
			rmajor = max(d1,d2)/2
			if angle > 90:
				angle = angle - 90
			else:
				angle = angle + 90
			xtop = xc + maths.cos(maths.radians(angle))*rmajor
			ytop = yc + maths.sin(maths.radians(angle))*rmajor
			xbot = xc + maths.cos(maths.radians(angle+180))*rmajor
			ybot = yc + maths.sin(maths.radians(angle+180))*rmajor

			centre = (int(yc), int(xc))
			width = maths.sqrt(((xtop-xbot)**2)+((ytop-ybot)**2))
			grasps.append(Grasp(centre, angle, width=width))

			# if len(grasps) == 0:
			if self.fig:
				# Draw contours
				drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
				
				for i in range(len(contours)):
					color = (255, 0, 0)
					cv2.drawContours(drawing, contours, i, color, 2)
					cv2.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)					
					cv2.line(drawing, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 3)
					
				ax = plt.subplot(111)
				ax.imshow(drawing)
				ax.set_title('Contours')
				ax.axis('off')

				# ax1 = plt.subplot(212)
				# ax1.imshow(drawing)
				# ax1.set_title('Grasp')
				# ax1.axis('off')
				self.fig.canvas.draw()
				# plt.show()
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
			return

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
		
		pixel_length = grasps[0].length
		print("length of grasp in pixels:", pixel_length)
		width = np.asarray(pixel_length * mpixel)	
		if width > max_width:
			width = max_width

		# Concatenate grasp pose with grasp angle
		grasp_pose = np.append(target_position, target_angle[2])
		grasp_pose = np.append(grasp_pose, width)
		print('grasp_pose: ', grasp_pose)

		np.save(self.grasp_pose, grasp_pose)

		self.savefile.create_dataset(str(attempt)+'/rgb', data=rgb_img)
		self.savefile.create_dataset(str(attempt)+'/depth', data=depth_img)
		self.savefile.create_dataset(str(attempt)+'/target', data=target)
		self.savefile.create_dataset(str(attempt)+'/grasp_centre', data=np.asarray([grasps[0].center[1], grasps[0].center[0]]))
		self.savefile.create_dataset(str(attempt)+'/grasp_angle', data=np.asarray(grasps[0].angle))
		self.savefile.create_dataset(str(attempt)+'/grasp_length', data=np.asarray(grasps[0].length))
		self.savefile.create_dataset(str(attempt)+'/grasp_width', data=np.asarray(grasps[0].width))
		self.savefile.create_dataset(str(attempt)+'/grasp_pose', data=grasp_pose)

		if self.fig:
			plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, save=True, save_folder=self.save_folder, attempt=attempt)


	def run(self):
		attempt = self.start
		successful = 0
		while attempt < self.attempts:
			try:
				if np.load(self.grasp_request, allow_pickle=True):
					try:
						group = self.savefile.create_group(str(attempt))
					except:
						del self.savefile[str(attempt)]
						group = self.savefile.create_group(str(attempt))
					print('Grasp attempt: ', attempt)
					self.generate(attempt=attempt)
					np.save(self.grasp_request, 0)
					np.save(self.grasp_available, 1)
					while not np.load(self.grasp_request, allow_pickle=True):
						time.sleep(0.1)
					success = np.load(self.grasp_success, allow_pickle=True)
					closure = np.load(self.grasp_closure, allow_pickle=True)

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
						successful += 1
						print(f'Successful grasp, ssim: {difference:.2f} closure: {closure:.4f}')
					else:
						print(f'Unsuccessful grasp, ssim: {difference:.2f} closure: {closure:.4f}')
						att = input('Reason for failed grasp: \n 0: Wrong position \n 1: Wrong angle \n 2: Wrong width \n 3: Wrong height \n 4: Gripper collision \n 5: Out of bounds \n')
						if att == str(5):
							raise Exception('Out of bounds')
						dset.attrs.create('failure', data=att)

					attempt += 1
				else:
					time.sleep(0.1)
			except OSError as err:
				# import ipdb; ipdb.set_trace()
				print("OS error: {0}".format(err))
				pass
				
		print('Total objects grasped correctly: ', successful, '/', self.attempts-self.start)
