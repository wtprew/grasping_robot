# Requires python2 and rospy
import copy
import os
import time
import traceback

import numpy as np
import rospy
from geometry_msgs.msg import Pose

from hardware.robot import WidowX
from hardware.config import *
from hardware.utils import inside_polygon
from utils.transforms import rotate_pose_msg_by_euler_angles, get_pose


class PickAndPlace:
	def __init__(
			self,
			place_position,
			datapath = homedir,
			hover_distance=0.12,
			step_size=0.05):
		"""
		@param place_position: Place position as [x, y, z]
		@param hover_distance: Distance above the pose in meters
		@param step_size: Step size for approaching the pose
		"""
		self.place_position = place_position
		self._hover_distance = hover_distance
		self.step_size = step_size

		self.robot = WidowX()

		self.datapath = datapath

		self.grasp_request = os.path.join(self.datapath, "grasp_request.npy")
		self.grasp_available = os.path.join(self.datapath, "grasp_available.npy")
		self.grasp_pose = os.path.join(self.datapath, "grasp_pose.npy")
		self.grasp_success = os.path.join(self.datapath, "grasp_success.npy")
		self.grasp_closure = os.path.join(self.datapath, "grasp_closure.npy")

		rospy.sleep(2)

	def get_pose(self):
		pose = self.robot.get_current_pose().pose
		pose_list = [pose.position.x,
					pose.position.y,
					pose.position.z,
					pose.orientation.w,
					pose.orientation.x,
					pose.orientation.y,
					pose.orientation.z]
		return pose_list

	def _approach(self, pose):
		print('approaching...')
		approach = copy.deepcopy(pose)
		try:
			x, y, z, theta, width = approach

			print('Attempting grasp: (%.4f, %.4f, %.4f, %.4f, %.4f)'
				  % (x, y, z, theta, width))

			assert inside_polygon(
				(x, y, z), END_EFFECTOR_BOUNDS), 'Grasp not in bounds'

			assert self.robot.orient_to_pregrasp(
				x, y, width), 'Failed to orient to target'

			assert self.robot.move_to_grasp(x, y, self._hover_distance, theta), \
				'Failed to reach pre-lift pose'

			assert self.robot.move_to_grasp(
				x, y, z, theta), 'Failed to execute grasp'

		except Exception as e:
			print('Error executing grasp -- returning...')
			traceback.print_exc(e)
			self.robot.move_to_neutral()
			return

	def _retract(self, pose):
		"""
		Retract up from current pose
		"""
		try:
			reached = self.robot.move_to_vertical(PRELIFT_HEIGHT)

			assert self.robot.move_to_place(pose, 0.2, angle=0), 'Failed to move to place location'

			assert self.robot.move_to_pose(pose), 'Failed to move to drop'

			time.sleep(1)

			success, error = self.robot.eval_grasp()

			return success, error

		except Exception as e:
			print('Error executing grasp -- returning...')
			traceback.print_exc(e)
			return False, 1

	def pick(self, grasp_pose):
		"""
		Pick from given pose
		"""
		# Orient arm to area above scene
		self.robot.orient_to_pregrasp(grasp_pose[0], grasp_pose[1])

		# open the gripper
		self.robot.open_gripper()
		# approach to the pose
		self._approach(grasp_pose)
		# close gripper
		self.robot.close_gripper()

	def place(self, place_position):
		"""
		Place to given pose
		"""
		# Calculate pose from place position
		pose = get_pose(position=place_position)

		# approach to the pose
		success, closure = self._retract(pose)

		# open the gripper
		self.robot.open_gripper()
		# retract to clear object
		self.robot.move_to_neutral()
		# Get the next grasp pose
		np.save(self.grasp_request, 1, allow_pickle=True, fix_imports=True)
		return success, closure

	def run(self):

		# Initialize grasp request and grasp available
		np.save(self.grasp_request, 0)
		np.save(self.grasp_available, 0)
		np.save(self.grasp_success, 0)

		# Move robot to home pose
		print('Moving to start position...')
		self.robot.move_to_neutral()
		self.robot.open_gripper()

		# Get the first grasp pose
		np.save(self.grasp_request, 1)

		while not rospy.is_shutdown():
			print('Waiting for grasp pose...')
			while not np.load(self.grasp_available) and not rospy.is_shutdown():
				time.sleep(0.1)
			grasp_pose = np.load(self.grasp_pose)
			np.save(self.grasp_available, 0)

			# Perform pick
			print 'Picking from ', grasp_pose
			self.pick(grasp_pose)

			# Perform place
			print 'Placing to ', self.place_position
			success, closure = self.place(self.place_position)
			np.save(self.grasp_closure, closure, allow_pickle=True)
			# print success
			if success:
				np.save(self.grasp_success, 1, allow_pickle=True)
				print 'Object grasped successfully, Error: {:.4f}'.format(closure)
			else:
				np.save(self.grasp_success, 0, allow_pickle=True)
				print 'Object incorrectly grasped, Error: {:.4f}'.format(closure)
			time.sleep(1)
