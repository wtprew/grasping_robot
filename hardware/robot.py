#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import pickle
import sys
import traceback

import moveit_commander
import moveit_msgs.msg
import numpy as np
import rospy
from arbotix_msgs.srv import *
from geometry_msgs.msg import (Point, PointStamped, Pose, PoseStamped,
							   Quaternion)
from moveit_commander import *
from moveit_commander.conversions import pose_to_list
from moveit_commander.exception import MoveItCommanderException
from moveit_msgs.msg import *
from moveit_msgs.srv import *
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float64, Header, String

try:
	from hardware.config import *
	from hardware.utils import *
except:
	from config import *
	from utils import *

class WidowX(object):
	def __init__(self, workspace_limits=None, verbose=True):
		super(WidowX, self).__init__()
		self._verbose = verbose
		self._iksvc = None

		if workspace_limits is not None:
			self.workspace_limits = workspace_limits

		rospy.init_node('robotic_grasping', anonymous=True)

		self.robot = moveit_commander.RobotCommander()
		self.scene = moveit_commander.PlanningSceneInterface()

		# We can get a list of all the commanders in the robot:
		self.commander_names = self.robot.get_group_names()
		#print("Available Planning commanders: ", self.robot.get_commander_names())

		self.commander = moveit_commander.MoveGroupCommander("widowx_arm")
		self.commander.set_end_effector_link('gripper_rail_link')
		self.gripper = moveit_commander.MoveGroupCommander("widowx_gripper")

		# if boundaries:
		# 	self.add_bounds()

		# Create a `DisplayTrajectory`_ ROS publisher which is used to display
		# trajectories in Rviz:
		self.joint_state_subscriber = rospy.Subscriber("/joint_states", JointState, self.joint_callback)

		self.display_trajectory_publisher = rospy.Publisher('/move_commander/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)		
		self.joint_pubs = [rospy.Publisher('/%s/command' % name, Float64, queue_size=1) for name in JOINT_NAMES]
		self.gripper_pub = rospy.Publisher('/gripper_prismatic_joint/command', Float64, queue_size=1)

		## Getting Basic Information
		# We can get the name of the reference frame for this robot:
		self.planning_frame = self.commander.get_planning_frame()
		self.eef_link = self.commander.get_end_effector_link()
		if verbose:
			print("Planning frame: ", self.planning_frame)

			# We can also print the name of the end-effector link for this commander:
			print("End effector link: ", self.eef_link)
		
			# Sometimes for debugging it is useful to print the entire state of the robot:
			print("Printing robot state")
			print(self.robot.get_current_state())

		homedir = os.path.join('/media/will/research/ros_ws/intrinsics')
		self.move_completed = os.path.join(homedir, "move_completed.npy")
		self.tool_position = os.path.join(homedir, "tool_position.npy")
	  
		self.sample = {}  

	def joint_callback(self, joint_state):
		self.joint_state = joint_state
	
	def state(self):
		print("Robot state", self.robot.get_current_state())
		print("Current commanders: ", self.commander_names, "\nInitialised commander: ", self.commander.get_name())
		print("Joint positions: \n", self.commander.get_active_joints(), "\n", self.commander.get_current_joint_values())
		print("Gripper positions: \n", self.gripper.get_active_joints(), "\n", self.gripper.get_current_joint_values(), "\n")
		print("Current pose: ", self.commander.get_current_pose(), "\n")
		print("Goal joint tolerance", self.commander.get_goal_joint_tolerance())
		print("Goal position tolerance", self.commander.get_goal_position_tolerance())
		print("Goal orientation tolerance", self.commander.get_goal_orientation_tolerance())
		print("Robot constraints Arm: ", self.commander.get_known_constraints(), "Gripper: ", self.gripper.get_known_constraints())
		print("Planning frame ", self.commander.get_planning_frame())
	
	def open_gripper(self, drop=False):
		plan = self.gripper.plan(GRIPPER_DROP if drop else GRIPPER_OPEN)
		return self.gripper.execute(plan, wait=True)
	
	def close_gripper(self):
		plan = self.gripper.plan(GRIPPER_CLOSED)
		return self.gripper.execute(plan, wait=True)

	def prepare_gripper(self, gripper_width=None):
		if gripper_width is not None:
			plan = self.gripper.plan([gripper_width, gripper_width])
		else:
			print("Invalid gripper width. Opening gripper instead.")
			plan = self.gripper.plan(GRIPPER_OPEN)
		return self.gripper.execute(plan, wait=True)
	
	def get_joint_values(self):
		return self.commander.get_current_joint_values()

	def get_current_pose(self):
		return self.commander.get_current_pose()
	
	def effort(self):
		return self.robot.get_current_state().joint_state.effort

	# def add_bounds(self):
	# 	floor = PoseStamped()
	# 	floor.header.frame_id = self.commander.get_planning_frame()
	# 	floor.pose.position.x = WORKSPACE_CENTRE_X
	# 	floor.pose.position.y = 0
	# 	floor.pose.position.z = BOUNDS_FLOOR
	# 	self.scene.add_box('floor', floor, (.5, .5, .001)) #add_box = name, position, size

	def remove_bounds(self):
		for obj in self.scene.get_objects().keys():
			self.scene.remove_world_object(obj)
	
	def get_ik_client(self, request):
		rospy.wait_for_service('/compute_ik')
		inverse_ik = rospy.ServiceProxy('/compute_ik', GetPositionIK)
		ret = inverse_ik(request)
		if ret.error_code.val != 1:
			return None
		return ret.solution.joint_state
	
	def get_fk_client(self, header, link_names, robot_state):
		rospy.wait_for_service('/compute_fk')
		fk = rospy.ServiceProxy('/compute_fk', GetPositionFK)
		ret = fk(header, link_names, robot_state)
		if ret.error_code.val != 1:
			return None
		return ret.pose_stamped

	def sweep_arena(self):
		plan = self.commander.plan(TL_CORNER)
		self.commander.execute(plan, wait=True)
		rospy.sleep(2)

		plan = self.commander.plan(BL_CORNER)
		self.commander.execute(plan, wait=True)
		rospy.sleep(2)

		plan = self.commander.plan(BR_CORNER)
		self.commander.execute(plan, wait=True)
		rospy.sleep(2)

		plan = self.commander.plan(TR_CORNER)
		self.commander.execute(plan, wait=True)
		rospy.sleep(2)

	def move_to(self, pose):
		"""
		Executes plan from given pose in cartesian space
		"""		
		current = self.commander.get_current_pose().pose
		point = Pose(position=Point(x=pose.position.x, y=pose.position.y, z=pose.position.z))#, orientation=quat)

		(plan, fraction) = self.commander.compute_cartesian_path(
			[current, point],   # waypoints to follow
			0.01,        # eef_step
			0.0)         # jump_threshold

		joint_goal = list(plan.joint_trajectory.points[-1].positions)

		try:
			plan = self.commander.plan(joint_goal)
			# np.save()
		except MoveItCommanderException as e:
			print('Exception while planning')
			traceback.print_exc(e)
			return False

		print(fraction)
		return self.commander.execute(plan, wait=True), fraction

	def move_to_pose(self, pose, angle=None):
		"""
		Returns plan for coordinate movement vector in Z-up space (X,Y,Z)
		from one pose to another
		"""		
		current = self.commander.get_current_pose().pose
		point = Pose(position=Point(x=pose.position.x, y=pose.position.y, z=pose.position.z), orientation=pose.orientation)

		(plan, fraction) = self.commander.compute_cartesian_path(
			[current, point],   # waypoints to follow
			0.01,        # eef_step
			0.0)         # jump_threshold

		joint_goal = list(plan.joint_trajectory.points[-1].positions)
		
		if angle:
			first_servo = joint_goal[0]
		
			joint_goal[4] = (angle - first_servo) % np.pi
			if joint_goal[4] > np.pi / 2:
				joint_goal[4] -= np.pi
			elif joint_goal[4] < -(np.pi / 2):
				joint_goal[4] += np.pi

		try:
			plan = self.commander.plan(joint_goal)
			# np.save()
		except MoveItCommanderException as e:
			print('Exception while planning')
			traceback.print_exc(e)
			return False

		print(fraction)
		return self.commander.execute(plan, wait=True), fraction

	def move_to_place(self, pose, z, angle=None):
		"""
		Move to area above position wheer object will be placed
		"""		
		current = self.commander.get_current_pose().pose
		point = Pose(position=Point(x=pose.position.x, y=pose.position.y, z=z), orientation=pose.orientation)

		(plan, fraction) = self.commander.compute_cartesian_path(
			[current, point],   # waypoints to follow
			0.01,        # eef_step
			0.0)         # jump_threshold

		joint_goal = list(plan.joint_trajectory.points[-1].positions)
		
		if angle:
			first_servo = joint_goal[0]
		
			joint_goal[4] = (angle - first_servo) % np.pi
			if joint_goal[4] > np.pi / 2:
				joint_goal[4] -= np.pi
			elif joint_goal[4] < -(np.pi / 2):
				joint_goal[4] += np.pi

		try:
			plan = self.commander.plan(joint_goal)
		except MoveItCommanderException as e:
			print('Exception while planning')
			traceback.print_exc(e)
			return False

		print(fraction)
		return self.commander.execute(plan, wait=True), fraction

	def move_vector(self, x, y, z, theta, quat=DOWN_ORIENTATION):
		'''
		Move from current position with added x, y, z, theta values
		'''
		current = self.commander.get_current_pose().pose
		point = Pose(position=Point(x = current.position.x + x, 
									y = current.position.y + y, 
									z = current.position.z + z), 
									orientation=quat)
		fraction = 0
		count = 0
		
		while fraction < 1.0 and count < 50:
			(plan, fraction) = self.commander.compute_cartesian_path(
				[current, point],   # waypoints to follow
				0.01,        # eef_step
				0.0)         # jump_threshold
			count +=1

		joint_goal = list(plan.joint_trajectory.points[-1].positions)
		
		first_servo = joint_goal[0]

		joint_goal[4] = (theta - first_servo) % np.pi
		if joint_goal[4] > np.pi / 2:
			joint_goal[4] -= np.pi
		elif joint_goal[4] < -(np.pi / 2):
			joint_goal[4] += np.pi

		try:
			plan = self.commander.plan(joint_goal)
		except MoveItCommanderException as e:
			print('Exception while planning')
			traceback.print_exc(e)
			return False

		return self.commander.execute(plan, wait=True), fraction

	def move_to_target(self, target):
		assert len(target) >= 6, 'Invalid target command'
		for i, pos in enumerate(target):
			self.joint_pubs[i].publish(pos)
			
	def move_to_joint_position(self, joints):
		"""
		Adds the given joint values to the current joint values, moves to position
		"""
		joint_state = self.joint_state
		joint_dict = dict(zip(joint_state.name, joint_state.position))
		for i in range(len(JOINT_NAMES)):
			joint_dict[JOINT_NAMES[i]] += joints[i]
		joint_state = JointState()
		joint_state.name = JOINT_NAMES
		joint_goal = [joint_dict[joint] for joint in JOINT_NAMES]
		joint_goal = np.clip(np.array(joint_goal), JOINT_MIN, JOINT_MAX)
		joint_state.position = joint_goal
		header = Header()
		robot_state = RobotState()
		robot_state.joint_state = joint_state
		link_names = ['gripper_rail_link']
		position = self.get_fk_client(header, link_names, robot_state)
		target_p = position[0].pose.position
		x, y, z = target_p.x, target_p.y, target_p.z
		conditions = [
			x <= BOUNDS_LEFTWALL,
			x >= BOUNDS_RIGHTWALL,
			y <= BOUNDS_BACKWALL,
			y >= BOUNDS_FRONTWALL,
			z <= BOUNDS_FLOOR,
			z >= 0.15
		]
		print("Target Position: %0.4f, %0.4f, %0.4f" % (x, y, z))
		for condition in conditions:
			if not condition:
				return
		self.move_to_target(joint_goal)
		rospy.sleep(0.15)
		
	def move_to_vertical(self, z, force_orientation=True, shift_factor=1.0):
		current_p = self.commander.get_current_pose().pose
		current_angle = self.get_joint_values()[4]
		orientation = current_p.orientation if force_orientation else None
		p1 = Pose(position=Point(x=current_p.position.x * shift_factor,
								 y=current_p.position.y * shift_factor, z=z), orientation=orientation)
		waypoints = [current_p, p1]
		plan, f = self.commander.compute_cartesian_path(waypoints, 0.001, 0.0)

		if not force_orientation:
			return self.commander.execute(plan, wait=True)
		else:
			if len(plan.joint_trajectory.points) > 0:
				joint_goal = list(plan.joint_trajectory.points[-1].positions)
			else:
				return False

			joint_goal[4] = current_angle

			plan = self.commander.plan(joint_goal)
			return self.commander.execute(plan, wait=True)
	
	def orient_gripper(self, quat=DOWN_ORIENTATION):
		'''
		Orient gripper to given quaternion in the form (x, y, z, w)
		'''
		current = self.commander.get_current_pose().pose
		point = Pose(orientation=quat)

		(plan, fraction) = self.commander.compute_cartesian_path(
			[current, point],   # waypoints to follow
			0.01,        # eef_step
			0.0)         # jump_threshold

		joint_goal = list(plan.joint_trajectory.points[-1].positions)
		
		try:
			plan = self.commander.plan(joint_goal)
		except MoveItCommanderException as e:
			print('Exception while planning')
			traceback.print_exc(e)
			return False

		print(fraction)
		return self.commander.execute(plan, wait=True), fraction

	def move_to_neutral(self):
		print('Moving to neutral...')
		plan = self.commander.plan(PULLED_BACK_VALUES)
		return self.commander.execute(plan, wait=True)

	def move_to_pregrasp(self, angle=None, width=None):
		grasp_positions = PREGRASP_VALUES[:]
		if angle:
			grasp_positions[4] = angle
		if width:
			self.prepare_gripper(width)
		plan = self.commander.plan(grasp_positions)
		return self.commander.execute(plan, wait=True)

	def move_to_initial(self):
		plan = self.commander.plan(INITIAL_VALUES)
		return self.commander.execute(plan, wait=True)

	def move_to_reset(self):
		print('Moving to reset...')
		plan = self.commander.plan(RESET_VALUES)
		return self.commander.execute(plan, wait=True)

	def move_to_drop(self):
		print('Moving to drop position...')
		plan = self.commander.plan(DROP_VALUES)
		return self.commander.execute(plan, wait=True)

	def move_to_grasp(self, x, y, z, angle, compensate_control_noise=True):
		if compensate_control_noise:
			x = (x - CONTROL_NOISE_COEFFICIENT_BETA) / CONTROL_NOISE_COEFFICIENT_ALPHA
			y = (y - CONTROL_NOISE_COEFFICIENT_BETA) / CONTROL_NOISE_COEFFICIENT_ALPHA
		
		current_p = self.commander.get_current_pose().pose
		p1 = Pose(position=Point(x=x, y=y, z=z), orientation=DOWN_ORIENTATION)
		plan, f = self.commander.compute_cartesian_path(
			[current_p, p1], 0.001, 0.0)

		joint_goal = list(plan.joint_trajectory.points[-1].positions)

		first_servo = joint_goal[0]

		joint_goal[4] = (angle - first_servo) % np.pi
		if joint_goal[4] > np.pi / 2:
			joint_goal[4] -= np.pi
		elif joint_goal[4] < -(np.pi / 2):
			joint_goal[4] += np.pi

		try:
			plan = self.commander.plan(joint_goal)
		except MoveItCommanderException as e:
			print('Exception while planning')
			traceback.print_exc(e)
			return False

		return self.commander.execute(plan, wait=True)

	def orient_to_pregrasp(self, x, y, width=None):
		angle = np.arctan2(y, x)
		return self.move_to_pregrasp(angle, width)

	def eval_grasp(self, threshold=.0001, manual=False):
		current = np.array(self.gripper.get_current_joint_values())
		if manual == True:
			user_input = None
			while user_input not in ('y', 'n', 'r'):
				user_input = raw_input('Successful grasp? [(y)es/(n)o/(r)edo]: ')
			if user_input == 'y':
				return current, 1, None
			elif user_input == 'n':
				return current, 0, None
			else:
				return current, -1, None
		else:
			target = np.array(GRIPPER_CLOSED)
			error = current[0] - target[0]
			return error > threshold, error

	def wrist_rotate(self, angle):
		rotated_values = self.commander.get_current_joint_values()
		rotated_values[4] = angle - rotated_values[0]
		if rotated_values[4] > np.pi / 2:
			rotated_values[4] -= np.pi
		elif rotated_values[4] < -(np.pi / 2):
			rotated_values[4] += np.pi
		plan = self.commander.plan(rotated_values)
		return self.commander.execute(plan, wait=True)
		
	def cartesian_point_generator(self):
		'''
		Generate a random valid point in space based on the current planning space joint limit
		Returns: Cartesian point (numpy array and torch tensor)
		'''
		commander = self.commander
		planning_frame = self.planning_frame

		current_pos = commander.get_current_pose().pose.position
		current_joint = commander.get_current_joint_values()

		x = np.random.uniform(low = BOUNDS_BACKWALL, high = BOUNDS_FRONTWALL)
		y = np.random.uniform(low = BOUNDS_RIGHTWALL, high = BOUNDS_LEFTWALL)
		z = np.random.uniform(low = BOUNDS_FLOOR+0.1, high = BOUNDS_FLOOR+0.2)
		sin = np.random.uniform(low = -1.5, high = 1.5)

		random_point = np.array((x, y, z, sin), dtype = np.float64)

		return random_point

	def param_generator(self, x_scaling=None, y_scaling=None):
		'''
		Generate small vector movements for testing
		'''
		commander = self.commander
		current_pos = commander.get_current_pose().pose.position

		x = np.random.uniform(low = -0.1, high = 0.1)
		if x_scaling is not None:
			scaled_x = (x * 3) + x_scaling
		y = np.random.uniform(low = -0.05, high = 0.05)
		if y_scaling is not None:
			scaled_y = (y * 3) + y_scaling
		z = np.random.uniform(low = -0.1, high = -0.08)
		sin = np.random.uniform(low = -1, high = 1)
		param_numpy = np.array((x, y, z, sin, current_pos.z+z), dtype = np.float32)
		params = torch.from_numpy(param_numpy)
		if x_scaling or y_scaling is not None:
			scaled_numpy = np.array((scaled_x, scaled_y, z, sin, current_pos.z+z), dtype = np.float32)
	
			params = torch.from_numpy(scaled_numpy)
			scaled_tensor = torch.unsqueeze(params, 0)
			return param_numpy, scaled_tensor
		param_tensor = torch.unsqueeze(params, 0)

		return param_numpy, param_tensor

def main():
	try:
		robot = WidowX()
		pose = robot.get_current_pose().pose
		#for debuggin to control the gripper directly
		import ipdb; ipdb.set_trace()
	except rospy.ROSInterruptException:
		return
  	except KeyboardInterrupt:
		return

if __name__ == '__main__':
	main()
