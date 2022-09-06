import os
import numpy as np
import rospy
import argparse

from clickcontrol import ClickControl
from sklearn.linear_model import LinearRegression
from scipy import optimize

# def compute_calibration(robot_points, camera_points):
#     lr = LinearRegression().fit(camera_points, robot_points)
#     predicted = lr.predict(camera_points)
#     residuals = np.abs(predicted - robot_points)

#     co = lr.coef_
#     trans = lr.intercept_
#     tf_matrix = np.matrix([[co[0, 0],	co[0, 1],	co[0, 2],	trans[0]],
#                            [co[1, 0],	co[1, 1],	co[1, 2],	trans[1]],
#                            [co[2, 0],	co[2, 1],	co[2, 2],	trans[2]],
#                            [0.0,		0.0,		0.0,		1.0		]])
#     return tf_matrix, residuals

class Calibration:
	def __init__(self, cam_id, execute, calibrate):

		self.cam_id = cam_id
		self.execute = execute
		self.calibrate = calibrate

		self.measured_pts = []
		self.observed_pts = []
		self.observed_pix = []
		self.world2camera = np.eye(4)

		self.homedir = "/home/capture/ros_ws/intrinsics"
		self.move_completed = os.path.join(self.homedir, "move_completed.npy")
		self.tool_position = os.path.join(self.homedir, "tool_position.npy")

	@staticmethod
	def get_rigid_transform(A, B):
		"""
		Estimate rigid transform with SVD (from Nghia Ho)
		"""
		assert len(A) == len(B)
	
		N = A.shape[0]  # Total points
		centroid_A = np.mean(A, axis=0)
		centroid_B = np.mean(B, axis=0)
		AA = A - np.tile(centroid_A, (N, 1))  # Centre the points
		BB = B - np.tile(centroid_B, (N, 1))
		H = np.dot(np.transpose(AA), BB)  # Dot is matrix multiplication for array
		U, S, Vt = np.linalg.svd(H)
		R = np.dot(Vt.T, U.T)
		if np.linalg.det(R) < 0:  # Special reflection case
			Vt[2, :] *= -1
			R = np.dot(Vt.T, U.T)
		t = np.dot(-R, centroid_A.T) + centroid_B.T
		return R, t

	def compute_calibration(self, robot_points, camera_points):
		R, t = self.get_rigid_transform(robot_points, camera_points)
		t = t.reshape(3, 1)
		world2camera = np.concatenate((np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
		print('Origin point to camera:', world2camera)
		camera_pose = np.asarray(np.linalg.inv(world2camera))
		return camera_pose

	def run(self):
		# rospy.init_node("calibration")

		executor = ClickControl(cam_id=self.cam_id, execute=self.execute, calibrate=self.calibrate)
		print('Please close the window with "q" to finish gathering correspondences')
		executor.run()

		rospy.sleep(2)
		
		self.measured_pts = executor.measured_pts
		self.observed_pts = executor.observed_pts
		self.observed_pix = executor.observed_pix

		calibration_matrix = self.compute_calibration(np.asarray(self.measured_pts), np.asarray(self.observed_pts))

		print('Calibration matrix:')
		print(np.asarray(calibration_matrix))
		np.save(os.path.join(self.homedir, 'camera_pose.npy'), calibration_matrix)
