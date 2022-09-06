#!/usr/bin/env python2

from __future__ import absolute_import, division, print_function

import argparse
import csv
import math
import os
import pdb
import random
import sys
import time
from datetime import datetime

import cv2
import message_filters
import numpy as np
import roslib
import rospy
import scipy
import sensor_msgs
import tf2_ros
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Quaternion
from rostopic import get_topic_type
from sensor_msgs.msg import Image
from tf import transformations
import matplotlib.pyplot as plt

import executor
from models.model import *
from grasping.config import *
from grasping.utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")	

path = DATAPATH

def main(datapath, save, model, manual, sweep, start, samples):
	# try:
	# rospy.init_node('data_collection')
	counter = 0
	ex = executor.Executor(scan=True, datapath=args.datapath, save=args.save)
	
	ex.widowx.move_to_neutral()
	ex.widowx.open_gripper()

	use_cuda = torch.cuda.is_available()
	print("Cuda available: " + str(use_cuda))
	device = torch.device("cuda" if use_cuda else "cpu")	
	
	if args.model == "LevineNet":
		model = LevineNet().to(device)
		sd = torch.load('/media/will/research/grasp/models/RGB-SD.pt')
		model.load_state_dict(sd)
		model.eval()
	
	sample_id = args.start

	if args.sweep == True:
		ex.widowx.sweep_arena()
		ex.widowx.move_to_neutral()

	while sample_id < args.start + args.samples:
		print('Grasp %d' % sample_id)

		initial, step = ex.pregrasp()
		# print(initial.shape, step.shape)
		
		starting_point = ex.widowx.get_current_pose()

		initial = transform(initial, 472).to(device)
		step = transform(step, 472).to(device)
		# print(initial.shape, step.shape)

		# param generation and model evaluation // to do
		best_prob = 0
		chosen_vector = np.zeros(5)
		scaled_tensor = torch.zeros(5)

		for i in range(args.start, args.start+50):
			param_numpy, param_tensor = ex.widowx.param_generator(x_scaling=-0.24, y_scaling=0.14)
			param_tensor = param_tensor.to(device)
			run_prob = model(initial, step, param_tensor)
			if run_prob.cpu().detach().numpy().item() > best_prob:
				chosen_vector = param_numpy
				scaled_tensor = param_tensor
				best_prob = run_prob.cpu().detach().numpy().item()

		print(chosen_vector, "\n", scaled_tensor, "\n", best_prob)
		grasp = [chosen_vector[0], chosen_vector[1], chosen_vector[2], chosen_vector[3]]
		sample = ex.execute_grasp(grasp, manual=args.manual)

		end_point = ex.widowx.get_current_pose()

		if starting_point != end_point:
			if args.save:
				ex.save_sample(sample_id)

		sample_id +=1
		counter +=1
		rospy.sleep(1)
	
	# except:
	# 	print('failed pregrasp')

	# try:
	# 	rospy.spin()
	# except rospy.ROSInterruptException:
	# 	return
	# except KeyboardInterrupt:
	# 	return

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--datapath', type=str, default=path)
	parser.add_argument('--save', type=bool, default=True)
	parser.add_argument('--model', help='model to load',
						default='LevineNet', type=str)
	parser.add_argument('--manual', help='manual labelling of grasps', 
						default=False, type=bool)
	parser.add_argument('--sweep', type=bool, default=False)
	parser.add_argument('--start', type=int, default=0)
	parser.add_argument('--samples', type=int, default=100)

	args = parser.parse_args()

	print('Called with args: \n', args)

	main(args.datapath, args.save, args.model, args.manual, args.sweep, args.start, args.samples)
