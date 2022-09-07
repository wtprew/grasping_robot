import argparse
import logging
from matplotlib import gridspec

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from utils.dataset_processing.post_process import post_process_output
from utils.dataset_processing.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results

from models.grconvnet_bins import GenerativeResnet
from utils.dataset_processing.grasp import detect_grasps

logging.basicConfig(level=logging.INFO)


def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate network')
	parser.add_argument('--network', type=str, default='~/grasping_robot/trained_models/GRConv_RGBD_Pos_1bin/epoch_33',
						help='Path to saved network to evaluate')
	parser.add_argument('--use-depth', type=int, default=1,
						help='Use Depth image for evaluation (1/0)')
	parser.add_argument('--use-rgb', type=int, default=1,
						help='Use RGB image for evaluation (1/0)')
	parser.add_argument('--n-grasps', type=int, default=1,
						help='Number of grasps to consider per image')
	parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
						help='Force code to run in CPU mode')

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_args()

	# Connect to Camera
	logging.info('Connecting to camera...')
	cam = RealSenseCamera(device_id=831612070538)
	cam.connect()
	cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

	# Load Network
	logging.info('Loading model...')
	# model = GenerativeResnet(4, 1)
	# import ipdb; ipdb.set_trace()
	# net = model.load_state_dict(torch.load(args.network)['model_state_dict'])
	net = torch.load(args.network)
	logging.info('Done')

	# Get the compute device
	device = get_device(args.force_cpu)

	try:
		fig = plt.figure()
		gs = gridspec.GridSpec(2, 4)
		gs.tight_layout(fig)

		plt.ion()

		while True:
			image_bundle = cam.get_image_bundle()
			rgb = image_bundle['rgb']
			depth = image_bundle['aligned_depth']
			x, depth_img, rgb_img = cam_data.get_data(rgb=rgb, depth=depth)
			with torch.no_grad():
				xc = x.to(device)
				pred = net.predict(xc)

				plt.clf()
				q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])

				# plot_results(fig=fig,
				# 			 rgb_img=cam_data.get_rgb(rgb, False),
				# 			 depth_img=np.squeeze(cam_data.get_depth(depth)),
				# 			 grasp_q_img=q_img,
				# 			 grasp_angle_img=ang_img,
				# 			 no_grasps=args.n_grasps,
				# 			 grasp_width_img=width_img)
				
				grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=1)
				
				grasp_ax = fig.add_subplot(gs[0:2, 0:2])
				grasp_ax.set_title('Grasp')
				grasp_ax.axis('off')
				
				depth_ax = fig.add_subplot(gs[0,2])
				depth_ax.set_title('Depth')
				depth_ax.axis('off')
				
				q_ax = fig.add_subplot(gs[0, 3])
				q_ax.set_title('Q')
				q_ax.axis('off')
		
				ang_ax = fig.add_subplot(gs[1, 2])
				ang_ax.set_title('Angle')
				ang_ax.axis('off')
				
				w_ax = fig.add_subplot(gs[1,3])
				w_ax.set_title('Width')
				w_ax.axis('off')
				
				grasp_ax.imshow(cam_data.get_rgb(rgb, False))
 				
				for g in grasps:
					g.plot(grasp_ax)

				depth_ax.imshow(np.squeeze(cam_data.get_depth(depth)), cmap='gray')

				q_map = q_ax.imshow(q_img, cmap='Reds', vmin=0, vmax=1)
				# fig.colorbar(q_map, ax=q_ax)
		
				ang_map = ang_ax.imshow(ang_img, cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
				# fig.colorbar(ang_map, ax=ang_ax)
		
				w_map = w_ax.imshow(width_img, cmap='Reds', vmin=0, vmax=150)
				# fig.colorbar(w_map, ax=w_ax)

				plt.pause(0.1)
				fig.canvas.draw()
	finally:
		save_results(
			rgb_img=cam_data.get_rgb(rgb, False),
			depth_img=np.squeeze(cam_data.get_depth(depth)),
			grasp_q_img=q_img,
			grasp_angle_img=ang_img,
			no_grasps=args.n_grasps,
			grasp_width_img=width_img
		)
