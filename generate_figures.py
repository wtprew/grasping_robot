import argparse
import os

import cv2
from PIL import Image
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.gridspec import GridSpec

import numpy as np
from scipy.ndimage.measurements import label
import torch
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, RocCurveDisplay
from skimage.draw import polygon
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.transform import rotate, resize

from src.tools.visualisation import grasp
from src.models.ggcnn import GGCNN2
from src.models.grconvnet import GenerativeResnet
from src.models.grconvnet3 import GenerativeResnet3
from src.models.unet import UNet

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--gt', action='store_true'
	)
	parser.add_argument(
		'--output', action='store_true'
	)
	parser.add_argument(
		'--gt_example', action='store_true'
	)
	parser.add_argument(
		'--binary_gt_example', action='store_true'
	)
	parser.add_argument(
		'--iou', action='store_true'
	)
	parser.add_argument(
		'--net_name', help='name of model to test/visualise',
		type=str, default=''
	)
	parser.add_argument(
		'--net', help='name of net to test/visualise',
		type=str, default=''
	)
	parser.add_argument(
		'--results_path', type=str,
		default='results'
	)
	parser.add_argument(
		'--grasp_output', help='file output name',
		type=str
	)
	parser.add_argument(
		'--gaussian', help='aggressiveness of gaussian drop off',
		type=float, default=2
	)
	parser.add_argument(
		'--compact', help='turn off compact polygon coordinates',
		action='store_false', default=True
	)
	parser.add_argument(
		'--binary', help='turn on binary representation of grasps',
		action='store_true', default=False
	)
	parser.add_argument(
		'--num_of_bins', help='number of bins in network output',
		type=int, default=3
	)
	parser.add_argument(
		'--threshold', help='threshold for IoU',
		type=float, default=0.25
	)
	parser.add_argument(
		'--filter', help='Apply filter to ground truth',
		action='store_true'
	)
	parser.add_argument(
		'--vis', help='whether to visualise results',
		action='store_true'
	)

	parser.add_argument(
		'--include_ang', help='include angle output',
		action='store_true'
	)
	parser.add_argument(
		'--include_width', help='include width output',
		action='store_true'
	)
	parser.add_argument(
		'--include_graspness', help='include grasp output',
		action='store_true'
	)

	return parser.parse_args()

def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1):
	"""
	Detect grasps in a GG-CNN output.
	:param q_img: Q image network output
	:param ang_img: Angle image network output
	:param width_img: (optional) Width image network output
	:param no_grasps: Max number of grasps to return
	:return: list of Grasps
	"""
	local_max = np.array([np.unravel_index(q_img.argmax(), q_img.shape)])

	grasps = []
	for grasp_point_array in local_max:
		grasp_point = tuple(grasp_point_array)

		grasp_angle = ang_img[grasp_point]

		g = grasp.Grasp(grasp_point[-2:], grasp_angle)
		if width_img is not None:
			g.length = width_img[grasp_point]
			g.width = g.length/2

		grasps.append(g)

	return grasps

def polygon_coords(grasp, shape=None, compact=True):
	"""Return pixels within the centre third of the grasp rectangle."""
	center = grasp.mean(axis=0)  # .astype(np.int)
	angle = np.arctan2(-grasp[1, 0] + grasp[0, 0], grasp[1, 1] - grasp[0, 1])
	angle = (angle + np.pi / 2) % np.pi - np.pi / 2
	length = np.sqrt(
		(grasp[1, 1] - grasp[0, 1]) ** 2 + (grasp[1, 0] - grasp[0, 0]) ** 2)
	width = np.sqrt(
		(grasp[2, 0] - grasp[1, 0]) ** 2 + (grasp[2, 1] - grasp[1, 1]) ** 2)
	if compact:
		length = length / 3  # center third

	# Create points
	x_0, y_0 = (np.cos(angle), np.sin(angle))
	y_1 = center[0] + length / 2 * y_0
	x_1 = center[1] - length / 2 * x_0
	y_2 = center[0] - length / 2 * y_0
	x_2 = center[1] + length / 2 * x_0
	points = np.array([
		[y_1 - width / 2 * x_0, x_1 - width / 2 * y_0],
		[y_2 - width / 2 * x_0, x_2 - width / 2 * y_0],
		[y_2 + width / 2 * x_0, x_2 + width / 2 * y_0],
		[y_1 + width / 2 * x_0, x_1 + width / 2 * y_0],
	]).astype(float)
	return polygon(points[:, 0], points[:, 1], shape)

def post_process_output(q_img, cos_img, sin_img, width_img):
	"""Post-process the raw output of the GG-CNN."""
	q_img = q_img.detach().cpu().numpy()
	ang_img = torch.atan2(sin_img, cos_img).detach().cpu().numpy() / 2.0
	width_img = width_img.detach().cpu().numpy() * 150.0

	q_img = np.stack([
		gaussian(img, 2.0, preserve_range=True) for img in q_img
	])
	return q_img, ang_img, width_img

def calculate_iou_match(q_img, angle_img, width_img, ground_truth_bbs, num_of_bins, im_size, jaw_size):
	"""
	Calculate grasp success using the IoU (Jacquard) metric.

	Success: grasp rectangle has a 25% IoU with a ground truth
	and is within 30 degrees.
	"""
	# Find local maximum
	local_max = np.array([np.unravel_index(q_img.argmax(), q_img.shape)])
	if not local_max.tolist():
		local_max = np.array([np.unravel_index(q_img.argmax(), q_img.shape)])
	grasp_point = tuple(local_max[0])

	center, angle, length, width = [
		grasp_point[-2:], angle_img[grasp_point], width_img[grasp_point],
		_compute_jaw_size(width_img[grasp_point], jaw_size)]
	x_0, y_0 = (np.cos(angle), np.sin(angle))
	y_1 = center[0] + length / 2 * y_0
	x_1 = center[1] - length / 2 * x_0
	y_2 = center[0] - length / 2 * y_0
	x_2 = center[1] + length / 2 * x_0
	det = np.array([  # detected shape
		[y_1 - width / 2 * x_0, x_1 - width / 2 * y_0],
		[y_2 - width / 2 * x_0, x_2 - width / 2 * y_0],
		[y_2 + width / 2 * x_0, x_2 + width / 2 * y_0],
		[y_1 + width / 2 * x_0, x_1 + width / 2 * y_0],
	]).astype(np.float)

	# Return max IoU
	return max(100 * iou(det, grasp) for grasp in ground_truth_bbs), det

def _compute_jaw_size(width, jaw_size):
	if jaw_size == 'half':
		return width / 2
	if jaw_size == 'full':
		return width
	return float(jaw_size)

def iou(det, grasp):
	"""Compute IoU between detected and ground-truth grasp."""
	angle = np.arctan2(-grasp[1, 0] + grasp[0, 0], grasp[1, 1] - grasp[0, 1])
	gt_angle = (angle + np.pi / 2) % np.pi - np.pi / 2
	angle = np.arctan2(-det[1, 0] + det[0, 0], det[1, 1] - det[0, 1])
	det_angle = (angle + np.pi / 2) % np.pi - np.pi / 2
	if abs((det_angle - gt_angle + np.pi / 2) % np.pi - np.pi / 2) > np.pi / 6:
		return 0
	rr1, cc1 = polygon(det[:, 0], det[:, 1])
	rr2, cc2 = polygon(grasp[:, 0], grasp[:, 1])
	if not all(itm.tolist() for itm in [rr1, cc1, rr2, cc2]):
		return 0
	r_max = max(rr1.max(), rr2.max()) + 1
	c_max = max(cc1.max(), cc2.max()) + 1
	canvas = np.zeros((r_max, c_max))
	canvas[rr1, cc1] += 1
	canvas[rr2, cc2] += 1
	union = np.sum(canvas > 0)
	if union == 0:
		return 0
	intersection = np.sum(canvas == 2)
	return intersection / union

def draw(grasps, shape, bins, gaussian=2, filter=0, use_filter=False, binary=False, compact=True):
	pos_out = np.zeros((bins, shape[0], shape[1]))
	ang_out = np.zeros((bins, shape[0], shape[1]))
	width_out = np.zeros((bins, shape[0], shape[1]))
	grasp_out = np.zeros(shape)

	for grasp in grasps:
		# Compute polygon and values to fill
		rows, cols = polygon_coords(grasp, shape, compact=compact)
		if not rows.tolist() or not cols.tolist():
			continue
		b_map = np.zeros(shape)  # auxiliary binary map
		b_map[rows, cols] = 1
		angle = np.arctan2(
			-grasp[1, 0] + grasp[0, 0],
			grasp[1, 1] - grasp[0, 1]
		)
		angle = (angle + np.pi / 2) % np.pi
		ang_bin = int(np.floor(bins * angle / np.pi))
		angle -= np.pi / 2
		width = np.sqrt(
			(grasp[1, 1] - grasp[0, 1]) ** 2
			+ (grasp[1, 0] - grasp[0, 0]) ** 2
		)

		# Fill grasp map
		# grasp_out[rows, cols] = 1.0

		# Fill quality map
		if not binary:
			grid_x, grid_y = np.meshgrid(
				np.linspace(-1, 1, max(cols) - min(cols) + 1),
				np.linspace(-1, 1, max(rows) - min(rows) + 1)
			)
			gauss_grid = np.exp(-(grid_x ** 2 + grid_y ** 2) / gaussian)
			if use_filter:
				gauss_grid[(gauss_grid > 0) & (gauss_grid < filter)] = filter
			gauss_map = np.zeros(b_map.shape)
			gauss_map[
				min(rows):max(rows) + 1,
				min(cols):max(cols) + 1
			] = gauss_grid
			gauss_map = gauss_map * b_map
			grasp_out = np.maximum(grasp_out, gauss_map)
			# grasp_out = gauss_map
			pos_out[ang_bin] = np.maximum(pos_out[ang_bin], gauss_map)
		else:
			# Fill grasp map
			grasp_out[rows, cols] = 1.0
			pos_out[ang_bin] = np.maximum(pos_out[ang_bin], b_map)

		# Fill angle map
		ang_out[ang_bin][(b_map > 0) & (ang_out[ang_bin] == 0)] = angle
		ang_out[ang_bin][b_map * ang_out[ang_bin] != 0] = np.minimum(
			ang_out[ang_bin][b_map * ang_out[ang_bin] != 0], angle
		)

		# Fill width map
		width_out[ang_bin][(b_map > 0) & (width_out[ang_bin] == 0)] = width
		width_out[ang_bin] = np.maximum(b_map * width, width_out[ang_bin])
	return (
		pos_out.squeeze(), ang_out.squeeze(),
		width_out.squeeze(), grasp_out
	)

def ground_truth_example(jdir ='/media/will/research/jacquard', use_rgbd_img=True, annos='json_annos/jacquard.json'):
	test_annos = []

	with open(annos) as fid:
		annotations = json.load(fid)

	for anno in annotations:
		if anno['split_id'] == 2:
			test_annos.append(anno)

	txts = '_grasps.txt'

	for i, grs in enumerate(tqdm(test_annos)):
		# if i < 686:
		# if i < 985:
		if i < 2300:
			continue
		object_id = grs['object_label']
		id = grs['id']

		grasps_path = os.path.join(jdir, object_id, id + txts)
		image_path = grasps_path.replace(txts, '_RGB.png')

		with open(grasps_path) as fid:
			txt_annos = np.array([[float(v) for v in line.strip('\n').split(';')] for line in fid])

		im = cv2.imread(image_path)
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		
		# #for zooming
		factor = 0.3

		sr = int(im.shape[0] * (1 - factor)) // 2
		sc = int(im.shape[1] * (1 - factor)) // 2
		orig_shape = im.shape
		im = im[sr:im.shape[0] - sr, sc: im.shape[1] - sc].copy()
		im = resize(im, orig_shape, mode='symmetric', preserve_range=True).astype(im.dtype)

		diff_grasps = defaultdict(list)

		for txt_anno in txt_annos:
			diff_grasps[tuple(txt_anno[:4])].append(txt_anno[4])
		for grasps in diff_grasps:
			diff_grasps[grasps] = max(diff_grasps[grasps])
			txt_annos = np.array([
				list(grasps) + [diff_grasps[grasps]] for grasps in diff_grasps
			])

		center = txt_annos[:, (1, 0)]
		angle = -txt_annos[:, 2] / 180.0 * np.pi
		length = txt_annos[:, 3]
		width = txt_annos[:, 4]
		cos = np.cos(angle)
		sin = np.sin(angle)
		y_1 = center[:, 0] + length / 2 * sin
		x_1 = center[:, 1] - length / 2 * cos
		y_2 = center[:, 0] - length / 2 * sin
		x_2 = center[:, 1] + length / 2 * cos
		grasps = np.stack([
			np.stack([y_1 - width / 2 * cos, x_1 - width / 2 * sin], -1),
			np.stack([y_2 - width / 2 * cos, x_2 - width / 2 * sin], -1),
			np.stack([y_2 + width / 2 * cos, x_2 + width / 2 * sin], -1),
			np.stack([y_1 + width / 2 * cos, x_1 + width / 2 * sin], -1)
		], axis=1)

		fig = plt.figure(figsize=(4,3),constrained_layout=True)
		# gs = GridSpec(1, 2, figure=fig)

		ax1 = fig.add_subplot()
		ax1.axis('off')
		ax1.imshow(im)

		# ax2 = fig.add_subplot(gs[0, 1])
		# ax2.axis('off')
		# ax2.imshow(im)

		T = np.array(
			[
				[1/factor, 0],
				[0, 1/factor]
			]
		)

		for i, g in enumerate(grasps):
			if i > 0:
				break

			max_iou = max(100 * iou(g, grasp) for grasp in grasps)
			g = grasp.GraspRectangle(g)
			g.zoom(factor, (im.shape[0]/2, im.shape[1]/2))
			g = g.as_grasp
			g.plot(ax1, color='green')
			plt.savefig('R2_gt_base.png')
			g.length = g.length/3 # get compact coordinate points
			gr = g.as_gr
			points = gr.points # get polygon corners
			g.length = g.length*3 # revert to regular grasp rectangles			

			gr.as_grasp.plot(ax1, color='blue')
			plt.savefig('R2_gt_center.png')

			# g1 = g.copy()
			# g1.center = points[0] # top left
			# g1.plot(ax2)
			
			g2 = g.copy()
			g2.center = points[1] # top right
			# max_iou = 100 * iou(g2.as_gr.points, g.as_gr.points)
			# print(max_iou)
			g2.plot(ax1, color='red')
			
			# g3 = g.copy()
			# g3.center = points[2] # bottom left
			# g3.plot(ax2)
			
			g4 = g.copy()
			g4.center = points[3] # bottom right
			# max_iou = 100 * iou(g4.as_gr.points, g.as_gr.points)
			# print(max_iou)
			g4.plot(ax1, color='red')


		# plt.savefig('gt.eps', format='eps')
		plt.savefig('R2_gt.png')

		plt.show()

	# for g in alt_grasps:
	# 	if g.max_iou(grasps) > 0.25:
	# 		results['correct'] += 1
	# 	else:
	# 		results['failure'] += 1

def ground_truth_Q_example(jdir ='/media/will/research/jacquard', use_rgbd_img=True, suffix=320, annos='json_annos/jacquard.json'):
	test_annos = []

	with open(annos) as fid:
		annotations = json.load(fid)

	for anno in annotations:
		if anno['split_id'] == 2:
			test_annos.append(anno)

	txts = '_grasps_' + str(suffix) + '.txt'

	for i, grs in enumerate(tqdm(test_annos)):
		if i < 282:
			continue
		object_id = grs['object_label']
		id = grs['id']

		grasps_path = os.path.join(jdir, object_id, id + txts)
		image_path = grasps_path.replace(txts, '_RGB_'+str(suffix)+'.png')
		depth_image_path = grasps_path.replace(txts, '_perfect_depth_'+str(suffix)+'.tiff')

		with open(grasps_path) as fid:
			txt_annos = np.array([[float(v) for v in line.strip('\n').split(';')] for line in fid])

		im = cv2.imread(image_path)
		depth_image = Image.open(depth_image_path)
		depth_im = np.array(depth_image).astype(float)

		fig = plt.figure(figsize=(5, 5), constrained_layout=True)
		gs = GridSpec(3, 3, figure=fig)

		#for zooming
		factor = 0.3

		sr = int(im.shape[0] * (1 - factor)) // 2
		sc = int(im.shape[1] * (1 - factor)) // 2
		orig_shape = im.shape
		im = im[sr:im.shape[0] - sr, sc: im.shape[1] - sc].copy()
		im = resize(im, orig_shape, mode='symmetric', preserve_range=True).astype(im.dtype)

		depth_sr = int(depth_im.shape[0] * (1 - factor)) // 2
		depth_sc = int(depth_im.shape[1] * (1 - factor)) // 2
		orig_shape = depth_im.shape
		depth_im = depth_im[depth_sr:depth_im.shape[0] - depth_sr, depth_sc: depth_im.shape[1] - depth_sc].copy()
		depth_im = resize(depth_im, orig_shape, mode='symmetric', preserve_range=True).astype(depth_im.dtype)

		ax1 = fig.add_subplot(gs[0, 0])
		ax1.axis('off')
		ax1.set_title(r'(a) RGB Input')
		ax1.imshow(im)

		ax2 = fig.add_subplot(gs[0, 1])
		ax2.axis('off')
		ax2.set_title(r'(b) Depth Input')
		ax2.imshow(depth_im)

		ax3 = fig.add_subplot(gs[0, 2])
		ax3.axis('off')
		ax3.set_title(r'(c) Annotated Grasps')
		ax3.imshow(im)

		diff_grasps = defaultdict(list)

		T = np.array(
			[
				[1/factor, 0],
				[0, 1/factor]
			]
		)

		c = (im.shape[0]/2, im.shape[1]/2)

		for txt_anno in txt_annos:
			diff_grasps[tuple(txt_anno[:4])].append(txt_anno[4])

		for grasp in diff_grasps:
			diff_grasps[grasp] = min(diff_grasps[grasp])
			txt_annos = np.array([
				list(grasp) + [diff_grasps[grasp]] for grasp in diff_grasps
			])

		center = txt_annos[:, (1, 0)]
		angle = -txt_annos[:, 2] / 180.0 * np.pi
		length = txt_annos[:, 3]
		width = txt_annos[:, 4]
		cos = np.cos(angle)
		sin = np.sin(angle)
		y_1 = center[:, 0] + length / 2 * sin
		x_1 = center[:, 1] - length / 2 * cos
		y_2 = center[:, 0] - length / 2 * sin
		x_2 = center[:, 1] + length / 2 * cos
		grasps = np.stack([
			np.stack([y_1 - width / 2 * cos, x_1 - width / 2 * sin], -1),
			np.stack([y_2 - width / 2 * cos, x_2 - width / 2 * sin], -1),
			np.stack([y_2 + width / 2 * cos, x_2 + width / 2 * sin], -1),
			np.stack([y_1 + width / 2 * cos, x_1 + width / 2 * sin], -1)
		], axis=1)

		grasps = grasps * suffix / 1024.0

		for i, g in enumerate(grasps):
			grasps[i] = ((np.dot(T, (g - c).T)).T + c)
			points = np.vstack((g, g[0]))
			ax3.plot(points[:, 1], points[:, 0], color='green')

		out_binary = draw(grasps, (im.shape[0], im.shape[1]), bins=3, gaussian=2, filter=True, binary=True, compact=True)
		out_orange = draw(grasps, (im.shape[0], im.shape[1]), bins=3, gaussian=2, filter=True, binary=False, compact=True)
		out_gaussian = draw(grasps, (im.shape[0], im.shape[1]), bins=3, gaussian=0.5, filter=False, binary=False, compact=True)
		#out_nc = draw(grasps, (im.shape[0], im.shape[1]), bins=3, gaussian=0.5, filter=False, binary=False, compact=False)

		# for i in range(0, 3):
		# 	ax = fig.add_subplot(gs[1, i])
		# 	ax.axis('off')
		# 	ob = out_binary[0][i]
		# 	ax.imshow(ob, cmap='Greys', vmin=0, vmax=1)

		# 	ax1 = fig.add_subplot(gs[2, i])
		# 	ax1.axis('off')
		# 	oo = out_orange[0][i]
		# 	ax1.imshow(oo, cmap='Greys', vmin=0, vmax=1)
			
		# 	ax2 = fig.add_subplot(gs[3, i])
		# 	ax2.axis('off')
		# 	og = out_gaussian[0][i]
		# 	ax2.imshow(og, cmap='Greys', vmin=0, vmax=1)
			
		# 	ax3 = fig.add_subplot(gs[4, i])
		# 	ax3.axis('off')
		# 	on = out_nc[0][i]
		# 	ax3.imshow(on, cmap='Greys', vmin=0, vmax=1)

		ax4 = fig.add_subplot(gs[1, 0])
		ax4.axis('off')
		ax4.set_title(r'(d) Binary $Q$')
		ob = out_binary[3]
		ax4.imshow(ob, cmap='Greys', vmin=0, vmax=1)

		ax5 = fig.add_subplot(gs[1, 1])
		ax5.axis('off')
		ax5.set_title(r'(e) Soft $Q$')
		oo = out_orange[3]
		ax5.imshow(oo, cmap='Greys', vmin=0, vmax=1)

		ax6 = fig.add_subplot(gs[1, 2])
		ax6.axis('off')
		ax6.set_title(r'(f) Strong $Q$')
		og = out_gaussian[3]
		ax6.imshow(og, cmap='Greys', vmin=0, vmax=1)

		ax7 = fig.add_subplot(gs[2, 0])
		ax7.axis('off')
		og = out_gaussian[0][0]
		ax7.set_title(r'$0-60^{\circ}$')
		ax7.imshow(og, cmap='Greys', vmin=0, vmax=1)

		ax8 = fig.add_subplot(gs[2, 1])
		ax8.axis('off')
		og = out_gaussian[0][1]
		ax8.set_title(r'$60-120^{\circ}$')
		ax8.imshow(og, cmap='Greys', vmin=0, vmax=1)

		ax9 = fig.add_subplot(gs[2, 2])
		ax9.axis('off')
		og = out_gaussian[0][2]
		ax9.set_title(r'$120-180^{\circ}$')
		ax9.imshow(og, cmap='Greys', vmin=0, vmax=1)

		# ax7 = fig.add_subplot(gs[1, 2])
		# ax7.axis('off')
		# on = out_nc[3]
		# ax7.imshow(on, cmap='Greys', vmin=0, vmax=1)

		plt.savefig('Q_gauss.png')
		plt.savefig('Q_Gaussians.eps', format='eps')
		plt.show()

def show_ground_truth(gaussian=2, jdir ='/media/will/research/jacquard', use_rgbd_img=True, num_of_bins=3, filter=False, suffix=320, annos='json_annos/jacquard.json', binary=False, compact=True):
	test_annos = []

	with open(annos) as fid:
		annotations = json.load(fid)

	for anno in annotations:
		if anno['split_id'] == 2:
			test_annos.append(anno)

	txts = '_grasps_' + str(suffix) + '.txt'

	for grs in test_annos:
		object_id = grs['object_label']
		id = grs['id']

		grasps_path = os.path.join(jdir, object_id, id + txts)
		image_path = grasps_path.replace(txts, '_RGB_'+str(suffix)+'.png')
		depth_image_path = grasps_path.replace(txts, '_perfect_depth_'+str(suffix)+'.tiff')

		with open(grasps_path) as fid:
			txt_annos = np.array([[float(v) for v in line.strip('\n').split(';')] for line in fid])

		im = cv2.imread(image_path)
		depth_image = Image.open(depth_image_path)
		depth_im = np.array(depth_image).astype(float)
		
		diff_grasps = defaultdict(list)

		for txt_anno in txt_annos:
			diff_grasps[tuple(txt_anno[:4])].append(txt_anno[4])
		for grasp in diff_grasps:
			# if jaw_size_policy == 'min':
			diff_grasps[grasp] = min(diff_grasps[grasp])
			# if jaw_size_policy == 'max':
			# diff_grasps[grasp] = max(diff_grasps[grasp])
			txt_annos = np.array([
				list(grasp) + [diff_grasps[grasp]] for grasp in diff_grasps
			])

		center = txt_annos[:, (1, 0)]
		angle = -txt_annos[:, 2] / 180.0 * np.pi
		length = txt_annos[:, 3]
		width = txt_annos[:, 4]
		cos = np.cos(angle)
		sin = np.sin(angle)
		y_1 = center[:, 0] + length / 2 * sin
		x_1 = center[:, 1] - length / 2 * cos
		y_2 = center[:, 0] - length / 2 * sin
		x_2 = center[:, 1] + length / 2 * cos
		grasps = np.stack([
			np.stack([y_1 - width / 2 * cos, x_1 - width / 2 * sin], -1),
			np.stack([y_2 - width / 2 * cos, x_2 - width / 2 * sin], -1),
			np.stack([y_2 + width / 2 * cos, x_2 + width / 2 * sin], -1),
			np.stack([y_1 + width / 2 * cos, x_1 + width / 2 * sin], -1)
		], axis=1)

		grasps = grasps * suffix / 1024.0

		out = draw(grasps, (im.shape[0], im.shape[1]), bins=num_of_bins, gaussian=gaussian, filter=filter, binary=binary, compact=compact)

		fig = plt.figure(constrained_layout=True)
		gs = GridSpec(1, 2, figure=fig)

		ax = fig.add_subplot(gs[0, 0])
		ax.axis('off')
		ax.imshow(im)

		ax = fig.add_subplot(gs[0, 1])
		ax.axis('off')
		ax.imshow(depth_im)

		cols = 5
		fig1 = plt.figure(constrained_layout=True)
		fig1.suptitle('Ground Truth')
		gs = GridSpec(num_of_bins, cols, figure=fig1)

		if num_of_bins == 1:
			col = 0
			ax = fig1.add_subplot(gs[0, col])
			ax.axis('off')		
			ax.imshow(out[0], cmap='Greys', vmin=0, vmax=1)

			col +=1
			ax1 = fig1.add_subplot(gs[0, col])
			ax1.axis('off')		
			ax1.imshow(out[1], cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
			
			col += 1
			ax2 = fig1.add_subplot(gs[0, col])
			ax2.axis('off')
			ax2.imshow(out[2], cmap='jet')
			
			col += 1
			ax3 = fig1.add_subplot(gs[0, col])
			ax3.axis('off')
			ax3.imshow((out[0] > 0).astype(float), cmap='Greys', vmin=0, vmax=1)

			col += 1
			ax4 = fig1.add_subplot(gs[0, col])
			ax4.axis('off')
			ax4.imshow(out[3], cmap='Greys', vmin=0, vmax=1)
		else:
			for i in range(0, num_of_bins):
				col = 0
				ax = fig1.add_subplot(gs[i, col])
				ax.axis('off')		
				ax.imshow(out[0][i], cmap='Greys', vmin=0, vmax=1)
	
				col +=1
				ax1 = fig1.add_subplot(gs[i, col])
				ax1.axis('off')		
				ax1.imshow(out[1][i], cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
				
				col += 1
				ax2 = fig1.add_subplot(gs[i, col])
				ax2.axis('off')
				ax2.imshow(out[2][i], cmap='jet')
				
				col += 1
				ax3 = fig1.add_subplot(gs[i, col])
				ax3.axis('off')
				ax3.imshow((out[0][i] > 0).astype(float), cmap='Greys', vmin=0, vmax=1)

				if i == 0:	
					col += 1
					ax4 = fig1.add_subplot(gs[0, col])
					ax4.axis('off')
					ax4.imshow(out[3], cmap='Greys', vmin=0, vmax=1)

		plt.show()

def show_network_output(net_name, threshold=0.25, gaussian=2, models_path='models', results_path='results', jdir ='/media/will/research/jacquard', use_rgbd_img=True, num_of_bins=3, filter=False, suffix=320, binary=False, compact=True):
	grasps = {}
	IoU_results = []

	model = 'unet'

	net_path = os.path.join(models_path, net_name, net_name + '.pt')
	print('network path: ', net_path)
	if model == 'unet':
		net = UNet(4 if use_rgbd_img else 1, num_of_bins)
	elif model == 'gr':
		net = GenerativeResnet(4 if use_rgbd_img else 1, num_of_bins)
	checkpoint = torch.load(net_path)
	net.load_state_dict(checkpoint['model_state_dict'], strict=False)
	net.eval()
	
	gfile = os.path.join(results_path, net_name, net_name + '_output.txt')

	if not os.path.exists(gfile):
		gfile = gfile.replace('_output', '_jacquard_output')

	with open(gfile) as g:
		for gr in g:
			object_id = str(gr.strip())
			id = object_id[2:]

			if object_id not in grasps:
				grasps_path = os.path.join(jdir, id, object_id) + '_grasps.txt'
				image_path = grasps_path.replace('_grasps.txt', '_RGB.png')
				depth_image_path = grasps_path.replace('_grasps.txt', '_perfect_depth.tiff')
				
				grasps[object_id] = {'image_file':image_path, 'depth_image_file':depth_image_path, 'grasps_file':grasps_path, 'gt_grasps':[], 'grasps':[], 'IoU_results':[], 'server_results':[]}
			
				gt_grasps = grasp.GraspRectangles.load_from_jacquard_file(grasps_path, scale=suffix/1024)
				grasps[object_id]['gt_grasps'].append(gt_grasps)

			x, y, theta, w, h = [float(v) for v in next(g)[:-1].split(';')]
			gr = grasp.Grasp(np.array([y, x]), -theta/180.0*np.pi, w, h).as_gr
			grasps[object_id]['grasps'].append(gr)
	g.close()

	for object_id in tqdm(grasps):
		for gt in grasps[object_id]['gt_grasps']:

			rgb_img_path = grasps[object_id]['image_file']
			rgb_img_path = rgb_img_path.replace('RGB', 'RGB_'+str(suffix))
			im = cv2.imread(rgb_img_path)
			rgb_img = Image.open(rgb_img_path)
			rgb_net_input = transforms.functional.to_tensor(rgb_img).float()
			rgb_net_input = (rgb_net_input - rgb_net_input.mean())/255
			
			depth_img_path = grasps[object_id]['depth_image_file']
			depth_img_path = depth_img_path.replace('depth', 'depth_'+str(suffix))
			depth_img = Image.open(depth_img_path)
			depth_net_input = transforms.functional.to_tensor(depth_img).float()
			depth_net_input = torch.clamp(depth_net_input - depth_net_input.mean(), -1, 1)

			img = torch.cat([depth_net_input, rgb_net_input], dim=0).unsqueeze(0)
			pos_out, cos_out, sin_out, width_out, grasp_out, bins = net(img)

			q_img, ang_img, width_img = post_process_output(pos_out, cos_out, sin_out, width_out)

			if len(q_img.shape) == 4:
				q_img = q_img.squeeze(0)
				ang_img = ang_img.squeeze(0)
				width_img = width_img.squeeze(0)
				grasp_out = grasp_out
				bins = bins.squeeze(0)

			g = detect_grasps(q_img, ang_img, width_img)
			g = g[0]


			out = gt.draw((im.shape[0], im.shape[1]), binary_map=binary, gaussian=gaussian, bins=num_of_bins, filter=filter, compact=compact)
			# out = draw(grasps, (im.shape[0], im.shape[1]), bins=num_of_bins, gaussian=gaussian, filter=filter, binary=binary, compact=compact)

			rows = 4
			fig1 = plt.figure(constrained_layout=True)
			fig1.suptitle('Ground Truth')
			fig2 = plt.figure(constrained_layout=True)
			fig2.suptitle('Network Output')
			gs = GridSpec(num_of_bins, rows, figure=fig1)
			gs1 = GridSpec(num_of_bins, rows, figure=fig2)

			for i in range(0, num_of_bins):
				row = 0
				ax1 = fig1.add_subplot(gs[i, 0])
				ax1.axis('off')		
				ax1.imshow(out[0][i], cmap='Greys', vmin=0, vmax=1)

				ax2 = fig2.add_subplot(gs1[i, 0], sharey=ax1)
				ax2.axis('off')
				ax2.imshow(q_img[i], cmap='Greys', vmin=0, vmax=1)

				row +=1
				ax3 = fig1.add_subplot(gs[i, row])
				ax3.axis('off')		
				ax3.imshow(out[1][i], cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
				
				ax4 = fig2.add_subplot(gs1[i, row], sharey=ax3)
				ax4.axis('off')
				ax4.imshow(ang_img[i], cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)
				# ax4.imshow(q_img[i]*ang_img[i], cmap='hsv', vmin=-np.pi/2, vmax=np.pi/2)

				row += 1
				ax5 = fig1.add_subplot(gs[i, row])
				ax5.axis('off')
				ax5.imshow(out[2][i], cmap='jet', vmin=0, vmax=150)

				ax6 = fig2.add_subplot(gs1[i, row], sharey=ax5)
				ax6.axis('off')
				ax6.imshow(width_img[i], cmap='jet', vmin=0, vmax=150)
				
				row += 1
				ax9 = fig1.add_subplot(gs[i, row])
				ax9.axis('off')
				ax9.imshow((out[0][i] > 0).astype(float), cmap='Greys', vmin=0, vmax=1)

				ax10 = fig2.add_subplot(gs1[i, row])
				ax10.axis('off')
				ax10.imshow(torch.sigmoid(bins.detach()).cpu().numpy()[i], cmap='Greys', vmin=0, vmax=1)

			# fig3 = plt.figure(constrained_layout=True)
			# fig3.suptitle('Graspness')
			# gs2 = GridSpec(1, 2, figure=fig3)
			# ax7 = fig3.add_subplot(gs2[0, 0])
			# ax7.set_title('Ground Truth')
			# ax7.axis('off')
			# ax7.imshow(out[3], cmap='Greys', vmin=0, vmax=1)
			
			# ax8 = fig3.add_subplot(gs2[0, 1], sharey=ax7)
			# ax8.set_title('Output')
			# ax8.axis('off')
			# ax8.imshow(torch.sigmoid(grasp_out.detach()).cpu().numpy()[0], cmap='Greys', vmin=0, vmax=1)

			fig = plt.figure()

			ax = fig.add_subplot(111)
			ax.set_title('Grasp')
			ax.axis('off')
			ax.imshow(im)

			g.plot(ax)

			max_iou = g.max_iou(gt)
			if max_iou > threshold:
				s = 1
			else:
				s = 0

			x = g.x
			y = g.y
			
			if s == 0:
				ax.scatter(x,y, color='red')
			else:
				ax.scatter(x,y, color='green')

			plt.show()

def separate_network_outputs(net_name, net, jdir ='/media/will/research/jacquard', use_rgbd_img=True, annos='json_annos/jacquard.json', models_path='models', num_of_bins=3):
	test_annos = []

	net_path = os.path.join(models_path, net, net_name, net_name + '.pt')
	print('network path: ', net_path)
	if net == 'unet':
		net = UNet(4 if use_rgbd_img else 1, num_of_bins)
	elif net == 'gr':
		net = GenerativeResnet(4 if use_rgbd_img else 1, num_of_bins)
	checkpoint = torch.load(net_path)
	net.load_state_dict(checkpoint['model_state_dict'], strict=False)
	net.eval()

	with open(annos) as fid:
		annotations = json.load(fid)

	for anno in annotations:
		if anno['split_id'] == 2:
			test_annos.append(anno)

	#for zooming
	factor = 0.3

	T = np.array(
		[
			[1/factor, 0],
			[0, 1/factor]
		]
	)

	txts = '_grasps_320.txt'

	for i, grs in enumerate(tqdm(test_annos)):
		# if i < 686:
		if i < 2300:
			continue

		object_id = grs['object_label']
		id = grs['id']

		grasps_path = os.path.join(jdir, object_id, id + txts)
		image_path = grasps_path.replace(txts, '_RGB.png')

		im = cv2.imread(image_path)
		im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

		image_path = image_path.replace('RGB', 'RGB_320')

		rgb_img = Image.open(image_path)
		
		rgb_net_input = transforms.functional.to_tensor(rgb_img).float()
		rgb_net_input = (rgb_net_input - rgb_net_input.mean())/255
		
		depth_img_path = image_path.replace('RGB_320.png', 'perfect_depth_320.tiff')
		depth_img = Image.open(depth_img_path)
		depth_im = np.array(depth_img).astype(float)
		depth_net_input = transforms.functional.to_tensor(depth_img).float()
		depth_net_input = torch.clamp(depth_net_input - depth_net_input.mean(), -1, 1)

		sr = int(im.shape[0] * (1 - factor)) // 2
		sc = int(im.shape[1] * (1 - factor)) // 2
		orig_shape = im.shape
		im = im[sr:im.shape[0] - sr, sc: im.shape[1] - sc].copy()
		im = resize(im, orig_shape, mode='symmetric', preserve_range=True).astype(im.dtype)

		depth_sr = int(depth_im.shape[0] * (1 - factor)) // 2
		depth_sc = int(depth_im.shape[1] * (1 - factor)) // 2
		orig_shape = depth_im.shape
		depth_im = depth_im[depth_sr:depth_im.shape[0] - depth_sr, depth_sc: depth_im.shape[1] - depth_sc].copy()
		depth_im = resize(depth_im, orig_shape, mode='symmetric', preserve_range=True).astype(depth_im.dtype)

		img = torch.cat([depth_net_input, rgb_net_input], dim=0).unsqueeze(0)
		pos_out, cos_out, sin_out, width_out, grasp_out, bins = net(img)

		fig1 = plt.figure(constrained_layout=True)
		ax1 = fig1.add_subplot()
		ax1.axis('off')
		ax1.imshow(im)
		fig1.savefig('Paper/figures/RGB_input.png')

		fig2 = plt.figure(constrained_layout=True)
		ax = fig2.add_subplot()
		ax.axis('off')
		ax.imshow(depth_im)
		fig2.savefig('Paper/figures/Depth_input.png')

		q_img, ang_img, width_img = post_process_output(pos_out, cos_out, sin_out, width_out)

		local_max = np.array([np.unravel_index(q_img.argmax(), q_img.shape)])
		if not local_max.tolist():
			local_max = np.array([np.unravel_index(q_img.argmax(), q_img.shape)])
		grasp_point = tuple(local_max[0])

		center, angle, length, width = [
			grasp_point[-2:], ang_img[grasp_point], width_img[grasp_point],
			_compute_jaw_size(width_img[grasp_point], 'half')]
		x_0, y_0 = (np.cos(angle), np.sin(angle))
		y_1 = center[0] + length / 2 * y_0
		x_1 = center[1] - length / 2 * x_0
		y_2 = center[0] - length / 2 * y_0
		x_2 = center[1] + length / 2 * x_0
		det = np.array([  # detected shape
			[y_1 - width / 2 * x_0, x_1 - width / 2 * y_0],
			[y_2 - width / 2 * x_0, x_2 - width / 2 * y_0],
			[y_2 + width / 2 * x_0, x_2 + width / 2 * y_0],
			[y_1 + width / 2 * x_0, x_1 + width / 2 * y_0],
		]).astype(float)

		det = det*1024/320

		c = np.array((1024/2, 1024/2)).reshape(1, 2)
		grasp = (np.dot(T, (det - c).T).T + c).astype(int)

		fig3 = plt.figure(constrained_layout=True)
		ax = fig3.add_subplot()
		ax.axis('off')
		ax.imshow(im)
		grasp = np.vstack([grasp, grasp[0]])
		ax.plot(grasp[:, 1], grasp[:, 0], color='green')
		fig3.savefig('Paper/figures/Grasp_det.png')

		# for i in range(0, num_of_bins):
		# 	fig3 = plt.figure(constrained_layout=True)
		# 	ax = fig3.add_subplot()
		# 	ax.axis('off')		
		# 	ax.imshow(q_img[0][i], cmap='Reds')
		# 	fig3.savefig('Paper/figures/Q_'+str(i)+'_out.png')

		# 	fig4 = plt.figure(constrained_layout=True)
		# 	ax1 = fig4.add_subplot()
		# 	ax1.axis('off')
		# 	ax1.imshow(ang_img[0][i], cmap='RdBu', vmin=-np.pi/2, vmax=np.pi/2)
		# 	fig4.savefig('Paper/figures/ang_'+str(i)+'_out.png')

		# 	fig5 = plt.figure(constrained_layout=True)
		# 	ax2 = fig5.add_subplot()
		# 	ax2.axis('off')
		# 	ax2.imshow(width_img[0][i], cmap='Reds')
		# 	fig5.savefig('Paper/figures/width_'+str(i)+'_out.png')
			
		# 	fig6 = plt.figure(constrained_layout=True)
		# 	ax3 = fig6.add_subplot()
		# 	ax3.axis('off')
		# 	ax3.imshow(cos_out[0][i].detach().cpu().numpy(), cmap='hsv')
		# 	fig6.savefig('Paper/figures/cos_'+str(i)+'_out.png')

		# 	fig7 = plt.figure(constrained_layout=True)
		# 	ax4 = fig7.add_subplot()
		# 	ax4.axis('off')
		# 	ax4.imshow(sin_out[0][i].detach().cpu().numpy(), cmap='hsv')
		# 	fig7.savefig('Paper/figures/sin_'+str(i)+'_out.png')

			# ax3 = fig1.add_subplot(gs[i, col])
			# ax3.axis('off')
			# ax3.imshow((out[0][i] > 0).astype(float), cmap='Greys', vmin=0, vmax=1)

			# if i == 0:	
				# col += 1
				# ax4 = fig1.add_subplot(gs[0, col])
				# ax4.axis('off')
				# ax4.imshow(out[3], cmap='Greys', vmin=0, vmax=1)

		# plt.savefig('gt.eps', format='eps')
		# plt.savefig('.png')

		plt.show()

def iou_threshold(models_path='models', results_path='results', jdir ='/media/will/research/jacquard', use_rgbd_img=True, suffix=320, annos='json_annos/jacquard.json'):

	fig = plt.figure()
	
	ax = fig.add_subplot(111)
	ax.set_title('UNet IoU threshold')
	# ax.axis('off')

	plt.ylabel('Accuracy')
	plt.xlabel('Threshold')

	test_annos = []

	with open(annos) as fid:
		annotations = json.load(fid)

	for anno in annotations:
		if anno['split_id'] == 2:
			test_annos.append(anno)

	txts = '_grasps_' + str(suffix) + '.txt'

	net_names = []

	#net_names.append('unet_rgbd_3bin')
	net_names.append('unet_rgbd_3bin_positional')
	#net_names.append('unet_rgbd_3bin_binary')
	net_names.append('unet_rgbd_3bin_binary_positional')

	for i in range(0,2):
		net_name = net_names[i]
		net_path = os.path.join(models_path, net_name, net_name + '.pt')
		print('network path: '+str(i+1), net_path)
		net = UNet(4, 3)
		checkpoint = torch.load(net_path)
		net.load_state_dict(checkpoint['model_state_dict'], strict=False)
		net.eval()
		net.cuda()

		detected = np.zeros(101)
		total_grasps = 0

		for grs in tqdm(test_annos):
			object_id = grs['object_label']
			id = grs['id']

			grasps_path = os.path.join(jdir, object_id, id + txts)
			image_path = grasps_path.replace(txts, '_RGB_'+str(suffix)+'.png')
			depth_image_path = grasps_path.replace(txts, '_perfect_depth_'+str(suffix)+'.tiff')

			diff_grasps = defaultdict(list)

			with open(grasps_path) as fid:
				txt_annos = np.array([[float(v) for v in line.strip('\n').split(';')] for line in fid])
		
			for txt_anno in txt_annos:
				diff_grasps[tuple(txt_anno[:4])].append(txt_anno[4])

			# for grasp in diff_grasps:
				# diff_grasps[grasp] = min(diff_grasps[grasp])
				# txt_annos = np.array([
					# list(grasp) + [diff_grasps[grasp]] for grasp in diff_grasps
				# ])

			center = txt_annos[:, (1, 0)]
			angle = -txt_annos[:, 2] / 180.0 * np.pi
			length = txt_annos[:, 3]
			width = txt_annos[:, 4]
			cos = np.cos(angle)
			sin = np.sin(angle)
			y_1 = center[:, 0] + length / 2 * sin
			x_1 = center[:, 1] - length / 2 * cos
			y_2 = center[:, 0] - length / 2 * sin
			x_2 = center[:, 1] + length / 2 * cos
			grasps = np.stack([
				np.stack([y_1 - width / 2 * cos, x_1 - width / 2 * sin], -1),
				np.stack([y_2 - width / 2 * cos, x_2 - width / 2 * sin], -1),
				np.stack([y_2 + width / 2 * cos, x_2 + width / 2 * sin], -1),
				np.stack([y_1 + width / 2 * cos, x_1 + width / 2 * sin], -1)
			], axis=1)
			grasps = grasps * suffix / 1024.0

			rgb_img = Image.open(image_path)
			# ax.imshow(rgb_img)
			# for grasp in grasps:
			# 	points = np.vstack((grasp, grasp[0]))
			# 	ax.plot(points[:, 1], points[:, 0])
			# plt.show()
			depth_img = Image.open(depth_image_path)

			rgb_net_input = transforms.functional.to_tensor(rgb_img).float()
			rgb_net_input = (rgb_net_input - rgb_net_input.mean())/255
				
			depth_net_input = transforms.functional.to_tensor(depth_img).float()
			depth_net_input = torch.clamp(depth_net_input - depth_net_input.mean(), -1, 1)

			img = torch.cat([depth_net_input, rgb_net_input], dim=0).unsqueeze(0)
			img = img.cuda()
			
			pos_out, cos_out, sin_out, width_out, grasp_out, bins = net(img)
			q_img, ang_img, width_img = post_process_output(pos_out, cos_out, sin_out, width_out)

			max_iou, det = calculate_iou_match(q_img, ang_img, width_img, grasps, 3, suffix, 'half')

			detected[:int(np.floor(max_iou)) + 1] += 1
			total_grasps += 1

		net_acc_per_thres = 100 * detected / total_grasps
		print(net_name)
		print("Accuracy@0.25: %f" % net_acc_per_thres[25])
		print("Accuracy@0.30: %f" % net_acc_per_thres[30])
		print("Accuracy@0.50: %f" % net_acc_per_thres[50])
		print("Accuracy@0.75: %f" % net_acc_per_thres[75])
		print("Avg. Accuracy: %f" % net_acc_per_thres.mean())
		line = plt.plot(net_acc_per_thres)#, label=net_name)
	
	plt.legend(loc='lower left')
	plt.savefig('IoU_threshold.png')
	plt.savefig('IoU_threshold.eps', format='eps')
	plt.show()

if __name__ == '__main__':
	np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
	rc('text', usetex=True)
	rc('font', family='serif')
	args = parse_args()
	if args.gt:
		show_ground_truth(gaussian=args.gaussian, jdir ='/media/will/research/jacquard', use_rgbd_img=True, num_of_bins=args.num_of_bins, filter=args.filter, suffix=320, annos='json_annos/jacquard.json', binary=args.binary, compact=args.compact)
	if args.output:
		# show_network_output(args.net_name, gaussian=args.gaussian, num_of_bins=args.num_of_bins, filter=args.filter, binary=args.binary, compact=args.compact)
		separate_network_outputs(args.net_name, args.net)
	if args.gt_example:
		ground_truth_example()
	if args.binary_gt_example:
		ground_truth_Q_example()
	if args.iou:
		iou_threshold()
