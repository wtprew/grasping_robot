import argparse
from utils.dataset_processing.demo import GraspGenerator

def parse_args():
	parser = argparse.ArgumentParser(description='Run grasp generator')

	parser.add_argument('-d', '--description', type=str, default='', help='Experiment description')
	parser.add_argument('-o', '--object_id', type=str, default='', help='Object code or description')	
	parser.add_argument('--outdir', type=str, default='results', help='Experiment directory')

	parser.add_argument('--cam-id', type=int, default=831612070538, help='Camera ID')
	parser.add_argument('--output_size', type=int, default=320)
	parser.add_argument('--use-depth', type=bool, default=True, help='Use Depth image for model (1/0)')
	parser.add_argument('--use-rgb', type=bool, default=True, help='Use RGB image for model (0/1)')
	parser.add_argument('--use-width', action='store_true', help='Use width output to control gripper width for grasping')
	parser.add_argument('--width_scaling', type=int, default=30, help='max gripper width')
	parser.add_argument('--bins', type = int, default=1, help='number of output bins to consider')
	parser.add_argument('--network', default='gr', type=str, help='model to load weights to')
	parser.add_argument('--network-path', type=str, default='~/grasping+robot/trained_models/GRConv_RGBD_Pos_1bin/epoch_33_iou_0.91', help='Path to trained network')
	parser.add_argument('-s', '--start', type=int, default=0, help='Starting grasp attempt')
	parser.add_argument('-a', '--attempts', type=int, default=20, help='Number of cycles to try')
	parser.add_argument('-v', '--vis', action='store_true', help='Visualise output')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	generator = GraspGenerator(
		description=args.description,
		results=args.outdir,
		cam_id=args.cam_id,
		output_size=args.output_size,
		saved_model_path=args.network_path,
		object_id=args.object_id,
		use_rgb=args.use_rgb,
		use_depth=args.use_depth,
		start=args.start,
		attempts=args.attempts,
		use_width=args.use_width,
		width_scaling=args.width_scaling,
		bins = args.bins,
		visualise=args.vis
	)
	if args.network is not None:
		generator.load_model_dict(args.network)
	else:
		generator.load_model()
	generator.run()
