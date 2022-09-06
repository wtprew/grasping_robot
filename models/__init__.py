def get_network(network_name):
	network_name = network_name.lower()
	if network_name == 'ggcnn':
		from .ggcnn import GGCNN
		return GGCNN
	elif network_name == 'mtgcnn':
		from .mtgcnn import MTGCNN
		return MTGCNN
	elif network_name == 'ggcnn2':
		from .ggcnn2 import GGCNN2
		return GGCNN2
	elif network_name == 'mtgcnn2':
		from .mtgcnn2 import MTGCNN2
		return MTGCNN2
	# Original GR-ConvNet
	elif network_name == 'grconvnet':
		from .grconvnet import GenerativeResnet
		return GenerativeResnet
	# Configurable GR-ConvNet with multiple dropouts
	elif network_name == 'grconvnet2':
		from .grconvnet2 import GenerativeResnet
		return GenerativeResnet
	# Configurable GR-ConvNet with dropout at the end
	elif network_name == 'grconvnet3':
		from .grconvnet3 import GenerativeResnet
		return GenerativeResnet
	# Inverted GR-ConvNet
	elif network_name == 'grconvnet4':
		from .grconvnet4 import GenerativeResnet
		return GenerativeResnet
	else:
		raise NotImplementedError('Network {} is not implemented'.format(network_name))
