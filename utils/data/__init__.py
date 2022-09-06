def get_dataset(dataset_name):
	dataset_name = str(dataset_name).lower()
	if dataset_name == 'cornell':
		from .cornell_data import CornellDataset
		return CornellDataset
	elif dataset_name == 'jacquard':
		from .jacquard_data import JacquardDataset
		return JacquardDataset
	elif dataset_name == 'cornell2':
		from .cornell_data_depth import CornellDataset
		return CornellDataset
	elif dataset_name == 'jacquard2':
		from .jacquard_data_depth import JacquardDataset
		return JacquardDataset
	elif dataset_name == 'jacquard_retrain':
		from .jacquard_data_mask import JacquardDataset
		return JacquardDataset
	else:
		raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))