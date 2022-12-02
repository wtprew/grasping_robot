# -*- coding: utf-8 -*-
"""Configuration parameters."""

import logging
import os

from torch.utils.tensorboard import SummaryWriter

from colorlog import ColoredFormatter
import torch

# Directories
PATHS = {
    'json_path': 'json_annos/',  # path of stored json annotations
    'loss_path': 'losses/',  # path to store loss history
    'models_path': 'models/',  # path to store trained models
    'results_path': 'results/'  # path to store test results
}


if os.path.exists('/media/will/research/'):
    DATA_FOLDER = '/media/will/research/'
elif os.path.exists('/home/will/Grasps'):
    DATA_FOLDER = '/home/will/Grasps' # If working on Daisy
else:
    DATA_FOLDER = os.getcwd()

# Variables
USE_CUDA = torch.cuda.is_available()  # whether to use GPU


class Config:
    """
    A class to configure global or training parameters.

    Inputs:
        Dataset/task:
            - dataset: str, dataset name (e.g. jacquard)
            - net_name: str, name of trained model
        Data handling params:
            - handle_as_ggcnn: boolean, handle annotations like GGCNN
            - im_size: int, (always consider square images)
            - jaw_size: float or 'half', jaw size during testing
            - jaw_size_policy: str, pick a jaw size during training
            - num_of_bins: int, angle bins when creating target maps
            - use_binary_map: boolean, binarize quality target map
            - gaussian: int, drop off of gaussian tail for training
            - compact: boolean, use compact polygon coordinates for training
            - use_rgbd_img: boolean, whether to use rgbd as input
        Loss functions:
            - use_smooth_l1_loss: boolean, use smooth l1 instead of mse
            - use_positional_loss: boolean, apply positional losses
            - use_angle_loss: boolean, force cos^2 + sin^2 = 1
            - use_bin_loss: boolean, bin classification loss
            - use_bin_attention_loss: boolean, supervise bin_cls * pos
            - use_graspness_loss: boolean, solve binary task on map
        Training params:
            - batch_size: int, batch size in images
            - learning_rate: float, learning rate of classifier
            - weight_decay: float, weight decay of optimizer
            - use_dropout: boolean, apply dropout to Generative Residual models
            - dropout_prob: float, dropout probability to apply
        Learning rate policy:
            - use_early_stopping: boolean, whether to use a dynammic
                learning rate policy with early stopping
            - restore_on_plateau: boolean, whether to restore checkpoint
                on validation metric's plateaus (only effective in early
                stopping)
            - patience: int, number of epochs to consider a plateau
        General:
            - num_workers: int, workers employed by the data loader
            - jacquard_output: boolean, generate jacquard output to send to server
            - run_on_server: boolean, silence tqdm output when on server to reduce output/file size
    """

    def __init__(self, dataset='jacquard', net_name='', epoch=0,
                 handle_as_ggcnn=False, handle_as_discrete_bins=False, im_size=320, jaw_size='half',
                 jaw_size_policy='min', num_of_bins=3, use_filter=False, filter=0.9,
                 use_binary_map=False, gaussian=2, compact=True, use_rgbd_img=False,
                 use_positional_loss=False, use_smooth_l1_loss=False, use_angle_loss=False, 
                 use_bin_loss=False, use_bin_attention_loss=False, use_graspness_loss=False,
                 batch_size=8, learning_rate=0.002, weight_decay=0,
                 use_dropout=False, dropout_prob=0.1,
                 use_early_stopping=True, restore_on_plateau=True, patience=1,
                 num_workers=2, jacquard_output=False, run_on_server=False):
        """Initialize configuration instance."""
        self.dataset = dataset
        self.net_name = net_name
        self.epoch = epoch
        self.handle_as_ggcnn = handle_as_ggcnn
        self.handle_as_discrete_bins = handle_as_discrete_bins
        self.im_size = im_size
        self.jaw_size = jaw_size
        self.jaw_size_policy = jaw_size_policy
        self.num_of_bins = num_of_bins if not handle_as_ggcnn else 1
        self.use_filter = use_filter
        self.filter = filter
        self.use_binary_map = use_binary_map
        self.gaussian = gaussian
        self.compact = compact
        self.use_rgbd_img = use_rgbd_img
        self.use_positional_loss = use_positional_loss
        self.use_smooth_l1_loss = use_smooth_l1_loss
        self.use_angle_loss = use_angle_loss
        self.use_bin_loss = use_bin_loss
        self.use_bin_attention_loss = use_bin_attention_loss
        self.use_graspness_loss = use_graspness_loss
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.use_early_stopping = use_early_stopping
        self.restore_on_plateau = restore_on_plateau
        self.patience = patience
        self.num_workers = num_workers
        self.jacquard_output = jacquard_output
        self.run_on_server = run_on_server
        self.writer = SummaryWriter(os.path.join(PATHS['results_path'], self.net_name))
        self._set_logger()

    def _set_logger(self):
        """Configure logger."""
        logging.getLogger("transformers").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        stream = logging.StreamHandler()
        stream.setFormatter(ColoredFormatter(
            '%(log_color)s%(asctime)s%(reset)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(stream)

    @property
    def use_cuda(self):
        """Return whether to use CUDA or not."""
        return USE_CUDA and torch.cuda.is_available()

    @property
    def paths(self):
        """Return a dict of paths useful to train/test/inference."""
        return PATHS

    @property
    def orig_data_path(self):
        """Return path of stored dataset files."""
        return os.path.join(DATA_FOLDER, self.dataset, '')
