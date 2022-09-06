import numpy as np
import cv2

import torch
import torch.utils.data

import random


class GraspDatasetBase(torch.utils.data.Dataset):
    """
    An abstract dataset for training GG-CNNs in a common format.
    """
    def __init__(self, ids=None, output_size=300, include_depth=True, include_rgb=False, random_rotate=False,
                 random_zoom=False, input_only=False, depth_background='inference/images/depth_background.tiff', rgb_background='inference/images/rgb_background.png'):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        if ids is not None:
            self.ids = np.array(ids)
        else:
            self.ids = None
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.depth_background = self.get_back_depth(depth_background)
        self.rgb_background = self.get_back_rgb(rgb_background)

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_mask(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_back_rgb(self, rgb_back, normalise=True):
        raise NotImplementedError()

    def get_back_depth(self, depth_back, normalise=True):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        img_mask = self.get_mask(idx, rot, zoom_factor)

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)
            depth_img = depth_img * img_mask
            depth_back = self.depth_background * ~img_mask
            depth_img = depth_img + depth_back
            depth_img = np.clip((depth_img - depth_img.mean()), -1, 1) # Normalise combined image

        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)
            rgb_img = rgb_img * img_mask
            # rgb_back = self.rgb_background * ~np.expand_dims(img_mask, axis=0)
            rgb_back = self.rgb_background * ~img_mask
            rgb_img = rgb_img + rgb_back
            rgb_img = rgb_img.astype(np.float32)/255.0 # Normalise the image
            rgb_img -= rgb_img.mean() # Then zero centre

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = np.clip(width_img, 0.0, 150.0)/150.0
        
        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2*ang_img))
        sin = self.numpy_to_torch(np.sin(2*ang_img))
        width = self.numpy_to_torch(width_img)

        return x, (pos, cos, sin, width), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)
