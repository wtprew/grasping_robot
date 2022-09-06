import logging

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id,
                 width=640,
                 height=480,
                 fps=6):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None

    def connect(self):
        # Start and configure        
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        depth_profile = cfg.get_stream(rs.stream.depth)
        self.depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        
    def get_image_bundle(self):
        frames = self.pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= self.scale
        color_image = np.asanyarray(color_frame.get_data())
        #update depth intrinsics which may change after alignment
        self.depth_intrinsics = rs.video_stream_profile(aligned_depth_frame.profile).get_intrinsics()

        depth_image = np.expand_dims(depth_image, axis=2)

        return {
            'rgb': color_image,
            'aligned_depth': depth_image,
        }

    def pixel_to_point(self, x, y, depth_point):
        depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depth_point)
        return depth_point

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        plt.show()

    def plot_cropped_image_bundle(self, output_size=300):
        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('depth')
        
        images = self.get_image_bundle()
        
        width = 640
        height = 480

        rgb = images['rgb']
        depth = images['aligned_depth']
        left = (width - output_size) // 2
        top = (height - output_size) // 2
        right = (width + output_size) // 2
        bottom = (height + output_size) // 2
        
        top_left = (top, left)
        bottom_right = (bottom, right)
        
        rgb = rgb[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        depth = depth[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        ax[0, 0].imshow(rgb)
        ax[0, 1].imshow(depth.squeeze(), cmap='binary')

        plt.show()

    def plot_cropped_image_refresh(self, output_size=300):
        fig, ax = plt.subplots(1, 1, squeeze=False)
        
        ax[0, 0].set_title('rgb')
        while True:
            images = self.get_image_bundle()
            
            width = 640
            height = 480

            rgb = images['rgb']
            left = (width - output_size) // 2
            top = (height - output_size) // 2
            right = (width + output_size) // 2
            bottom = (height + output_size) // 2
            
            top_left = (top, left)
            bottom_right = (bottom, right)
            
            rgb = rgb[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

            ax[0, 0].imshow(rgb)

            fig.canvas.draw()
            # fig.canvas.flush_events()
            plt.pause(0.2)

if __name__ == '__main__':
    cam = RealSenseCamera(device_id=831612070538)
    cam.connect()
    while True:
        cam.plot_cropped_image_refresh()
        plt.close()