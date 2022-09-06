#!/usr/bin/env python

import os
import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs

import argparse
import traceback

from camera import RealSenseCamera

from robot import WidowX
from config import *
from utils import *


class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

        self.x = 0
        self.y = 0

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

class ClickControl:
    def __init__(self, cam_id=831612070538, execute=False, calibrate=False, width=640, height=480):
        self.x = 0
        self.y = 0

        self.ch_x = 0
        self.ch_y = 0

        # WidowX controller interface
        self.robot = WidowX()

        self.camera = RealSenseCamera(cam_id)

        self.measured_pts = []
        self.observed_pts = []
        self.observed_pix = []

        self.execute = execute
        self.calibrate = calibrate

        homedir = "/home/capture/ros_ws/intrinsics"
        self.move_completed = os.path.join(homedir, "move_completed.npy")
        self.tool_position = os.path.join(homedir, "tool_position.npy")
        #Rotation Matrix
        cm = os.path.join(homedir, "camera_pose.npy")
        self.cm = np.load(cm)
    
        self.depth_intrinsics = None    
        self.w = width
        self.h = height
    
        self.state = AppState()
        self.out = np.empty((self.w, self.h, 3))
        self.depth = np.empty((self.w, self.h), dtype=np.float32)

    def setCrosshairs(self, x, y):
        self.ch_x = x
        self.ch_y = y

    def mouse_cb(self, event, x, y, flags, param):
        self.setCrosshairs(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.state.mouse_btns[0] = True
            depth_point = self.depth[y, x]
            self.take_action(x, y, depth_point)

        if event == cv2.EVENT_LBUTTONUP:
            self.state.mouse_btns[0] = False

        if event == cv2.EVENT_RBUTTONDOWN:
            self.state.mouse_btns[1] = True

        if event == cv2.EVENT_RBUTTONUP:
            self.state.mouse_btns[1] = False

        if event == cv2.EVENT_MBUTTONDOWN:
            self.state.mouse_btns[2] = True

        if event == cv2.EVENT_MBUTTONUP:
            self.state.mouse_btns[2] = False

        if event == cv2.EVENT_MOUSEMOVE:

            h, w = self.out.shape[:2]
            dx, dy = x - self.state.prev_mouse[0], y - self.state.prev_mouse[1]

            # if self.state.mouse_btns[0]:
            #     self.state.yaw += float(dx) / w * 2
            #     self.state.pitch -= float(dy) / h * 2

            # elif self.state.mouse_btns[1]:
            #     dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            #     self.state.translation -= np.dot(self.state.rotation, dp)

            if self.state.mouse_btns[2]:
                dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
                self.state.translation[2] += dz
                self.state.distance -= dz

        if event == cv2.EVENT_MOUSEWHEEL:
            dz = math.copysign(0.1, flags)
            self.state.translation[2] += dz
            self.state.distance -= dz

        self.state.prev_mouse = (x, y)

    def camera_connect(self):
        self.camera.connect()
        self.camera.get_pc()

        self.depth_intrinsics = self.camera.depth_intrinsics

    def take_action(self, x, y, depth):

        depth_point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depth)

        print('Pixel: (%s, %s)' % (x, y))
        print('XYZ point: ' + str(depth_point))

        if depth_point[2] == 0:
            print('No depth reading')
            print(depth_point)
            return

        if self.execute:
            depth_point = np.array([depth_point[0], depth_point[1], depth_point[2], 1.])
            transformed = np.dot(self.cm, depth_point)[:3]
            print('transformed', transformed)

            theta = 0.0
            grasp = np.concatenate([transformed, [theta]], axis=0)
            # grasp[2] -= Z_OFFSET

            print('Grasp: ' + str(grasp))
            # np.save(self.tool_position, grasp)
            # np.save(self.move_completed, 0)
            # while not np.load(self.move_completed):
            #     time.sleep(0.1)
            self.execute_grasp(grasp)
            time.sleep(2)       

        elif self.calibrate:
            user_in = raw_input('Keep recorded point? [y/n] ')
            
            if user_in == 'y':
                pose = self.robot.get_current_pose().pose.position
                print(pose.x, pose.y, pose.z)

                self.measured_pts.append([pose.x, pose.y, pose.z])
                self.observed_pts.append(depth_point[:3])
                self.observed_pix.append([x, y])
                print('Saved')

            elif user_in == 'n':
                print('Not saved')

    def get_pose(self):
        pose = self.robot.get_current_pose().pose
        pose_list = [pose.position.x,
                     pose.position.y,
                     pose.position.z,
                     pose.orientation.w,
                     pose.orientation.x,
                     pose.orientation.y,
                     pose.orientation.z]
        return pose_list

    def execute_grasp(self, grasp):
        try:
            x, y, z, theta = grasp

            if z < 0:
                z = Z_MIN

            print('Attempting grasp: (%.4f, %.4f, %.4f, %.4f)'
                  % (x, y, z, theta))

            assert inside_polygon(
                (x, y, z), END_EFFECTOR_BOUNDS), 'Grasp not in bounds'

            assert self.robot.orient_to_pregrasp(
                x, y), 'Failed to orient to target'

            assert self.robot.move_to_grasp(x, y, PRELIFT_HEIGHT, theta), \
                'Failed to reach pre-lift pose'

            assert self.robot.move_to_grasp(
                x, y, z, theta), 'Failed to execute grasp'

            self.robot.close_gripper()

            reached = self.robot.move_to_vertical(PRELIFT_HEIGHT)

            assert self.robot.move_to_drop(), 'Failed to move to drop'

            self.robot.open_gripper()

            self.robot.move_to_neutral()

        except Exception as e:
            print('Error executing grasp -- returning...')
            traceback.print_exc(e)

    def run(self):

        self.robot.move_to_neutral()
        self.camera.connect()

        # self.pc = camera.get_pc()

        self.depth_intrinsics = self.camera.depth_intrinsics 
        self.w, self.h = self.depth_intrinsics.width, self.depth_intrinsics.height
        cv2.namedWindow(self.state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.state.WIN_NAME, self.w, self.h)
        cv2.setMouseCallback(self.state.WIN_NAME, self.mouse_cb)

        while True:
            if not self.state.paused:
                frames = self.camera.pipeline.wait_for_frames()

                align = rs.align(rs.stream.color)
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.first(rs.stream.color)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                
                #Grab new intrinsics after aligning (may be changed)
                self.depth_intrinsics = rs.video_stream_profile(aligned_depth_frame.profile).get_intrinsics()
                self.w, self.h = self.depth_intrinsics.width, self.depth_intrinsics.height

                depth_image = np.asanyarray(aligned_depth_frame.get_data(), dtype=np.float32)
                depth_image *= self.camera.scale
                color_image = np.asanyarray(color_frame.get_data())
    
                self.out = color_image
                cv2.line(self.out, (self.ch_x-10, self.ch_y), (self.ch_x+10, self.ch_y), (255,255,0))
                cv2.line(self.out, (self.ch_x, self.ch_y-10), (self.ch_x, self.ch_y+10), (255,255,0))
                self.depth = depth_image
    
                # Render
                # now = time.time()
    
                cv2.setWindowTitle(
                    self.state.WIN_NAME, "RealSense (%dx%d) %s" % (self.w, self.h, "PAUSED" if self.state.paused else ""))
    
                cv2.imshow(self.state.WIN_NAME, cv2.cvtColor(self.out, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1)

                if key == ord("r"):
                    self.state.reset()
    
                if key == ord("p"):
                    self.state.paused ^= True
    
                # if key == ord("d"):
                #     state.decimate = (state.decimate + 1) % 3
                #     decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    
                if key == ord("z"):
                    self.state.scale ^= True
    
                if key == ord("c"):
                    self.state.color ^= True
    
                if key == ord("s"):
                    cv2.imwrite('./out.png', self.out)
    
                # if key == ord("e"):
                #     points.export_to_ply('./out.ply', mapped_frame)
    
                if key in (27, ord("q")) or cv2.getWindowProperty(self.state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
                    break
        
        self.camera.pipeline.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Executes user-specified grasps from a GUI window")
    parser.add_argument('--debug', action="store_true", default=False,
                        help="Prevents grasp from being executed (for debugging purposes)")
    args = parser.parse_args()

    executor = ClickControl(execute=(not args.debug), calibrate=False)

    executor.run()

if __name__ == '__main__':
    main()
