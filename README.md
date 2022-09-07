# Evaluating Gaussian Grasp Maps for Generative Grasping Models

[Link to ArXiV paper](https://arxiv.org/abs/2206.00432 "ArXiV paper link")

## Requirements:

The requirements for this repo are as follows:

* Ubuntu 16.04
* ROS Kinetic
* Python 2 (for ROS and WidowX Arm control)
* Python 3 (for PyTorch and model inference)

Specifically for the Arbotix WidowX robot arm and Intel Realsense D435
See github.com/Interbotix/widowx_arm and github.com/IntelRealSense/realsense-ros for further details

To use this repo clone it into your workspace:
```
git clone https://github.com/wtprew/grasping_robot.git
```

In order to make inferences using Pytorch and trained models, first create a virtual environment and install requirements.txt.
```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```
<details>
<summary>Setting up ROS environment</summary>

Create ROS workspace
```
mkdir -p ~/ros_ws/src
cd ~/ros_ws/src
```

Clone repositories for WidowX arm ROS control:
```
git clone https://github.com/Interbotix/arbotix_ros.git
git clone https://github.com/Interbotix/widowx_arm.git
```

Clone Realsense D435 packages:
```
sudo apt-get install ros-kinetic-realsense2-camera ros-kinetic-realsense2-description
git clone https://github.com/IntelRealSense/realsense-ros.git
```

Catkin_make and install packages, then source the environment
```
cd ..
catkin_make
source ~/ros_ws/devel/setup.sh
```

You can launch the ROS realsense with the command:
```
roslaunch realsense2_camera rs_camera.launch
```
see https://github.com/IntelRealSense/realsense-ros for details
</details>

You can launch an RVIZ control for the WidowX arm with the command:
```
roslaunch widowx_arm_bringup arm_moveit.launch sim:=false sr300:=false
```
see https://github.com/Interbotix/widowx_arm for details


***

## To run demo:

<details>
<summary>To set up control of WidowX arm</summary>

Ensure WidowX arm is turned on and camera is plugged in above the scene

1. first terminal, source ros environment (I use zsh so refer to relevant source script)

```
source ros_ws/devel/setup.zsh
```

2. launch ROS window to send commands to WidowX arm, opens RVIZ environment for GUI control
```
roslaunch widowx_arm_bringup arm_moveit.launch sim:=false sr300:=false
```

3. Second terminal, change into grasping_robot folder

```
cd grasping_robot
```
</details>

***

<details>
<summary>Control WidowX arm via Python</summary>

Utilises Python2 environment

You can send commands directly to arm via python using the robot.py script. Opens debugging environment to send commands and can be run with other programs and used to control gripper and joints or positions although this is not necessary for demo. Useful for calibration
```
python hardware/robot.py
```

Examples of often used commands include:
```
robot.open_gripper()
robot.close_gripper()
robot.move_to_neutral()
robot.move_to_pregrasp()
```
</details>

***

<details>
<summary>Arm Calibration</summary>

Utilises Python2 environment

To calibrate arm before use, you can generate a rotation matrix from the camera POV to the robot base using the script:
```
python run_calibration.py
```

Calibration matrix is automatically saved to ~/ros_ws/intrinsics as .npy file

Calibrate by clicking on a 2D pixel coordinate in matplotlib camera window and moving arm close to corresponding pixel.
Helps if gripper is closed, e.g. robot.close_gripper() (see above for sending commands to arm via Python).
Easiest way is to use RVIZ to visually move arm.
Ideally collect greater than six corresponding points for accurate matrices.

Press "q" on camera window to close it and save calibration matrix
</details>

***

## To generate example grasps using model:

This uses Pytorch and runs in Python3 so open new terminal the source venv to run with pytorch in python3 instead on python2

```
source venv/bin/activate
python run_grasp_demo.py --description demo --use-depth True --output_size=320 --bins 1 --network-path ~/grasping_robot/trained_models/GRConv_RGBD_Pos_1bin/epoch_33.pt --width_scaling 150 --use-width --vis
```

To allow for model inference to send commands to robot arm, launch pick 'n' place program in a separate terminal to communicate with arm. This uses the ROS environment.
```
source ros_ws/devel/setup.zsh
python run_pnp.py
```

Ctrl+C on grasp demo terminal to execute a grasp at a given moment.

If you use any of the work published in this paper please use the reference below:
```
@inproceedings{Prew2022,
  title={Evaluating Gaussian Grasp Maps for Generative Grasping Models},
  author={Prew, William and Breckon, Toby P and Bordewich, Magnus and Beierholm, Ulrik},
  booktitle = {Proc. IEEE International Joint Conference on Neural Networks},
  arxivId={2206.00432},
  year={2022}
}
```
