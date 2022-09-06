How to operate the WidowX robot arm:

Ensure WidowX arm is turned on and camera is plugged in

1. first terminal, source ros environment

source ros_ws/devel/setup.zsh

2. launch WidowX arm rviz environment

roslaunch widowx_arm_bringup arm_moveit.launch sim:=false sr300:=false

3. Second terminal, cd into WidowX_OpenDay_Demo/robot folder

***
4. To send commands directly to arm in python, run robot.py. Can be run with other programs and used to control gripper and joints or positions.

python hardware/robot.py
Opens debugging environment to send commands e.g. robot.close_gripper()/robot.open_gripper()

Can also be controlled in GUI by RViZ

5. Calibrate arm before use

python run_calibration.py
Calibration matrix is saved to ros_ws/intrinsics

6. Calibrate by clicking on a 2D pixel coordinate in camera window and moving arm close to corresponding pixel (helps if gripper is closed)

Press "q" on camera window to close it and save calibration matrix

***

To run demo:

Launch pick 'n' place program to communicate with arm

python run_pnp.py

Fourth terminal, source venv to run with pytorch in python3 instead on python2

python run_grasp_demo.py --description demo --network-path /home/capture/WidowX_OpenDay_Demo/robot/trained_models/GRConv_RGBD_Pos_1bin/epoch_33_iou_0.91 --use-depth True --vis --output_size=320 --object_id examples --bins 1 --width_scaling 150 --use-width 

Ctrl+C on grasp demo terminal to execute a grasp at a given moment.
