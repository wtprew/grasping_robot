#!/bin/bash
#Alternative script to initialise joints after powering up

echo "Going to default pose: Moving joint 1"
rostopic pub -1 /joint_1/command std_msgs/Float64 "0.0"
echo "Moving joint 2"
rostopic pub -1 /joint_2/command std_msgs/Float64 "0.0"
echo "Moving joint 3"
rostopic pub -1 /joint_3/command std_msgs/Float64 "0.0"
echo "Moving joint 4"
rostopic pub -1 /joint_4/command std_msgs/Float64 "0.0"
echo "Moving joint 5"
rostopic pub -1 /joint_5/command std_msgs/Float64 "0.0"
echo "Opening gripper"
rostopic pub -1 /gripper_joint/command std_msgs/Float64 "0.0"
