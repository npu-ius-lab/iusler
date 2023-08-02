# NPU-IUSLer Robotics Team

## About

NPU-IUSLer is a team that represents the Intelligent Unmanned Systems Laboratory (IUSL) at the Unmanned System Research Institute (USRI), Northwestern Polytechnical University (NPU), China, founded in 2020, aiming to participate in robotics competitions.
Our team focuses on the research of perception, self-localization, and robot navigation in changing environments. Under our research objectives, we had participated in a number of robot competitions including drone obstacle avoidance and F1TENTH autonomous racing (competitions under ICAUS 2021, 2022) in China. We are currently extending our research into the area of multi-robot perception and mission planning.

## Members

Dr. Tao Yang, Assistant Professor, founder and team advisor.

BinHong Liu, undergraduate student, team leader. 

Bohui Fang, undergraduate student.

Zhen Liu, undergraduate student.

Dexin Yao, undergraduate student.

Guanyin Chen, undergraduate student. 

Yongzhou Pan, bachelor's student. 


## Team Description Paper

RoboCup Rescue Simulation Virtual Robot Competitions 2023:

https://github.com/cavayangtao/iusler/blob/main/iusler_robocup2023.pdf

## Open Source

We have an air-ground robotics [course (in Chinese)](https://github.com/cavayangtao/npurobocourse) at NPU. In the course, you can find some [ROS code](https://github.com/cavayangtao/rmtt_ros/tree/main/rmtt_tracker/scripts) of object tracker, gesture controller, path tracker, etc. We will continue to enrich the open-source content by participating in robotics competitions.

## Build
cd catkin_ws/src
```
git clone git@github.com:npu-ius-lab/iusler.git
colcon build
```
## Usage
cd catkin_ws
1. Start three robots in gazebo with:
```bash
source install/setup.bash
ros2 launch rvrl_gazebo house_map.launch.py
```
2. Start gmapping for each robot
```bash
source install/setup.bash
ros2 launch slam_gmapping slam_gmapping.launch.py namespace:=robot1
bash
source install/setup.bash
ros2 launch slam_gmapping slam_gmapping.launch.py namespace:=robot2
```
3. Start map merge
```bash
source install/setup.bash
ros2 launch multirobot_map_merge map_merge.launch.py
```
You can see two maps from robot1 and robot2 merge to a single one

4.Yolov5 in ros2
```bash
cd path to yolov5_ros2_node.py
source ~/ananconda/bin/activate
conda activate your env
python yolov5_ros2_node.py
```

## Related Publications

1. Pan Y, Wang J, Chen F, Lin Z, Zhang S, Yang T. How Does Monocular Depth Estimation Work for MAV Navigation in the Real World?. International Conference on Autonomous Unmanned Systems (ICAUS). 2022.

2. Yang T, Cappelle C, Ruichek Y, et al. Multi-object Tracking with Discriminant Correlation Filter Based Deep Learning Tracker. Integrated Computer-Aided Engineering (ICAE), 2019.

3. Yang T, Cappelle C, Ruichek Y, et al. Online Multi-object Tracking Combining Optical Flow and Compressive Tracking in Markov Decision Process. Journal of Visual Communication and Image Representation (JVCIR), 2019.
