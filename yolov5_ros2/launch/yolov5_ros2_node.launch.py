import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    ld = LaunchDescription()

    node = Node(
        package="yolov5_ros2",
        name="yolov5_ros2",
        executable="yolov5_ros2_node.py",
        parameters=[
            {"confidence_threshold": "0.55"}, 
            {"iou_threshold": "0.45"},
            {"agnostic_nms": "True"},
            {"maximum_detections": "0.55"}, 
            {"line_thickness": "3"},
            {"view_image": False},
            {"weights": "/home/tianbot/yolov5_ws/src/yolov5_ros/src/yolov5/yolov5n.pt"}, 
            {"device": "0"},
            {"dnn": False},
            {"data": "/home/tianbot/yolov5_ws/src/yolov5_ros/src/yolov5/data/coco128.yaml"}, 
            {"inference_size_w": "640"},
            {"inference_size_h": "480"},
            {"half": "False"}, 
            {"input_image_topic": "/robot1/camera/image_raw"},
            {"output_topic": "/yolov5/detections"},
        ],
        output="screen"
    )

    ld.add_action(node)
    return ld































# <launch>
#     <!-- Detection configuration -->
#     <!-- <arg name="weights"               default="/home/tianbot/yolov5_ros_tensorrt/src/yolov5_ros/src/yolov5/yolov5s.engine"/> -->
#     <arg name="weights"               default="/home/tianbot/yolov5_ros_tensorrt/src/yolov5_ros/src/yolov5/yolov5n640 480.engine"/> 
#     <arg name="data"                  default="$(find yolov5_ros)/src/yolov5/data/coco128.yaml"/>
#     <arg name="confidence_threshold"  default="0.55"/>
#     <arg name="iou_threshold"         default="0.45"/>
#     <arg name="maximum_detections"    default="1000"/>
#     <arg name="device"                default="0"/>
#     <arg name="agnostic_nms"          default="true"/>
#     <arg name="line_thickness"        default="3"/>
#     <arg name="dnn"                   default="false"/>
#     <!-- tensorrt改成true，.pt false -->
#     <arg name="half"                  default="true"/>  
    
#     <!-- replace imgsz -->
#     <arg name="inference_size_h"      default="480"/>
#     <arg name="inference_size_w"      default="640"/>
    
#     <!-- Visualize using OpenCV window -->
#     <arg name="view_image"            default="false"/>

#     <!-- ROS topics -->
#     <arg name="input_image_topic"       default="/hikrobot_camera/rgb"/>
#     <arg name="output_topic"            default="/yolov5/detections"/>

#     <!-- Optional topic (publishing annotated image) -->
#     <arg name="publish_image"           default="true"/>
#     <arg name="output_image_topic"      default="/yolov5/image_out"/>


#     <node pkg="yolov5_ros" name="detect" type="detect_1.py" output="screen">
#         <param name="weights"               value="$(arg weights)"/>
#         <param name="data"                  value="$(arg data)"/>
#         <param name="confidence_threshold"  value="$(arg confidence_threshold)"/>
#         <param name="iou_threshold"         value="$(arg iou_threshold)" />
#         <param name="maximum_detections"    value="$(arg maximum_detections)"/>
#         <param name="device"                value="$(arg device)" />
#         <param name="agnostic_nms"          value="$(arg agnostic_nms)" />
#         <param name="line_thickness"        value="$(arg line_thickness)"/>
#         <param name="dnn"                   value="$(arg dnn)"/>
#         <param name="half"                  value="$(arg half)"/>

#         <param name="inference_size_h"      value="$(arg inference_size_h)"/>
#         <param name="inference_size_w"      value="$(arg inference_size_w)"/>

#         <param name="input_image_topic"     value="$(arg input_image_topic)"/>
#         <param name="output_topic"          value="$(arg output_topic)"/>

#         <param name="view_image"            value="$(arg view_image)"/>

#         <param name="publish_image"         value="$(arg publish_image)"/>
#         <param name="output_image_topic"    value="$(arg output_image_topic)"/>
#     </node>
#     <!-- <include file="$(find camera_launch)/launch/d435.launch"/> -->


# </launch>