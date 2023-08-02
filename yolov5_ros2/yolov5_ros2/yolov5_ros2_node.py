#!/usr/bin/env python3
import rclpy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
# from rostopic import get_topic_type
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox, BoundingBoxes,ObjectHypothesisWithPose
from sensor_msgs.msg import Image, CompressedImage
# from detection_msgs.msg import BoundingBox, BoundingBoxes
from rclpy.node import Node

# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATHimage_pub
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
class Yolov5Detector(Node):
    def __init__(self):
        super().__init__('YOLOV5_ROS2_NODE')
        # ###声明参数
        # self.declare_parameter('confidence_threshold')
        # self.declare_parameter('iou_threshold')
        # self.declare_parameter('agnostic_nms')
        # self.declare_parameter('maximum_detections')
        # # self.declare_parameter('classes')
        # self.declare_parameter('line_thickness')
        # self.declare_parameter('view_image')
        # self.declare_parameter('weights')
        # self.declare_parameter('device')
        # self.declare_parameter('dnn')
        # self.declare_parameter('data')
        # self.declare_parameter('inference_size_w')
        # self.declare_parameter('inference_size_h')
        # self.declare_parameter('half')
        # self.declare_parameter('input_image_topic')
        # self.declare_parameter('output_topic')
        # self.declare_parameter('publish_image')
        # self.declare_parameter('output_image_topic')
        ###参数获取
        self.conf_thres = 0.55
        self.iou_thres = 0.45
        self.agnostic_nms = True
        self.max_det = 1000
        # self.classes = self.get_parameter("~classes", None)
        self.classes = None
        self.line_thickness = 3
        self.view_image = True
        weights = "/home/tianbot/yolov5_ws/src/yolov5_ros/src/yolov5/yolov5n.pt"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dnn = False
        self.data = "/home/tianbot/yolov5_ws/src/yolov5_ros/src/yolov5/data/coco128.yaml"
        self.model = DetectMultiBackend(weights, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), dnn=False, data=self.data)
        self.inference_size_w = 640
        self.inference_size_h = 480
        self.half = False
        self.input_image_topic = "/robot3/camera/image_raw"

        self.output_topic = "/yolov5/detections"
        self.output_image_topic = "/yolov5/image_out"

        self.publish_image = False
        # Initialize weights
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )
        self.img_size = [self.inference_size_w, self.inference_size_h]
        
        self.img_size = check_img_size(self.img_size, s=self.stride)
        print(self.img_size,self.stride)
        # Half
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.img_size))  # warmup        
        
        # # Initialize create_subscription to Image/CompressedImage topic
        # input_image_type, input_image_topic, _ = get_topic_type(self.input_image_topic , blocking = True)
        # self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"
        self.compressed_input = False
        if self.compressed_input:
            self.image_sub = self.create_subscription(
                CompressedImage, input_image_topic, self.callback, 1
            )
        else:
            self.image_sub = self.create_subscription(
                Image,self.input_image_topic,  self.callback, 1
            )

        # Initialize prediction create_publisher


        self.pred_pub = self.create_publisher(
            Detection2DArray,self.output_topic,  10
        )



        # Initialize image create_publisher
        if self.publish_image:
            self.image_pub = self.create_publisher(
                Image,self.output_image_topic,  10
            )
        
        # Initialize CV_Bridge
        # self.bridge = CvBridge()

    def callback(self, data):
        """adapted from yolov5/detect.py"""


        im = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height,data.width, -1)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            
        im, im0 = self.preprocess(im)

        im = torch.from_numpy(im).to(self.device) 
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )
        ### To-do move pred to CPU and fill BoundingBox messages
        
        # Process predictions 
        det = pred[0].cpu().numpy()
        vision_msg = Detection2DArray()
        vision_msg.header = data.header

        t_1 = ObjectHypothesisWithPose()

        
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                bounding_box = Detection2D()
                t_1 = ObjectHypothesisWithPose()
                c = int(cls)
                # Fill in bounding box message
                if self.names[c] == "person":
                    t_1.id = 1
                    t_1.score =float(conf)
                elif self.names[c] == "cyclist":
                    t_1.id = 2
                    t_1.score = float(conf)
                elif self.names[c] == "car":
                    t_1.id = 0
                    t_1.score = float(conf)
                else:
                    t_1.id = 9
                    t_1.score = float(conf)                                                
                bounding_box.bbox.center.x = float((int(xyxy[0])+int(xyxy[2]))/2.0)
                bounding_box.bbox.center.y = float((int(xyxy[1])+int(xyxy[3]))/2.0)
                bounding_box.bbox.size_x = float(int(xyxy[2]) - int(xyxy[0]))
                bounding_box.bbox.size_y = float(int(xyxy[3]) - int(xyxy[1]))
                bounding_box.results.append(t_1)
                bounding_box.header = data.header
                vision_msg.detections.append(bounding_box)
        
                # Annotate the image
                if self.publish_image or self.view_image:  # Add bbox to image
                      # integer class
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))       
        
                
                ### POPULATE THE DETECTION MESSAGE HERE

            # Stream results
            im0 = annotator.result()
        else:
            bounding_box = Detection2D()
            bounding_box.header = data.header
            t_1 = ObjectHypothesisWithPose()
            t_1.id = -1
            bounding_box.results.append(t_1)
            vision_msg.detections.append(bounding_box)
        # Publish prediction
        self.pred_pub.publish(vision_msg)

        # Publish & visualize images
        if self.view_image:
            cv2.imshow(str(0), im0)
            cv2.waitKey(1)  # 1 millisecond
        if self.publish_image:
            image = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            img_msg = Image()
            img_msg.header = data.header           
            img_msg.encoding = 'bgr8'            
            img_msg.height = image.shape[0]
            img_msg.width = image.shape[1]                
            img_msg.step = image.shape[1] * image.shape[2]
            img_msg.data = np.array(image).tostring()
            self.image_pub.publish(img_msg)
        

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        return img, img0 


if __name__ == "__main__":

    # check_requirements(exclude=("tensorboard", "thop"))
    
    rclpy.init()

    detector = Yolov5Detector()
    
    rclpy.spin(detector)
