#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from cv_bridge import CvBridge
import os
import sys
import cv2 as cv
import numpy as np

# Add the lib/ directory to Python path
project_root = os.path.abspath("/workspace/")
if project_root not in sys.path:
    sys.path.append(project_root)

from lib.masked_yolo_detector import MaskedYOLODetector

class BoxDetectorNode:
    def __init__(self):
        rospy.init_node("box_detector_node", anonymous=True)

        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None

        # Load YOLO model path from parameter
        self.model_path = rospy.get_param("~model_path", "/workspace/data/model/best.pt")

        rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback, queue_size=1)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback, queue_size=1)

        self.box_pub = rospy.Publisher("/detected_boxes", Detection2DArray, queue_size=10)
        self.image_pub = rospy.Publisher("/box_detector/image", Image, queue_size=10)

        rospy.loginfo("BoxDetectorNode started.")
        rospy.spin()

    def color_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.try_detect()
        except Exception as e:
            rospy.logerr(f"Color image callback failed: {e}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.try_detect()
        except Exception as e:
            rospy.logerr(f"Depth image callback failed: {e}")

    def try_detect(self):
        if self.color_image is None or self.depth_image is None:
            return

        try:
            detector = MaskedYOLODetector(
                color_image=self.color_image,
                depth_image=self.depth_image,
                model_path=self.model_path,
                min_box_area=800
            )

            detector.create_mask()
            boxes = detector.run_detection(visualize=False)
            detector.print_box_details()

            # === Publish detections ===
            msg = Detection2DArray()
            msg.header = Header()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "camera_link"

            output_img = self.color_image.copy()

            for cls_name, conf, (x1, y1, x2, y2) in boxes:
                detection = Detection2D()
                detection.header = msg.header

                bbox = BoundingBox2D()
                bbox.center.x = (x1 + x2) / 2.0
                bbox.center.y = (y1 + y2) / 2.0
                bbox.size_x = x2 - x1
                bbox.size_y = y2 - y1
                detection.bbox = bbox

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.id = hash(cls_name) % 1000
                hypothesis.score = conf
                detection.results.append(hypothesis)

                msg.detections.append(detection)

                # Draw on output image
                label = f"{cls_name} {conf:.2f}"
                cv.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(output_img, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            self.box_pub.publish(msg)

            # === Publish image with boxes ===
            try:
                img_msg = self.bridge.cv2_to_imgmsg(output_img, encoding="bgr8")
                img_msg.header.stamp = rospy.Time.now()
                img_msg.header.frame_id = "camera_link"
                self.image_pub.publish(img_msg)
            except Exception as e:
                rospy.logwarn(f"Could not publish output image: {e}")

        except Exception as e:
            rospy.logerr(f"Detection failed: {e}")

if __name__ == "__main__":
    try:
        BoxDetectorNode()
    except rospy.ROSInterruptException:
        pass
