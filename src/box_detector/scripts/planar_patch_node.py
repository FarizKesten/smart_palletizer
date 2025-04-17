#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import os
import sys
import cv2 as cv
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

project_root = os.path.abspath("/workspace")
if project_root not in sys.path:
    sys.path.append(project_root)

from lib.planar_patch_detector import PlanarPatchDetector
from lib.utils import get_camera_intrinsics

class PlanarPatchNode:
    def __init__(self):
        rospy.init_node("planar_patch_node", anonymous=True)

        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None
        self.detections = None

        self.intrinsics_path = rospy.get_param("~intrinsics_path")
        self.known_dims = rospy.get_param("~known_dims", {})

        self.detector_params = {
            'normal_variance_threshold_deg': rospy.get_param("~normal_variance_threshold_deg"),
            'coplanarity_deg': rospy.get_param("~coplanarity_deg"),
            'min_plane_edge_length': rospy.get_param("~min_plane_edge_length"),
            'min_num_points': rospy.get_param("~min_num_points"),
            'knn': rospy.get_param("~knn"),
            'voxel_size': rospy.get_param("~voxel_size"),
            'distance_threshold': rospy.get_param("~distance_threshold"),
            'nb_neighbors': rospy.get_param("~nb_neighbors"),
            'std_ratio': rospy.get_param("~std_ratio"),
            'min_plane_area': rospy.get_param("~min_plane_area")
        }

        self.camera_intrinsics = get_camera_intrinsics(self.intrinsics_path)

        self.image_pub = rospy.Publisher("/planar_patch/image", Image, queue_size=1)
        self.overlay_pub = rospy.Publisher("/planar_patch/overlay", Image, queue_size=1)
        self.overlay_all_pub = rospy.Publisher("/planar_patch/overlay_all", Image, queue_size=1)
        self.overlay_top_pub = rospy.Publisher("/planar_patch/overlay_top", Image, queue_size=1)
        self.cloud_pub = rospy.Publisher("/planar_patch/points", PointCloud2, queue_size=1)

        rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        rospy.Subscriber("/detected_boxes", Detection2DArray, self.detection_callback)

        rospy.loginfo("PlanarPatchNode started.")
        rospy.spin()

    def color_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.try_process()
        except Exception as e:
            rospy.logerr(f"Color image callback failed: {e}")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.try_process()
        except Exception as e:
            rospy.logerr(f"Depth image callback failed: {e}")

    def detection_callback(self, msg):
        self.detections = msg
        self.try_process()

    def convert_ros_detections(self, msg):
        boxes = []
        for detection in msg.detections:
            if not detection.results:
                continue
            hypothesis = detection.results[0]
            cls_name = "box"
            conf = hypothesis.score
            x = detection.bbox.center.x
            y = detection.bbox.center.y
            w = detection.bbox.size_x
            h = detection.bbox.size_y
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            boxes.append((cls_name, conf, (x1, y1, x2, y2)))
        return boxes

    def try_process(self):
        if self.color_image is None or self.depth_image is None or self.detections is None:
            return

        try:
            boxes = self.convert_ros_detections(self.detections)
            detector = PlanarPatchDetector(
                color_img=self.color_image,
                depth_img=self.depth_image,
                intrinsics=self.camera_intrinsics,
                known_dims=self.known_dims
            )
            detector.set_yolo_detections(boxes)
            detector.process_all(
                visualize=False,
                visualize_individual=False,
                mode='2d',
                **self.detector_params
            )

            overlay = detector.visualize_full_2d()
            if overlay is not None:
                overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
                overlay_msg.header = Header()
                overlay_msg.header.stamp = rospy.Time.now()
                overlay_msg.header.frame_id = "camera_link"
                self.overlay_pub.publish(overlay_msg)

            overlay_all = detector.visualize_overlay_all(return_image=True)
            if overlay_all is not None:
                msg = self.bridge.cv2_to_imgmsg(overlay_all, encoding="bgr8")
                msg.header = Header()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = "camera_link"
                self.overlay_all_pub.publish(msg)

            overlay_top = detector.visualize_overlay_top(return_image=True)
            if overlay_top is not None:
                msg = self.bridge.cv2_to_imgmsg(overlay_top, encoding="bgr8")
                msg.header = Header()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = "camera_link"
                self.overlay_top_pub.publish(msg)

            points = detector.get_top_patch_points()
            rospy.loginfo(f"Publishing {len(points)} points from top planar patches")

            vis_image = self.color_image.copy()
            for cls, conf, (x1, y1, x2, y2) in boxes:
                label = f"{cls} {conf:.2f}"
                cv.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(vis_image, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            img_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding="bgr8")
            img_msg.header = Header()
            img_msg.header.stamp = rospy.Time.now()
            img_msg.header.frame_id = "camera_link"
            self.image_pub.publish(img_msg)

            if points is not None and len(points) > 0:
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "camera_link"
                cloud_msg = pc2.create_cloud_xyz32(header, points)
                self.cloud_pub.publish(cloud_msg)

        except Exception as e:
            rospy.logerr(f"Planar patch detection failed: {e}")

if __name__ == "__main__":
    try:
        PlanarPatchNode()
    except rospy.ROSInterruptException:
        pass
