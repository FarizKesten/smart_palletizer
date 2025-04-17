#!/bin/bash
# conda init
# conda activate rosenv

# Manually launch everything (instead of roslaunch)
rosparam load /workspace/src/box_detector/config/planar_patch.yaml
rosbag play /workspace/smart_palletizing_data.bag --clock &
rosrun box_detector box_detector_node.py &
rosrun box_detector planar_patch_node.py &

# Now launch GUI viewers
rqt_image_view /box_detector/image &
rqt_image_view /planar_patch/image &
rqt_image_view /planar_patch/overlay &
rqt_image_view /planar_patch/overlay_all &
