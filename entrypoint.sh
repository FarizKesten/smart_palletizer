#!/bin/bash
source /opt/ros/noetic/setup.bash
export QT_X11_NO_MITSHM=1
exec "$@"
