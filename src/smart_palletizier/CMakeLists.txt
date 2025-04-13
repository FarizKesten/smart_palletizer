cmake_minimum_required(VERSION 3.0.2)
project(smart_palletizer)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
  image_transport
  message_filters
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(detect_boxes_node src/detect_boxes_node_simple.cpp)
target_link_libraries(detect_boxes_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
