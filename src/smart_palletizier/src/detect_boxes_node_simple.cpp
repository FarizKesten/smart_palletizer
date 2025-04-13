// detect_boxes_node.cpp - ROS 1 version
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <string>

// Intrinsic parameters (fallback if no camera_info is used)
float fx = 615.0f;
float fy = 615.0f;
float cx = 320.0f;
float cy = 240.0f;

// Adjustable parameters (set initially, but will be controlled by GUI)
int blur_kernel_size = 5;
int canny_thresh1 = 50;
int canny_thresh2 = 150;

image_transport::Publisher gray_pub, blurred_pub, edges_pub;

float pixelToMeters(int pixel_width, float depth, float focal_length) {
    return (depth * pixel_width) / focal_length;
}

void onTrackbar(int, void*) {
    // Trackbar callback — no action needed, values are read directly from globals
}

void callback(const sensor_msgs::ImageConstPtr& color_msg,
              const sensor_msgs::ImageConstPtr& depth_msg) {
    try {
        cv::Mat color = cv_bridge::toCvShare(color_msg, sensor_msgs::image_encodings::BGR8)->image;
        cv::Mat depth_raw = cv_bridge::toCvShare(depth_msg)->image;

        // Ensure odd kernel size
        if (blur_kernel_size % 2 == 0) blur_kernel_size++;

        // Preprocess
        cv::Mat gray, blurred, edges;
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(blur_kernel_size, blur_kernel_size), 0);
        cv::Canny(blurred, edges, canny_thresh1, canny_thresh2);

        // Publish intermediate images
        ros::Time stamp = color_msg->header.stamp;
        std_msgs::Header header = color_msg->header;
        gray_pub.publish(cv_bridge::CvImage(header, "mono8", gray).toImageMsg());
        blurred_pub.publish(cv_bridge::CvImage(header, "mono8", blurred).toImageMsg());
        edges_pub.publish(cv_bridge::CvImage(header, "mono8", edges).toImageMsg());

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            cv::Rect box = cv::boundingRect(contour);
            if (box.area() < 1000) continue;

            // Estimate average depth in region
            cv::Mat roi = depth_raw(box);
            cv::Scalar mean_depth = cv::mean(roi);
            float avg_depth_m = static_cast<float>(mean_depth[0]) / 1000.0f; // mm to meters

            float real_width = pixelToMeters(box.width, avg_depth_m, fx);
            float real_height = pixelToMeters(box.height, avg_depth_m, fy);

            std::string label = "Unknown";
            if (real_width > 0.23 && real_width < 0.27 && real_height > 0.13 && real_height < 0.17)
                label = "Medium Box";
            else if (real_width > 0.32 && real_width < 0.36 && real_height > 0.23 && real_height < 0.27)
                label = "Small Box";

            cv::rectangle(color, box, cv::Scalar(0, 255, 0), 2);
            cv::putText(color, label, box.tl() + cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255, 0, 0), 1);

            ROS_INFO("Detected box at (%d, %d) size %.3fm x %.3fm => %s", box.x, box.y, real_width, real_height, label.c_str());
        }

        cv::imshow("Detected Boxes", color);
        cv::waitKey(1);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "box_detector_node");
    ros::NodeHandle nh("~");
    image_transport::ImageTransport it(nh);

    // GUI trackbars for parameter tuning
    cv::namedWindow("Parameters", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Blur Kernel", "Parameters", &blur_kernel_size, 21, onTrackbar);
    cv::createTrackbar("Canny Thresh1", "Parameters", &canny_thresh1, 255, onTrackbar);
    cv::createTrackbar("Canny Thresh2", "Parameters", &canny_thresh2, 255, onTrackbar);

    gray_pub = it.advertise("/detected_boxes/gray", 1);
    blurred_pub = it.advertise("/detected_boxes/blurred", 1);
    edges_pub = it.advertise("/detected_boxes/edges", 1);

    message_filters::Subscriber<sensor_msgs::Image> color_sub(nh, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/aligned_depth_to_color/image_raw", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), color_sub, depth_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    cv::namedWindow("Detected Boxes", cv::WINDOW_AUTOSIZE);
    ros::spin();
    cv::destroyAllWindows();
    return 0;
}
