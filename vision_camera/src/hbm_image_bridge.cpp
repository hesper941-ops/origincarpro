#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "cv_bridge/cv_bridge.h"
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"

using hbm_img_msgs::msg::HbmMsg1080P;
using sensor_msgs::msg::CompressedImage;
using sensor_msgs::msg::Image;

class HbmImageBridge : public rclcpp::Node
{
public:
  HbmImageBridge() : Node("hbm_image_bridge")
  {
    image_sub_ = this->create_subscription<HbmMsg1080P>(
      "/hbmem_img",
      rclcpp::SensorDataQoS(),
      std::bind(&HbmImageBridge::image_callback, this, std::placeholders::_1));

    image_pub_ = this->create_publisher<Image>("/image_out", 10);
    compressed_pub_ = this->create_publisher<CompressedImage>("/image_out/compressed", 10);

    RCLCPP_INFO(
      this->get_logger(),
      "hbm_image_bridge started: /hbmem_img -> /image_out, /image_out/compressed");
  }

private:
  static std::string encoding_to_string(const std::array<unsigned char, 12> & encoding)
  {
    auto end_it = std::find(encoding.begin(), encoding.end(), static_cast<unsigned char>(0));
    return std::string(encoding.begin(), end_it);
  }

  void image_callback(const HbmMsg1080P::ConstSharedPtr msg)
  {
    if (!msg) {
      return;
    }

    const std::string encoding = encoding_to_string(msg->encoding);
    if (encoding != "nv12") {
      RCLCPP_ERROR(
        this->get_logger(),
        "Only nv12 input is supported, got: %s",
        encoding.c_str());
      return;
    }

    if (msg->height == 0 || msg->width == 0) {
      RCLCPP_WARN(this->get_logger(), "Invalid image size: %ux%u", msg->width, msg->height);
      return;
    }

    cv::Mat nv12_image(
      static_cast<int>(msg->height * 3 / 2),
      static_cast<int>(msg->width),
      CV_8UC1,
      const_cast<unsigned char *>(msg->data.data()));

    cv::Mat bgr_image;
    cv::cvtColor(nv12_image, bgr_image, cv::COLOR_YUV2BGR_NV12);

    std_msgs::msg::Header header;
    header.stamp = msg->time_stamp;
    header.frame_id = "camera";

    auto cv_image = cv_bridge::CvImage(header, "bgr8", bgr_image);

    auto image_msg = cv_image.toImageMsg();
    image_pub_->publish(*image_msg);

    std::vector<unsigned char> jpg_buffer;
    if (cv::imencode(".jpg", bgr_image, jpg_buffer)) {
      CompressedImage compressed_msg;
      compressed_msg.header = header;
      compressed_msg.format = "jpeg";
      compressed_msg.data = std::move(jpg_buffer);
      compressed_pub_->publish(compressed_msg);
    } else {
      RCLCPP_WARN(this->get_logger(), "Failed to encode compressed JPEG image");
    }
  }

  rclcpp::Subscription<HbmMsg1080P>::SharedPtr image_sub_;
  rclcpp::Publisher<Image>::SharedPtr image_pub_;
  rclcpp::Publisher<CompressedImage>::SharedPtr compressed_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HbmImageBridge>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
