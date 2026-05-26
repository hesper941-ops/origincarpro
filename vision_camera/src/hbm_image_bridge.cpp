#include <functional>
#include <memory>
#include <string>

#include "cv_bridge/cv_bridge.h"
#include "hbm_img_msgs/msg/hbm_msg1080_p.hpp"
#include "opencv2/imgproc.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"

using hbm_img_msgs::msg::HbmMsg1080P;
using sensor_msgs::msg::CompressedImage;
using sensor_msgs::msg::Image;

class HbmImageBridge : public rclcpp::Node
{
public:
  HbmImageBridge()
  : Node("hbm_image_bridge")
  {
    image_sub_ = this->create_subscription<HbmMsg1080P>(
      "/hbmem_img", 10, std::bind(&HbmImageBridge::image_callback, this, std::placeholders::_1));
    image_pub_ = this->create_publisher<Image>("/image_out", 10);
    compressed_pub_ = this->create_publisher<CompressedImage>("/image_out/compressed", 10);
  }

private:
  void image_callback(const HbmMsg1080P::ConstSharedPtr msg)
  {
    if (!msg) {
      return;
    }

    const std::string encoding(reinterpret_cast<const char *>(msg->encoding.data()));
    if (encoding != "nv12") {
      RCLCPP_ERROR(this->get_logger(), "Only nv12 input is supported, got: %s", encoding.c_str());
      return;
    }

    cv::Mat nv12_image(
      static_cast<int>(msg->height * 3 / 2),
      static_cast<int>(msg->width),
      CV_8UC1,
      const_cast<uint8_t *>(msg->data.data()));
    cv::Mat bgr_image;
    cv::cvtColor(nv12_image, bgr_image, cv::COLOR_YUV2BGR_NV12);

    auto cv_image = cv_bridge::CvImage(msg->header, "bgr8", bgr_image);
    auto image_msg = cv_image.toImageMsg();
    auto compressed_msg = cv_image.toCompressedImageMsg(cv_bridge::JPEG);

    image_pub_->publish(*image_msg);
    compressed_pub_->publish(*compressed_msg);
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
