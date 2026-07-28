cmake_minimum_required(VERSION 3.5)
project(vision_camera)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(hbm_img_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(hbm_image_bridge
  src/hbm_image_bridge.cpp
)

ament_target_dependencies(
  hbm_image_bridge
  cv_bridge
  hbm_img_msgs
  rclcpp
  sensor_msgs
)

target_link_libraries(hbm_image_bridge
  ${OpenCV_LIBS}
)

install(
  TARGETS hbm_image_bridge
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)


install(PROGRAMS
  scripts/aurora_color_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
