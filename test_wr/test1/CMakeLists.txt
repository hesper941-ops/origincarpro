cmake_minimum_required(VERSION 3.8)
project(origincar_avoid)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# 安装 launch 文件
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# 安装 config 文件，如果你有 config 文件夹，建议加上
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

# 安装 teleop_smooth_ros2.py
install(PROGRAMS
  scripts/teleop_smooth_ros2.py
  DESTINATION lib/${PROJECT_NAME}
)

# 安装 cmd_vel_mux.py
install(PROGRAMS
  scripts/cmd_vel_mux.py
  DESTINATION lib/${PROJECT_NAME}
)


install(PROGRAMS
  scripts/odom_tf_broadcaster.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME odom_tf_broadcaster
)

# 把 yolo_box_viewer.py 安装成 yolo_box_viewer
# 这样 launch 里就可以写 executable='yolo_box_viewer'
install(PROGRAMS
  scripts/yolo_box_viewer.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME yolo_box_viewer
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()