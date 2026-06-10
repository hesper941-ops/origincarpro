cmake_minimum_required(VERSION 3.8)
project(origincar_avoid)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

install(PROGRAMS
  scripts/teleop_smooth_ros2.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME teleop_smooth_ros2
)

install(PROGRAMS
  scripts/cmd_vel_mux.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cmd_vel_mux
)

install(PROGRAMS
  scripts/yolo_box_viewer.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME yolo_box_viewer
)

execute_process(COMMAND chmod -R +x ${CMAKE_CURRENT_SOURCE_DIR}/scripts/teleop_smooth_ros2.py)
execute_process(COMMAND chmod -R +x ${CMAKE_CURRENT_SOURCE_DIR}/scripts/cmd_vel_mux.py)
execute_process(COMMAND chmod -R +x ${CMAKE_CURRENT_SOURCE_DIR}/scripts/yolo_box_viewer.py)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()




endif()

ament_package()
