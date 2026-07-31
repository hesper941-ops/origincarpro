#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


NEW_SENSOR_AND_CONTROL = r'''bool origincar_base::Get_Sensor_Data()
{
    short transition_16 = 0;
    uint8_t raw[RECEIVE_DATA_SIZE] = {0};

    const size_t bytes_read = Stm32_Serial.read(raw, sizeof(raw));
    if (bytes_read != RECEIVE_DATA_SIZE) {
      return false;
    }

    bool valid_frame = false;
    for (int start = 0; start < RECEIVE_DATA_SIZE; ++start) {
      if (raw[start] != FRAME_HEADER) {
        continue;
      }

      for (int i = 0; i < RECEIVE_DATA_SIZE; ++i) {
        Receive_Data.rx[i] = raw[(start + i) % RECEIVE_DATA_SIZE];
      }

      if (Receive_Data.rx[23] != FRAME_TAIL) {
        continue;
      }

      if (Receive_Data.rx[22] != Check_Sum(22, READ_DATA_CHECK)) {
        continue;
      }

      valid_frame = true;
      break;
    }

    if (!valid_frame) {
      return false;
    }

    Receive_Data.Frame_Header = Receive_Data.rx[0];
    Receive_Data.Frame_Tail = Receive_Data.rx[23];
    Receive_Data.Flag_Stop = Receive_Data.rx[1];

    Robot_Vel.X = Odom_Trans(Receive_Data.rx[2], Receive_Data.rx[3]);
    Robot_Vel.Y = Odom_Trans(Receive_Data.rx[4], Receive_Data.rx[5]);
    Robot_Vel.Z = Odom_Trans(Receive_Data.rx[6], Receive_Data.rx[7]);

    Mpu6050_Data.accele_x_data =
        IMU_Trans(Receive_Data.rx[8], Receive_Data.rx[9]);
    Mpu6050_Data.accele_y_data =
        IMU_Trans(Receive_Data.rx[10], Receive_Data.rx[11]);
    Mpu6050_Data.accele_z_data =
        IMU_Trans(Receive_Data.rx[12], Receive_Data.rx[13]);
    Mpu6050_Data.gyros_x_data =
        IMU_Trans(Receive_Data.rx[14], Receive_Data.rx[15]);
    Mpu6050_Data.gyros_y_data =
        IMU_Trans(Receive_Data.rx[16], Receive_Data.rx[17]);
    Mpu6050_Data.gyros_z_data =
        IMU_Trans(Receive_Data.rx[18], Receive_Data.rx[19]);

    Mpu6050.linear_acceleration.x =
        Mpu6050_Data.accele_x_data / ACCEl_RATIO;
    Mpu6050.linear_acceleration.y =
        Mpu6050_Data.accele_y_data / ACCEl_RATIO;
    Mpu6050.linear_acceleration.z =
        Mpu6050_Data.accele_z_data / ACCEl_RATIO;

    Mpu6050.angular_velocity.x =
        Mpu6050_Data.gyros_x_data * GYROSCOPE_RATIO;
    Mpu6050.angular_velocity.y =
        Mpu6050_Data.gyros_y_data * GYROSCOPE_RATIO;
    Mpu6050.angular_velocity.z =
        Mpu6050_Data.gyros_z_data * GYROSCOPE_RATIO;

    transition_16 = 0;
    transition_16 |= Receive_Data.rx[20] << 8;
    transition_16 |= Receive_Data.rx[21];
    Power_voltage =
        transition_16 / 1000 + (transition_16 % 1000) * 0.001;

    user_key = 0;
    return true;
}

void origincar_base::Control()
{
    rclcpp::Time current_time = rclcpp::Node::now();
    rclcpp::Time last_sensor_time = current_time;

    while (rclcpp::ok()) {
      rclcpp::spin_some(this->get_node_base_interface());

      if (Get_Sensor_Data()) {
        current_time = rclcpp::Node::now();
        Sampling_Time = (current_time - last_sensor_time).seconds();

        Robot_Pos.X +=
            1.03 *
            (Robot_Vel.X * cos(Robot_Pos.Z) -
             Robot_Vel.Y * sin(Robot_Pos.Z)) *
            Sampling_Time;
        Robot_Pos.Y +=
            1.125 *
            (Robot_Vel.X * sin(Robot_Pos.Z) +
             Robot_Vel.Y * cos(Robot_Pos.Z)) *
            Sampling_Time;
        Robot_Pos.Z += Robot_Vel.Z * Sampling_Time;

        Quaternion_Solution(
            Mpu6050.angular_velocity.x,
            Mpu6050.angular_velocity.y,
            Mpu6050.angular_velocity.z,
            Mpu6050.linear_acceleration.x,
            Mpu6050.linear_acceleration.y,
            Mpu6050.linear_acceleration.z);

        Publish_ImuSensor();
        Publish_Voltage();
        Publish_Odom();

        last_sensor_time = current_time;
      }
    }
}
origincar_base::origincar_base()'''


def fail(message: str) -> None:
    print(f"[错误] {message}", file=sys.stderr)
    raise SystemExit(1)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        fail(f"未找到待修改内容：{label}")
    if count > 1:
        fail(f"待修改内容出现 {count} 次，拒绝模糊替换：{label}")
    print(f"[修改] {label}")
    return text.replace(old, new, 1)


def replace_first(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0:
        fail(f"未找到待修改内容：{label}")
    print(f"[修改] {label}（匹配 {count} 处，仅修改第一处）")
    return text.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace-src",
        default="/root/intelligent_car_ws/src",
        help="ROS2 工作空间 src 目录",
    )
    args = parser.parse_args()

    src_root = Path(args.workspace_src).resolve()
    package_root = src_root / "origincar_base"

    cpp_path = package_root / "src/origincar_base.cpp"
    header_path = package_root / "include/origincar_base/origincar_base.h"
    launch_path = package_root / "launch/base_serial.launch.py"

    for path in (cpp_path, header_path, launch_path):
        if not path.is_file():
            fail(f"找不到文件：{path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = src_root / ".origincar_base_protocol_backup" / timestamp

    for path in (cpp_path, header_path, launch_path):
        relative = path.relative_to(src_root)
        target = backup_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    print(f"[备份] {backup_root}")

    header = header_path.read_text(encoding="utf-8")
    cpp = cpp_path.read_text(encoding="utf-8")
    launch = launch_path.read_text(encoding="utf-8")

    header = replace_once(
        header,
        "#define RECEIVE_DATA_SIZE 25",
        "#define RECEIVE_DATA_SIZE 24",
        "RECEIVE_DATA_SIZE：25 -> 24",
    )

    pattern = re.compile(
        r"bool\s+origincar_base::Get_Sensor_Data\(\)\s*"
        r"\{.*?\n\}\s*"
        r"void\s+origincar_base::Control\(\)\s*"
        r"\{.*?\n\}\s*"
        r"origincar_base::origincar_base\(\)",
        re.DOTALL,
    )
    cpp, count = pattern.subn(NEW_SENSOR_AND_CONTROL, cpp, count=1)
    if count != 1:
        fail("无法唯一定位 Get_Sensor_Data() 与 Control()")

    print("[修改] 替换 24 字节反馈解析与独立 ROS 回调循环")

    cpp = replace_once(
        cpp,
        "int serial_baud_rate = 921600;",
        "int serial_baud_rate = 115200;",
        "C++ 默认波特率：921600 -> 115200",
    )

    cpp = replace_once(
        cpp,
        'this->declare_parameter<std::string>("usart_port_name", "/dev/ttyCH343USB0");',
        'this->declare_parameter<int>("serial_baud_rate", 115200);\n'
        '  this->declare_parameter<std::string>("usart_port_name", "/dev/ttyACM0");',
        "显式声明 serial_baud_rate 参数",
    )

    cpp = replace_first(
        cpp,
        'Stm32_Serial.setPort("/dev/ttyACM0");',
        "Stm32_Serial.setPort(usart_port_name);",
        "构造函数使用 usart_port_name 参数",
    )

    cpp = replace_first(
        cpp,
        "serial::Timeout _time = serial::Timeout::simpleTimeout(2000);",
        "serial::Timeout _time = serial::Timeout::simpleTimeout(100);",
        "构造函数串口读取超时：2000 ms -> 100 ms",
    )

    cpp = replace_once(
        cpp,
        "Stm32_Serial.setBaudrate(921600);",
        "Stm32_Serial.setBaudrate(115200);",
        "SIGINT 停车波特率：921600 -> 115200",
    )

    cpp = cpp.replace(
        "STM32 firmware now uses 921600 for X5 communication; keep the ROS-side default in sync.",
        "The connected STM32 firmware uses 115200 baud and a 24-byte feedback frame.",
    )

    launch = replace_once(
        launch,
        "'serial_baud_rate': 921600",
        "'serial_baud_rate': 115200",
        "launch 波特率：921600 -> 115200",
    )
    launch = launch.replace(
        "STM32 firmware now uses 921600 for X5 communication; keep X5 side in sync.",
        "The connected STM32 firmware uses 115200 baud and a 24-byte feedback frame.",
    )

    required_cpp_tokens = [
        "Receive_Data.rx[23] != FRAME_TAIL",
        "Receive_Data.rx[22] != Check_Sum(22, READ_DATA_CHECK)",
        'declare_parameter<int>("serial_baud_rate", 115200)',
        "rclcpp::spin_some(this->get_node_base_interface());",
        "Stm32_Serial.setPort(usart_port_name);",
    ]
    for token in required_cpp_tokens:
        if token not in cpp:
            fail(f"生成结果缺少关键内容：{token}")

    header_path.write_text(header, encoding="utf-8")
    cpp_path.write_text(cpp, encoding="utf-8")
    launch_path.write_text(launch, encoding="utf-8")

    print()
    print("[完成] 源码修复已写入。")
    print(f"[备份目录] {backup_root}")
    print()
    print("下一步执行：")
    print("  source /opt/tros/humble/setup.bash")
    print("  cd /root/intelligent_car_ws")
    print("  colcon build --symlink-install --packages-select origincar_base")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
