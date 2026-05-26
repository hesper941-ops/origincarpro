# Racing Vision AI Package

[English](./README.md) | 简体中文

这个ROS2功能包用于接收sign4return信号，当信号值达到配置的目标值时，订阅一帧图像并使用火山引擎大模型进行图生文字分析。

## 功能特性

- 监听`sign4return`话题的std_msgs/Int32消息
- 当接收到配置的目标sign值时，订阅图像话题获取一帧图像
- 将图像发送给火山引擎AI大模型进行分析
- 输出AI分析结果到日志
- 支持配置文件管理API密钥和参数
- 支持通过ROS2参数和环境变量配置

## 安装依赖

1. 安装Python依赖：
```bash
pip install -r requirements.txt
```

2. 配置参数：
编辑 `config/vision_ai_config.yaml` 文件，设置你的API密钥和模型信息：
```yaml
volcengine:
  api_key: "your_volcengine_api_key_here"
  model_id: "your_model_id_here"
detection:
  target_sign: 9  # 触发图像分析的sign值
```

## 编译和运行

1. 编译功能包：
```bash
cd /root/ros2_ws
colcon build --packages-select racing_vision_ai
source install/setup.bash
```

2. 运行节点（使用默认配置）：
```bash
ros2 run racing_vision_ai vision_ai_node
```

3. 指定配置文件路径运行：
```bash
ros2 run racing_vision_ai vision_ai_node --ros-args -p config_path:=/path/to/your/vision_ai_config.yaml
```

4. 使用环境变量设置API密钥和模型ID：
```bash
export ARK_API_KEY="your_api_key_here" 
export ARK_MODEL_ID="your_model_id_here"
ros2 run racing_vision_ai vision_ai_node
```

## 配置优先级

系统按以下优先级加载配置：

1. 通过参数指定的配置文件 (`--ros-args -p config_path:=...`)
2. 当前工作目录下的配置文件
3. 开发环境的配置文件
4. 安装环境的配置文件

对于API密钥和模型ID，优先级为：
1. 配置文件中的设置
2. 环境变量 (`ARK_API_KEY` 和 `ARK_MODEL_ID`)
3. 默认值

## 话题接口

### 订阅的话题

- `sign4return` (std_msgs/Int32): 控制信号，当值为配置的target_sign时触发图像分析
- `/image` (sensor_msgs/CompressedImage): 压缩图像数据话题（仅在接收到触发信号时订阅）

## 配置文件

配置文件位于 `config/vision_ai_config.yaml`，包含以下配置项：

### 火山引擎配置
- `volcengine.api_key`: API密钥
- `volcengine.model_id`: 模型ID
- `volcengine.prompt`: AI分析提示词

### 检测配置
- `detection.target_sign`: 触发图像分析的sign值（默认为9）
- `detection.image_topic`: 图像话题名称
- `detection.sign_topic`: sign话题名称

### 图像处理配置
- `image.jpeg_quality`: JPEG图像质量（1-100）
- `image.format`: 图像格式

## 注意事项

1. 确保已正确配置配置文件中的API密钥和模型ID，或通过环境变量设置它们
2. 根据实际情况调整配置文件中的话题名称和目标sign值
3. 节点接收到触发信号后只处理一帧图像，然后取消图像话题订阅
4. AI分析结果会输出到ROS日志中
5. 可以随时修改配置文件，下次启动节点时新配置将生效

## 故障排除

如果出现"volcengine-python-sdk[ark] not installed"警告，请安装依赖：
```bash
pip install volcengine-python-sdk[ark]
```

如果无法找到配置文件，请尝试以下方法：
1. 检查配置文件路径是否正确
2. 使用`--ros-args -p config_path:=`参数指定配置文件的绝对路径
3. 将配置文件放在当前工作目录下
4. 设置环境变量替代配置文件中的关键参数
