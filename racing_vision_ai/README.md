# Racing Vision AI Package

English | [简体中文](./README_cn.md)

This ROS2 package is designed to receive the `sign4return` signal. When the signal value reaches the configured target, it subscribes to an image topic, sends a single frame to the Volcengine AI model for image-to-text analysis, and outputs the result.

## Features

- Listens to the `sign4return` topic (`std_msgs/Int32`)
- Subscribes to an image topic and captures a single frame when the target sign value is received
- Sends the image to the Volcengine AI model for analysis
- Outputs AI analysis results to the ROS log
- Supports configuration files for managing API keys and parameters
- Supports configuration via ROS2 parameters and environment variables

## Dependencies

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Configure parameters:
Edit the `config/vision_ai_config.yaml` file and set your API key and model information:
```yaml
volcengine:
  api_key: "your_volcengine_api_key_here"
  model_id: "your_model_id_here"
detection:
  target_sign: 9  # The sign value that triggers image analysis
```

## Build and Run

1. Build the package:
```bash
cd /root/ros2_ws
colcon build --packages-select racing_vision_ai
source install/setup.bash
```

2. Run the node (with default configuration):
```bash
ros2 run racing_vision_ai vision_ai_node
```

3. Run with a specific config file:
```bash
ros2 run racing_vision_ai vision_ai_node --ros-args -p config_path:=/path/to/your/vision_ai_config.yaml
```

4. Set API key and model ID via environment variables:
```bash
export ARK_API_KEY="your_api_key_here" 
export ARK_MODEL_ID="your_model_id_here"
ros2 run racing_vision_ai vision_ai_node
```

## Configuration Priority

The system loads configuration in the following order:

1. Config file specified via parameter (`--ros-args -p config_path:=...`)
2. Config file in the current working directory
3. Development environment config file
4. Installed environment config file

For API key and model ID, the priority is:
1. Settings in the config file
2. Environment variables (`ARK_API_KEY` and `ARK_MODEL_ID`)
3. Default values

## Topic Interfaces

### Subscribed Topics

- `sign4return` (`std_msgs/Int32`): Control signal; triggers image analysis when the value matches `target_sign`
- `/image` (`sensor_msgs/CompressedImage`): Compressed image topic (subscribed only when triggered)

## Configuration File

The configuration file is located at `config/vision_ai_config.yaml` and includes:

### Volcengine Settings
- `volcengine.api_key`: API key
- `volcengine.model_id`: Model ID
- `volcengine.prompt`: AI analysis prompt

### Detection Settings
- `detection.target_sign`: The sign value that triggers image analysis (default: 9)
- `detection.image_topic`: Image topic name
- `detection.sign_topic`: Sign topic name

### Image Processing Settings
- `image.jpeg_quality`: JPEG image quality (1-100)
- `image.format`: Image format

## Notes

1. Ensure the API key and model ID are correctly set in the config file or via environment variables
2. Adjust topic names and target sign value in the config file as needed
3. The node processes only one image frame per trigger and then unsubscribes from the image topic
4. AI analysis results are output to the ROS log
5. You can modify the config file at any time; changes take effect on the next node startup

## Troubleshooting

If you see a "volcengine-python-sdk[ark] not installed" warning, install the dependency:
```bash
pip install volcengine-python-sdk[ark]
```

If the config file cannot be found, try the following:
1. Check if the config file path is correct
2. Use the `--ros-args -p config_path:=` parameter to specify the absolute path
3. Place the config file in the current working directory
4. Use environment variables to override key parameters
