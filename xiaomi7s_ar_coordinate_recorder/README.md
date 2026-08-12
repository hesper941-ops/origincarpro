# ARCoordinateRecorder

独立的 Android / ARCore 局部坐标记录测试应用。

## 功能
- 首次稳定 TRACKING 时建立局部原点 `(0,0,0)`。
- 实时显示 ARCore 相机相对位姿的 X/Y/Z（米）。
- 手动“记录当前位置”。
- 10 Hz 连续轨迹记录。
- 当前位置可重新设为原点。
- CSV 导出到 `Downloads/ARCoordinateRecorder/`。
- CSV 同时记录 Android `elapsedRealtimeNanos` 和 ARCore Frame timestamp，便于后续实验对时分析。

## 坐标定义
使用 `originPose.inverse().compose(cameraPose)`，因此坐标轴与建立原点时的相机姿态对齐：
- +X：初始相机右侧
- +Y：上方
- 相机朝向为 -Z，因此 +Z 是初始相机后方

这些是局部相对坐标，不是 GPS 经纬度；新的 ARCore Session 不保证与上一次 Session 使用相同世界原点。

## 构建
项目使用：
- compileSdk 35 / targetSdk 35 / minSdk 24
- Android Gradle Plugin 8.7.3
- Gradle 8.9
- Java 17
- ARCore `com.google.ar:core:1.33.0`

GitHub Actions 工作流会生成 debug APK artifact。
