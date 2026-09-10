# Android DVS 与 Edge AI 视觉系统

[English](README.md)

这是一个运行于 Android 智能手机端的 **DVS（Dynamic Vision Sensor，动态视觉传感器）仿真与 Edge AI 视觉系统**。项目使用普通智能手机摄像头近似模拟事件视觉，在手机端完成运动分析和人员检测，并支持将处理结果无线显示到 Windows 电脑。

> 本项目通过传统逐帧摄像头数据模拟 DVS 风格的事件输出，并不等同于真实硬件事件相机。

## 项目简介

系统使用 **Android CameraX** 实时获取相机图像并提取亮度信息，根据连续图像中的亮度变化生成类似 DVS 的 **ON/OFF 事件**。针对手机手持拍摄时容易出现的抖动和噪声，系统结合自适应阈值、基于 SAD 的轻量级平移补偿，以及基于 IMU 的相机运动检测。

项目同时在手机端部署了两个轻量级 AI 功能：

- **运动分类**：识别 `Object_Move`、`Camera_Move` 和 `Both_Move`
- **人员检测**：在事件图上检测人员，并显示目标框与置信度

主要图像处理与 AI 推理均在智能手机本地完成。

## 系统架构

![系统总体架构](docs/figures/system_architecture.png)

整体流程为：

`CameraX → Y-plane 亮度提取 → DVS 事件生成 → 运动分析 → 条件式人员检测 → 结果融合与显示`

同时使用陀螺仪和线性加速度数据辅助判断相机自身运动。

## 主要功能

| 模块 | 功能 |
|---|---|
| 相机输入 | 使用 CameraX 实时获取图像 |
| DVS 仿真 | 根据亮度变化生成 ON/OFF 事件图 |
| 自适应阈值 | 根据事件活动动态调整灵敏度 |
| 运动补偿 | 使用 SAD 估计小范围全局平移 |
| IMU 处理 | 使用陀螺仪与线性加速度判断手机运动 |
| 运动分类 | LiteRT/TFLite 识别 Object / Camera / Both movement |
| 人员检测 | 手机端人员检测、目标框与置信度 |
| 多模式显示 | Camera、Grayscale、Difference、DVS、Comparison 等 |
| 蓝牙输出 | 将处理后的图像发送至 Windows 外接显示端 |

## DVS 仿真

DVS 处理核心位于 `DvsProcessor.kt`。

对于每一帧待分析图像，系统将当前亮度图与参考亮度图进行比较：

- **ON event**：亮度增加超过阈值
- **OFF event**：亮度降低超过阈值
- **No event**：变化没有达到阈值

最终得到稀疏的 DVS 风格事件图。

### 主要处理阶段

下面的图片展示了 Android 应用得到的四种主要结果：普通相机画面、灰度图、Difference 图和最终 DVS Event Map。

![DVS主要处理阶段](docs/figures/dvs_processing_stages.png)

### 阈值控制

系统同时提供手动阈值与自适应阈值。较低的阈值会提高灵敏度，但也会保留更多背景变化；较高的阈值会生成更加稀疏的事件图。

![不同手动阈值效果](docs/figures/threshold_comparison.jpg)

论文实验中的事件数量为：

| Threshold | ON events | OFF events | Total events |
|---:|---:|---:|---:|
| 5 | 449 | 143 | 592 |
| 37 | 87 | 141 | 228 |
| 80 | 18 | 42 | 60 |

### 手持场景稳定性优化

**自适应阈值**  
自动模式动态调整事件阈值，在降低重复噪声的同时尽量保留有效变化。

**SAD 平移补偿**  
使用 Sum of Absolute Differences（SAD）对相邻图像的小范围全局 x/y 平移进行估计，降低轻微手持抖动产生的假事件。

**IMU 相机运动检测**  
结合手机陀螺仪与线性加速度信息判断相机自身运动，并在需要时提高事件生成的限制程度。

**空间采样**  
对亮度图进行空间降采样，以降低在单台手机上同时进行 DVS、运动分析、AI 推理和界面显示时的计算负担。

## Edge AI

### 运动分类

`MotionClassifier.kt` 加载 `saed_motion_classifier.tflite`，通过 LiteRT/TensorFlow Lite 在手机端完成推理。

模型输入：

```text
Shape: [1, 96, 128, 1]

类别:
- Object_Move
- Camera_Move
- Both_Move
```

该轻量级 CNN 共包含 **5,763 个参数**，训练完成后转换为 TFLite 模型部署到 Android。

#### 训练结果

![运动分类训练曲线](docs/figures/motion_training_curves.png)

训练和验证准确率最终收敛到约 0.90。在独立测试集上，最终分类器取得约 **87.6% 的总体准确率**。

#### 混淆矩阵

![运动分类混淆矩阵](docs/figures/motion_confusion_matrix.png)

| 类别 | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Object_Move | 1.000 | 1.000 | 1.000 |
| Camera_Move | 0.891 | 0.786 | 0.835 |
| Both_Move | 0.764 | 0.877 | 0.817 |

主要混淆发生在 `Camera_Move` 与 `Both_Move` 之间，因为二者都包含明显的全局相机运动特征。

### 人员检测

`PersonDetector.kt` 加载 `pedro_person_detector.tflite`，在累积后的事件图上执行手机端人员检测。

![人员检测示例](docs/figures/person_detection_examples.png)

实验表明，当事件图中能够形成较完整的人体轮廓时，模型可以定位人员。但当前实现也存在误检，主要原因之一是检测模型训练使用的真实事件相机数据与手机通过帧差生成的 DVS-like 数据之间存在 domain gap。

当前部署参数：

```text
输入尺寸: 320 × 320
置信度阈值: 0.35
NMS IoU 阈值: 0.40
最大检测数量: 3
```

## 显示模式

Android 应用提供多种显示模式：

- **Camera Only**：普通实时相机画面
- **Grayscale**：灰度/亮度图
- **Difference**：运动补偿后的帧差结果
- **DVS Only**：DVS 风格 ON/OFF 事件图
- **Comparison**：相机与处理结果对比
- **Parameters**：处理参数、事件数量和运动信息

应用同时支持手动阈值和自动阈值控制。

## 蓝牙外接显示

`BluetoothSenderManager.kt` 使用 Bluetooth Classic SPP 发送处理后的图像。为了避免传输速度低于处理速度时积压大量旧帧，发送端只保留最新的待发送图像。

当前传输配置：

```text
最大传输帧率: 15 FPS
图像最长边: 320 px
JPEG Quality: 35
数据协议: 4 字节大端序 JPEG 长度 + JPEG 图像数据
```

在手机到 Windows 的端到端实验中，稳定状态下测得约 **10.3–12.9 FPS**，平均约 **11.5 FPS**。

> 论文中的蓝牙截图暂时没有直接放入公开 README，因为原图包含已配对设备名称及 MAC 地址。后续可以先脱敏再加入。

## 技术栈

- Kotlin
- Android Studio
- Android CameraX
- Android Sensor API
- LiteRT / TensorFlow Lite
- Bluetooth Classic / SPP
- Gradle

## 运行环境

- Android Studio
- Android SDK 36
- 最低 Android API 24（Android 7.0）
- 带摄像头的 Android 手机
- 推荐设备具有陀螺仪和线性加速度传感器
- 蓝牙仅在使用外接显示功能时需要

## 如何运行

克隆仓库：

```bash
git clone https://github.com/JiaxuanJin/Android-DVS-EdgeAI-Vision-System.git
```

使用 Android Studio 打开项目，等待 Gradle Sync 完成，然后连接 Android 真机并运行 `app`。

应用需要 Camera 权限。在 Android 12 及以上系统中，如果使用蓝牙外接显示，还需要 Bluetooth Connect 权限。

AI 模型位于：

```text
app/src/main/assets/
├── motion_labels.txt
├── saed_motion_classifier.tflite
└── pedro_person_detector.tflite
```

## 项目结构

```text
app/src/main/
├── AndroidManifest.xml
├── assets/
│   ├── motion_labels.txt
│   ├── saed_motion_classifier.tflite
│   └── pedro_person_detector.tflite
├── java/com/example/framesenderapp/
│   ├── MainActivity.kt
│   ├── DvsProcessor.kt
│   ├── CameraMotionDetector.kt
│   ├── MotionClassifier.kt
│   ├── PersonDetector.kt
│   └── BluetoothSenderManager.kt
└── res/

docs/
└── figures/
    ├── system_architecture.png
    ├── dvs_processing_stages.png
    ├── threshold_comparison.jpg
    ├── motion_training_curves.png
    ├── motion_confusion_matrix.png
    └── person_detection_examples.png
```

## 项目局限

本系统使用普通智能手机摄像头的连续帧近似模拟事件相机，因此其时间分辨率仍受到摄像头帧率和曝光流程限制，无法像真实事件传感器一样在像素层面异步产生事件。

系统还会受到相机抖动、光照变化、真实事件数据与手机模拟事件数据之间的 domain gap、手机计算资源以及无线传输带宽的影响。

## 后续工作

后续可以继续扩展：

- 提升大幅光照变化下的事件生成稳定性
- 改进相机运动补偿
- 使用手机生成并人工标注的事件数据微调人员检测器
- 对 AI 模型进行量化或硬件加速
- 加入障碍物及距离估计
- 加入语音或触觉反馈，用于辅助导航
- 降低无线图像传输延迟

## 作者

**Jiaxuan Jin**

本项目为 University of Sheffield Electronic & Electrical Engineering MSc Final Project，研究方向为智能手机端事件视觉仿真与 Edge AI 实时视觉处理。
