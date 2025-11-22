# ComfyUI-Easy-Sam3

[English](README.md) | [中文](README_CN.md)

一个用于 [SAM3 (Segment Anything Model 3)](https://github.com/facebookresearch/sam3) 的 ComfyUI 自定义节点包，提供强大的图像和视频分割功能，支持文本提示。

## 概述

本节点包将 Meta 的 SAM3 模型引入 ComfyUI，实现以下功能：
- **图像分割**：使用文本描述分割图像中的对象
- **视频跟踪**：跨视频帧追踪和分割对象
- **高级配置**：微调视频跟踪参数以获得最佳效果

### 图像分割
![图像分割示例](assets/image.png)
*图像语义分割示例*

### 视频分割
![视频分割示例](assets/video.png)
*视频帧点坐标分割示例*


## 功能特性

- 🖼️ **图像分割**：使用自然语言提示分割对象
- 🎬 **视频分割**：使用一致的 ID 跨视频帧跟踪对象
- 🎨 **背景选项**：为分割的图像添加自定义背景（黑色、白色、灰色）
- ⚙️ **灵活配置**：支持不同设备（CUDA、CPU、MPS）和精度（fp32、fp16、bf16）
- 🔧 **高级控制**：全面的视频跟踪参数可供微调

## 节点说明

### 1. 加载 SAM3 模型
加载用于图像或视频分割的 SAM3 模型。

**输入：**
- `model`：来自 models/sam3 文件夹的 SAM3 模型文件
- `segmentor`：选择 "image"（图像）或 "video"（视频）模式
- `device`：加载模型的设备（cuda、cpu、mps）
- `precision`：模型精度（fp32、fp16、bf16）

**输出：**
- `sam3_model`：已加载的 SAM3 模型，供下游节点使用

### 2. SAM3 图像分割
使用文本提示和可选的几何提示分割图像中的对象。

**输入：**
- `sam3_model`：来自"加载 SAM3 模型"节点的 SAM3 模型
- `images`：要分割的输入图像
- `prompt`：要分割的对象的文本描述（例如："一只猫"、"人"）
- `threshold`：检测的置信度阈值（0.0-1.0，默认：0.60）
- `keep_model_loaded`：推理后将模型保留在显存中
- `add_background`：添加背景颜色（无、黑色、白色、灰色）
- `coordinates_positive`（可选）：正向点坐标以细化分割
- `coordinates_negative`（可选）：负向点坐标以排除区域
- `bboxes`（可选）：边界框来引导分割
- `mask`（可选）：用于细化的输入遮罩

**输出：**
- `masks`：分割遮罩
- `images`：分割后的图像（可选背景）
- `boxes`：检测到的对象的边界框坐标
- `scores`：每个检测的置信度分数

### 3. SAM3 视频分割
跨视频帧跟踪和分割对象，支持高级提示选项。

**输入：**
- `sam3_model`：视频模式的 SAM3 模型
- `session_id`（可选）：会话 ID，用于从之前的会话恢复跟踪
- `video_frames`：作为图像序列的视频帧
- `prompt`：要跟踪的对象的文本描述（例如："人"、"汽车"）
- `frame_index`：应用初始提示的帧位置（0 到最大帧数）
- `object_id`：多对象跟踪的唯一 ID（1-1000，默认：1）
- `score_threshold_detection`：检测置信度阈值（0.0-1.0，默认：0.5）
- `new_det_thresh`：添加新对象的阈值（0.0-1.0，默认：0.7）
- `propagation_direction`：传播方向（双向、前向、后向）
- `start_frame_index`：开始传播的帧索引（默认：0）
- `max_frames_to_track`：要处理的最大帧数（-1 表示所有帧）
- `close_after_propagation`：完成后关闭会话（默认：True）
- `keep_model_loaded`：推理后将模型保留在显存中
- `extra_config`（可选）：来自额外配置节点的附加配置
- `positive_coords`（可选）：正向点击坐标，JSON 数组格式
- `negative_coords`（可选）：负向点击坐标，JSON 数组格式
- `bbox`（可选）：用于初始化跟踪的边界框

**输出：**
- `masks`：所有帧的跟踪分割遮罩
- `session_id`：用于恢复跟踪的会话 ID
- `objects`：对象跟踪信息和元数据

### 4. SAM3 视频模型额外配置
配置视频分割的高级参数，以微调跟踪行为。

**参数：**
- `assoc_iou_thresh`：检测到跟踪匹配的 IoU 阈值（0.0-1.0，默认：0.1）
- `det_nms_thresh`：检测 NMS 的 IoU 阈值（0.0-1.0，默认：0.1）
- `new_det_thresh`：添加新对象的阈值（0.0-1.0，默认：0.7）
- `hotstart_delay`：延迟 N 帧输出以移除不匹配/重复的轨迹（0-100，默认：15）
- `hotstart_unmatch_thresh`：在热启动期间移除未匹配此帧数的轨迹（0-100，默认：8）
- `hotstart_dup_thresh`：在热启动期间移除重叠的轨迹（0-100，默认：8）
- `suppress_unmatched_within_hotstart`：仅在热启动期间抑制未匹配的遮罩（默认：True）
- `min_trk_keep_alive`：最小保持活动值（-100-0，默认：-1，负值表示立即移除）
- `max_trk_keep_alive`：没有检测时保持跟踪活动的最大帧数（0-100，默认：30）
- `init_trk_keep_alive`：创建新跟踪时的初始保持活动值（-10-100，默认：30）
- `suppress_overlap_occlusion_thresh`：基于近期遮挡抑制重叠对象的阈值（0.0-1.0，默认：0.7，0.0 表示禁用）
- `suppress_det_at_boundary`：抑制接近图像边界的检测（默认：False）
- `fill_hole_area`：填充遮罩中小于此面积（像素）的孔洞（0-1000，默认：16）
- `recondition_every_nth_frame`：每 N 帧重新调整跟踪（-1-1000，默认：16，-1 表示禁用）
- `enable_masklet_confirmation`：启用掩码确认以抑制未确认的轨迹（默认：False）
- `decrease_alive_for_empty_masks`：减少空掩码的保持活动计数器（默认：False）
- `image_size`：模型的输入图像大小（256-2048，步长：8，默认：1008）

**输出：**
- `extra_config`：视频分割节点的配置字典

## 使用示例

### 基础图像分割
1. 加载 SAM3 模型（模式：image）
2. 连接到 SAM3 图像分割节点
3. 提供输入图像和文本提示
4. 获取分割遮罩和图像

### 视频对象跟踪
1. 加载 SAM3 模型（模式：video）
2. （可选）创建额外配置节点进行高级设置
3. 连接到 SAM3 视频分割节点
4. 提供视频帧和跟踪参数
5. 获取所有帧的跟踪遮罩

## 模型下载

从官方仓库下载 SAM3 模型权重：
- [SAM3 模型](https://huggingface.co/facebook/sam3)

将下载的模型放置在：`ComfyUI/models/sam3/`

## 系统要求

- Python 3.8+
- PyTorch 2.0+
- ComfyUI
- CUDA 兼容的 GPU（推荐）

## 本地化支持

本节点包支持多种语言：
- 英语（`locales/en/nodeDefs.json`）
- 中文（`locales/zh/nodeDefs.json`）

## 致谢

- **SAM3**：[Facebook Research](https://github.com/facebookresearch/sam3)
- **ComfyUI**：[comfyanonymous](https://github.com/comfyanonymous/ComfyUI)

## 许可证

本项目遵循原始 SAM3 仓库的许可证。

## 贡献

欢迎贡献！请随时提交问题或拉取请求。

## 更新日志

### v1.0.0
- 首次发布
- 支持文本提示的图像分割
- 视频跟踪和分割
- 图像分割包含背景颜色选项
- 高级视频跟踪配置
- 多语言支持（英文/中文）
