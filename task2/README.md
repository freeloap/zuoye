# RoboMaster 机甲大师视觉作业：装甲板 + 数字识别

## 1. 作业内容

本目录对应任务二进阶版：检测 RoboMaster 完整装甲板，并识别装甲板数字 `1`–`5`。数字看不清、过曝、遮挡或 ROI 太小时输出 `unknown`，不强行猜测。

任务二提交内容完整包含：训练好的模型权重、两段推理演示视频、含标注的训练数据集及来源/划分说明、训练与推理脚本，以及输入尺寸、预处理、输出格式、后处理阈值、数字识别方式和运行环境说明。

## 2. 目录

```text
task2/
├─ weights/（两个训练好的 .pt 模型）
├─ videos/（两段推理演示视频）
├─ datasets/submission_datasets.tar.gz（含标注数据集）
├─ scripts/（训练、推理和数据处理脚本）
├─ requirements.txt
└─ README.md
```

## 3. 功能

采用两阶段识别流程：

1. YOLO11s 检测完整装甲板；
2. 对装甲板 ROI 识别数字 `1`–`5`；
3. 模糊、过曝、遮挡或置信度不足时输出 `unknown`；
4. 使用 ByteTrack 和连续帧投票稳定结果。

## 4. 环境

- Ubuntu 20.04+
- Python 3.8+
- PyTorch 2.0 + CUDA 11.8
- Ultralytics 8.4.123
- OpenCV、NumPy

```bash
pip install -r requirements.txt
```

## 5. 数据集、来源与划分

### 装甲板检测

来源：GitHub `ansidd/RMPlateDetection` 公开 RoboMaster 比赛数据。保留原始 YOLO 图片和标注，训练/验证/测试划分为 580/196/197，类别统一为 `armor_plate`。

### 数字分类

来源：Kaggle `henrychur/rmdata2`。数字 `1`–`5` 为正类，`base/outpost/sentry/wrongpic` 归入 `unknown`；按原始 `update_train/update_test` 划分训练集和验证集。

数据集压缩包位于 `datasets/submission_datasets.tar.gz`，其中包含装甲板检测图片/标注和数字分类原始数据及划分信息。

## 6. 训练

装甲板检测：

```bash
python scripts/train_armor.py --data data/rm_open/data.yaml \
  --epochs 40 --model yolo11s.pt --imgsz 960 --device 0 --name rm_armor
```

数字分类：

```bash
python scripts/prepare_digit_dataset.py --source data/rm_digit_raw --output data/rm_digit
python scripts/train_digit.py --data data/rm_digit --epochs 15 --device 0 --name rm_digit
```

## 7. 推理输入与预处理

- 输入：BGR 视频帧，支持竖屏 1080×2368 等尺寸。
- 装甲板检测：YOLO letterbox，推理尺寸 960×960。
- 数字识别：从检测框裁剪 ROI，缩放到 128×128 分类输入。
- ROI 质量门控：过小、极暗/过曝、边缘能量过低时直接输出 `unknown`。

```bash
python scripts/infer_advanced_video.py \
  --detector weights/armor_detector_yolo11s_960.pt \
  --classifier weights/digit_classifier_yolo11n_cls.pt \
  --source input.mp4 \
  --output output.mp4
```

## 8. 输出与后处理

- 视频输入：BGR 图像；装甲板检测尺寸为 960×960 letterbox。
- 数字输入：检测框 ROI，分类输入尺寸 128×128。
- 输出：`track_id, bbox, armor_conf, digit, digit_conf`。
- 装甲板置信度阈值：0.32。
- NMS IoU：0.45。
- 数字置信度阈值：0.72。
- 同一轨迹使用最近 5 帧投票。

数字分类置信度低于 `0.72` 或质量门控失败时输出 `unknown`。

## 9. 结果说明

数字分类器公开验证集 Top-1 约 99.9%。两段演示视频展示了清晰数字输出 1–5，以及模糊/遮挡情况下输出 `unknown` 的拒识策略。最终部署到开发板时还需实测 FPS、端到端延迟和丢帧率。

数字分类器公开验证集 Top-1 准确率约 99.9%。实际开发板部署仍需测试帧率、端到端延迟和丢帧率。
