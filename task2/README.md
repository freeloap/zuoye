# 任务二：RoboMaster 装甲板及数字识别

## 功能

采用两阶段识别流程：

1. YOLO11s 检测完整装甲板；
2. 对装甲板 ROI 识别数字 `1`–`5`；
3. 模糊、过曝、遮挡或置信度不足时输出 `unknown`；
4. 使用 ByteTrack 和连续帧投票稳定结果。

## 环境

- Ubuntu 20.04+
- Python 3.8+
- PyTorch 2.0 + CUDA 11.8
- Ultralytics 8.4.123
- OpenCV、NumPy

```bash
pip install -r requirements.txt
```

## 数据集

### 装甲板检测

来源：GitHub `ansidd/RMPlateDetection` 公开 RoboMaster 比赛数据。保留原始 YOLO 图片和标注，训练/验证/测试划分为 580/196/197，类别统一为 `armor_plate`。

### 数字分类

来源：Kaggle `henrychur/rmdata2`。数字 `1`–`5` 为正类，`base/outpost/sentry/wrongpic` 归入 `unknown`；按原始 `update_train/update_test` 划分训练集和验证集。

数据集位于 `datasets/submission_datasets.tar.gz`。

## 训练

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

## 推理

```bash
python scripts/infer_advanced_video.py \
  --detector weights/armor_detector_yolo11s_960.pt \
  --classifier weights/digit_classifier_yolo11n_cls.pt \
  --source input.mp4 \
  --output output.mp4
```

## 输入、输出和后处理

- 视频输入：BGR 图像；装甲板检测尺寸为 960×960 letterbox。
- 数字输入：检测框 ROI，分类输入尺寸 128×128。
- 输出：`track_id, bbox, armor_conf, digit, digit_conf`。
- 装甲板置信度阈值：0.32。
- NMS IoU：0.45。
- 数字置信度阈值：0.72。
- 同一轨迹使用最近 5 帧投票。

数字分类器公开验证集 Top-1 准确率约 99.9%。实际开发板部署仍需测试帧率、端到端延迟和丢帧率。
