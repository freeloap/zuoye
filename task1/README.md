# 任务一：妙脆角猫与刀盾目标检测

## 目标类别

- `miaocui_cat`：妙脆角猫 meme
- `dao_dun`：刀盾 meme

## 环境

- Python 3.10+
- PyTorch
- Ultralytics YOLO
- OpenCV

安装依赖：

```bash
pip install -r requirements.txt
```

## 训练

```bash
python scripts/train.py --epochs 30
```

## 推理

```bash
python scripts/infer_video.py \
  --weights weights/best.pt \
  --source input.mp4 \
  --output output.mp4
```

模型输入尺寸为 640，默认置信度阈值为 0.25。推理视频绘制目标框、类别和置信度。

数据目录使用 YOLO 格式，包含训练集、验证集图片和对应标注。
