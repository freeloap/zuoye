import argparse
from pathlib import Path
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--device', default='0')
parser.add_argument('--name', default='rm_digit')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
model = YOLO('yolo11n-cls.pt')
model.train(data=args.data, epochs=args.epochs, imgsz=128, batch=128,
            device=args.device, project=str(root/'runs'), name=args.name,
            workers=2, patience=8)
