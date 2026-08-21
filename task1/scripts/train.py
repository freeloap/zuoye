import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--device',default='0'); a=ap.parse_args()
runtime_yaml = ROOT/'data/data.runtime.yaml'
runtime_yaml.write_text(yaml.safe_dump({'path': str(ROOT/'data'), 'train': 'images/train', 'val': 'images/val', 'names': {0: 'miaocui_cat', 1: 'dao_dun'}}, allow_unicode=True), encoding='utf-8')
model=YOLO('yolo11n.pt'); model.train(data=str(runtime_yaml),epochs=a.epochs,imgsz=640,batch=16,device=a.device,project=str(ROOT/'runs'),name='task1',exist_ok=True,patience=10,workers=0)
