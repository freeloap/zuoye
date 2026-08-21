import argparse
from pathlib import Path
from ultralytics import YOLO
ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--epochs',type=int,default=80); ap.add_argument('--device',default='0'); ap.add_argument('--model',default='yolo11n.pt'); ap.add_argument('--imgsz',type=int,default=640); ap.add_argument('--name',default='rm_armor'); a=ap.parse_args()
root=Path(__file__).resolve().parents[1]; model=YOLO(a.model); model.train(data=a.data,epochs=a.epochs,imgsz=a.imgsz,batch=8 if a.imgsz>=960 else 16,device=a.device,project=str(root/'runs'),name=a.name,exist_ok=True,workers=2,patience=25,cos_lr=True,close_mosaic=10)
