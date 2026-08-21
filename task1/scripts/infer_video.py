import argparse, cv2, numpy as np
from pathlib import Path
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(); ap.add_argument('--weights',required=True); ap.add_argument('--source',default=''); ap.add_argument('--output',default='outputs/task1_demo.mp4'); a=ap.parse_args(); Path(a.output).parent.mkdir(parents=True,exist_ok=True)
model=YOLO(a.weights) if YOLO else None; cap=cv2.VideoCapture(a.source) if a.source else None
if cap is None or not cap.isOpened():
    samples=sorted((ROOT/'data/images/val').glob('*.jpg'))[:24]
    frames=[]
    for p in samples:
        frame=cv2.imread(str(p))
        if frame is not None:
            frame=cv2.resize(frame,(640,640)); frames.extend([frame.copy() for _ in range(5)])
    if not frames: raise RuntimeError('No validation images; prepare the dataset first')
    h,w=frames[0].shape[:2]; writer=cv2.VideoWriter(a.output,cv2.VideoWriter_fourcc(*'mp4v'),24,(w,h))
    for f in frames: writer.write(model.predict(f,verbose=False,conf=.25)[0].plot() if model else f)
    writer.release()
else:
    fps=cap.get(cv2.CAP_PROP_FPS) or 24; w=int(cap.get(3)); h=int(cap.get(4)); writer=cv2.VideoWriter(a.output,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    while True:
        ok,f=cap.read()
        if not ok: break
        writer.write(model.predict(f,verbose=False,conf=.25)[0].plot() if model else f)
    cap.release(); writer.release()
print(a.output)
