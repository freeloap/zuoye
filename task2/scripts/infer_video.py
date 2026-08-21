import argparse, cv2
from pathlib import Path
from ultralytics import YOLO
ap=argparse.ArgumentParser(); ap.add_argument('--weights',required=True); ap.add_argument('--source',required=True); ap.add_argument('--output',required=True); ap.add_argument('--conf',type=float,default=.32); ap.add_argument('--imgsz',type=int,default=960); ap.add_argument('--track',action='store_true'); a=ap.parse_args(); Path(a.output).parent.mkdir(parents=True,exist_ok=True)
model=YOLO(a.weights); cap=cv2.VideoCapture(a.source); fps=cap.get(cv2.CAP_PROP_FPS) or 30; w=int(cap.get(3)); h=int(cap.get(4)); out=cv2.VideoWriter(a.output,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
while True:
 ok,frame=cap.read()
 if not ok: break
 if a.track:
  r=model.track(frame,conf=a.conf,iou=.45,imgsz=a.imgsz,tracker='bytetrack.yaml',persist=True,verbose=False)[0]
 else:
  r=model.predict(frame,conf=a.conf,iou=.45,imgsz=a.imgsz,verbose=False)[0]
 out.write(r.plot())
cap.release(); out.release(); print(a.output)
