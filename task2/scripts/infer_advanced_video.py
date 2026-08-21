"""Two-stage RM armor detection and digit recognition with unknown rejection."""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
from digit_pipeline import quality_gate, TemporalVote

ap = argparse.ArgumentParser()
ap.add_argument('--detector', required=True)
ap.add_argument('--classifier', required=True)
ap.add_argument('--source', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--det-conf', type=float, default=.32)
ap.add_argument('--digit-conf', type=float, default=.72)
a = ap.parse_args()
Path(a.output).parent.mkdir(parents=True, exist_ok=True)

detector, classifier = YOLO(a.detector), YOLO(a.classifier)
vote = TemporalVote(window=5, threshold=.55)
cap = cv2.VideoCapture(a.source)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(3)), int(cap.get(4))
writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
fallback_id = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = detector.track(frame, conf=a.det_conf, iou=.45, imgsz=960,
                            tracker='bytetrack.yaml', persist=True, verbose=False)[0]
    canvas = frame.copy()
    if result.boxes is not None:
        xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else None
        for i, ((x1,y1,x2,y2), armor_conf) in enumerate(zip(xyxy, confs)):
            x1,y1=max(0,x1),max(0,y1); x2,y2=min(w,x2),min(h,y2)
            track_id = int(ids[i]) if ids is not None else fallback_id+i
            roi = frame[y1:y2, x1:x2]
            reason = quality_gate(roi)
            raw_label, raw_conf = 'unknown', 0.0
            if reason is None:
                pred = classifier.predict(roi, imgsz=128, verbose=False)[0]
                idx = int(pred.probs.top1); raw_conf = float(pred.probs.top1conf)
                raw_label = pred.names[idx] if raw_conf >= a.digit_conf else 'unknown'
            label, digit_conf = vote.update(track_id, raw_label, raw_conf)
            color = (0,255,0) if label != 'unknown' else (0,165,255)
            cv2.rectangle(canvas, (x1,y1), (x2,y2), color, 3)
            text = f'id:{track_id} armor:{armor_conf:.2f} digit:{label} {digit_conf:.2f}'
            cv2.putText(canvas, text, (x1,max(25,y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                        .7, color, 2, cv2.LINE_AA)
    writer.write(canvas)

cap.release(); writer.release(); print(a.output)
