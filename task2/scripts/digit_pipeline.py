"""ROI quality rejection and track-level digit voting."""
from collections import defaultdict, deque
import numpy as np


def quality_gate(roi: np.ndarray):
    if roi is None or min(roi.shape[:2]) < 16:
        return 'too_small'
    gray = roi.mean(axis=2) if roi.ndim == 3 else roi.astype(float)
    if gray.mean() > 245 or gray.mean() < 5:
        return 'exposure'
    edge_energy = np.var(np.diff(gray, axis=0)) + np.var(np.diff(gray, axis=1))
    if edge_energy < 20:
        return 'blurry'
    return None


class TemporalVote:
    def __init__(self, window=5, threshold=.55):
        self.window = window
        self.threshold = threshold
        self.history = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id, label, confidence):
        self.history[int(track_id)].append((label, float(confidence)))
        scores = defaultdict(float)
        counts = defaultdict(int)
        for item, score in self.history[int(track_id)]:
            if item != 'unknown':
                scores[item] += score
                counts[item] += 1
        total = sum(scores.values())
        if not scores or total <= 0:
            return 'unknown', 0.0
        label, score = max(scores.items(), key=lambda x: x[1])
        dominance = score / total
        probability = dominance * (score / counts[label])
        return (label, probability) if probability >= self.threshold else ('unknown', probability)
