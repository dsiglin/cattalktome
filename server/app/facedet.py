"""
Learned cat-face detector: YOLOX-Nano fine-tuned on cat faces.

Trained by training/catface_detector_colab.ipynb (YOLOX, Apache-2.0;
data: Roboflow Universe "cat-face-data", CC BY 4.0). The model file is
optional: when server/models/catface_yolox_nano.onnx is absent this module
reports `available() == False` and detect.py falls back to the Haar
stages. When present it becomes the first and preferred stage.

Input contract (from the notebook's model card): 416x416 letterbox padded
with 114, BGR, 0-255, no normalisation. Output (1, N, 6): cx, cy, w, h,
objectness, class score - decoded to input-pixel coordinates.

Box framing: the dlib landmark predictor was trained on Haar-cascade-style
face boxes. A learned detector draws tighter or looser boxes, and a shape
predictor is sensitive to that framing. `FRAMING` maps a detector box to
the framing the predictor expects; tools/calibrate_face_box.py measures it
on real photos where both detectors fire. Identity until calibrated.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

_MODEL = Path(__file__).resolve().parent.parent / "models" / "catface_yolox_nano.onnx"
_INPUT = 416
_PAD = 114
CONF = 0.40
NMS = 0.45

CONF_RELIABLE = 0.80
"""Score floor for treating a detection as a trustworthy, high-confidence
face - not just "found something above the NMS floor".

Calibrated against the training run's held-out test split (real photos,
never trained on): among 1,290 detections that were genuinely correct
(IoU >= 0.3 against ground truth), only the bottom 1% scored below 0.80,
and the single lowest was 0.476. A real cat photographed in profile -
one eye and one ear occluded - scored 0.72: the model still draws a
correctly-placed box (verified visually), but dlib's shape predictor is
then forced to invent a plausible position for the eye it cannot see,
and that fabrication can accidentally look symmetric enough to pass the
nose-symmetry reliability check. The confidence score is the one signal
that caught it when geometry did not. Below this floor, detect.py treats
the box as not found and falls through to the Haar stages (and from
there to catdet's "a cat is here but its face is not readable" message)
rather than trusting a plausible-looking but partly invented face."""

FRAMING = {"scale": 1.0, "dx": 0.0, "dy": 0.0}
"""Detector box -> predictor box: w,h *= scale; centre += (dx*w, dy*h).

Measured against the training pipeline's held-out test split (1,290/1,295
real photos matched at IoU >= 0.3 against ground truth): scale 0.998,
dx 0.000, dy 0.004 - identity within noise, so left at the default. This
model was trained directly on boxes framed to match the extended Haar
cascade (training/prepare_catface_dataset.py box_from_landmarks), so no
separate correction was expected. Re-run tools/calibrate_face_box.py if
the model is ever retrained on a different box convention."""

_net = None


@dataclass(frozen=True)
class FaceBox:
    box: tuple  # (x, y, w, h) in the original image
    score: float


def available() -> bool:
    return _MODEL.exists()


def _load():
    global _net
    if _net is None:
        _net = cv2.dnn.readNetFromONNX(str(_MODEL))
    return _net


def _letterbox(bgr):
    h, w = bgr.shape[:2]
    scale = min(_INPUT / h, _INPUT / w)
    resized = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((_INPUT, _INPUT, 3), _PAD, dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas, scale


def find_faces(bgr, conf: float = CONF) -> List[FaceBox]:
    """All cat faces above `conf`, highest score first. Empty if the model is missing."""
    if not available():
        return []
    net = _load()
    img, scale = _letterbox(bgr)
    blob = img.transpose(2, 0, 1)[None].astype(np.float32)
    net.setInput(blob)
    out = net.forward()
    out = out[0] if out.ndim == 3 else out
    scores = out[:, 4] * out[:, 5]
    keep = scores > conf
    if not keep.any():
        return []
    o, s = out[keep], scores[keep]
    boxes = np.stack([o[:, 0] - o[:, 2] / 2, o[:, 1] - o[:, 3] / 2, o[:, 2], o[:, 3]], axis=1) / scale
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), s.astype(float).tolist(), conf, NMS)
    if len(idx) == 0:
        return []
    faces = [FaceBox(box=tuple(int(v) for v in boxes[i]), score=float(s[i])) for i in np.array(idx).flatten()]
    faces.sort(key=lambda f: -f.score)
    return faces


def to_predictor_framing(box) -> tuple:
    x, y, w, h = box
    cx, cy = x + w / 2 + FRAMING["dx"] * w, y + h / 2 + FRAMING["dy"] * h
    nw, nh = w * FRAMING["scale"], h * FRAMING["scale"]
    return (int(round(cx - nw / 2)), int(round(cy - nh / 2)), int(round(nw)), int(round(nh)))
