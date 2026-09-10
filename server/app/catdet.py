"""
Whole-cat detection with NanoDet (OpenCV model zoo, Apache-2.0).

Answers one question the face detector cannot: is there a cat in this
photo at all? That splits "no cat face found" into two honest messages -
"I do not see a cat" and "I see a cat, but not its face" - and it lets the
face stages reject candidates that sit outside every cat (a leaf patch, a
person's face).

Model: object_detection_nanodet_2022nov.onnx, COCO 80 classes, 416x416
input, 3.6 MB, from https://github.com/opencv/opencv_zoo (Apache-2.0).
Pre/post-processing follows opencv_zoo/models/object_detection_nanodet/
nanodet.py (Apache-2.0), reduced to what this service needs.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

_MODEL = Path(__file__).resolve().parent.parent / "models" / "object_detection_nanodet_2022nov.onnx"
_INPUT = 416
_STRIDES = (8, 16, 32, 64)
_REG_MAX = 7
_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
_COCO_CAT = 15
_PROB = 0.35
_IOU = 0.6

_net = None
_anchors = None


@dataclass(frozen=True)
class CatBox:
    box: tuple  # (x, y, w, h) in the original image
    score: float


def available() -> bool:
    return _MODEL.exists()


def _load():
    global _net, _anchors
    if _net is None:
        _net = cv2.dnn.readNet(str(_MODEL))
        _anchors = []
        for stride in _STRIDES:
            n = _INPUT // stride
            xs, ys = np.meshgrid(np.arange(n) * stride, np.arange(n) * stride)
            _anchors.append(np.column_stack((xs.flatten() + 0.5 * (stride - 1), ys.flatten() + 0.5 * (stride - 1))))
    return _net, _anchors


def _letterbox(bgr):
    h, w = bgr.shape[:2]
    scale = min(_INPUT / h, _INPUT / w)
    resized = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((_INPUT, _INPUT, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas, scale


def find_cats(bgr) -> List[CatBox]:
    """All cats in the image, highest score first. Empty if the model is missing."""
    if not available():
        return []
    net, anchors = _load()
    img, scale = _letterbox(bgr)
    blob = cv2.dnn.blobFromImage((img.astype(np.float32) - _MEAN) / _STD)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    cls_scores, bbox_preds = outs[::2], outs[1::2]

    boxes, scores = [], []
    project = np.arange(_REG_MAX + 1)
    for stride, cls, reg, anc in zip(_STRIDES, cls_scores, bbox_preds, anchors):
        cls = cls.squeeze(0) if cls.ndim == 3 else cls
        reg = reg.squeeze(0) if reg.ndim == 3 else reg
        e = np.exp(reg.reshape(-1, _REG_MAX + 1))
        dist = (e / e.sum(axis=1, keepdims=True)) @ project
        dist = dist.reshape(-1, 4) * stride
        cat = cls[:, _COCO_CAT]
        keep = cat > _PROB
        if not keep.any():
            continue
        a, d, s = anc[keep], dist[keep], cat[keep]
        x1 = np.clip(a[:, 0] - d[:, 0], 0, _INPUT)
        y1 = np.clip(a[:, 1] - d[:, 1], 0, _INPUT)
        x2 = np.clip(a[:, 0] + d[:, 2], 0, _INPUT)
        y2 = np.clip(a[:, 1] + d[:, 3], 0, _INPUT)
        boxes.append(np.column_stack([x1, y1, x2 - x1, y2 - y1]))
        scores.append(s)
    if not boxes:
        return []
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.astype(float).tolist(), _PROB, _IOU)
    if len(idx) == 0:
        return []
    out = []
    for i in np.array(idx).flatten():
        x, y, w, h = boxes[i] / scale
        out.append(CatBox(box=(int(x), int(y), int(w), int(h)), score=float(scores[i])))
    out.sort(key=lambda c: -c.score)
    return out


def inside_any(box, cats: List[CatBox], slack: float = 0.15) -> bool:
    """True if the box's centre lies inside some cat box (grown by `slack`)."""
    if not cats:
        return True  # no cat knowledge: do not veto anything
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    for c in cats:
        x, y, w, h = c.box
        gx, gy = w * slack, h * slack
        if x - gx <= cx <= x + w + gx and y - gy <= cy <= y + h + gy:
            return True
    return False
