"""
Cat face detection and 8-point landmark extraction.

Detection is staged, most-trustworthy first. This shape came out of a real
failure: on an ordinary phone photo of an alert cat with its head pitched
up ~10 degrees, a single loosened Haar pass found *only* a 90px false
positive on the cat's eye - and the landmark predictor then fitted an
entire face inside that eyeball, so every downstream measurement was
garbage delivered with 74% confidence. Meanwhile the strict pass found
nothing, which at least was honest.

Measured across 11 real photos (see tests/test_pipeline_integration.py):

  Stage 1 - strict Haar, unrotated (scaleFactor 1.05, minNeighbors 3).
            Found the correct face on every ordinary photo (IoU >= 0.98).
            Its only misses are pitched/rotated/extreme heads. Largest box
            wins here - on this stage, that rule was never wrong.
  Stage 2 - both cat cascades on rotated copies (+/-20, +/-35 degrees),
            still strict. Recovers pitched heads. A box needs >= 2 agreeing
            detections (clustered by IoU), because a single rotated hit is
            often a warp artifact. Rotation uses black borders and rejects
            boxes touching the padded corners for the same reason.
  Stage 3 - loose Haar, unrotated (1.02 / 2). Recovers extreme poses
            (a hissing Sphynx in profile) but also produces false positives,
            so anything found here is flagged `reliable=False` and the app
            hedges instead of asserting.

pycatfd's own FHOG detector was dropped: it found nothing on any of the
11 photos, at any upsampling, so it was only adding latency.

Landmarks: pycatfd's dlib shape predictor (8 points), which fits *some*
face inside any box it is handed - which is exactly why box quality has
to be settled before the predictor ever runs.
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import dlib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_CASCADE_EXT_PATH = MODELS_DIR / "haarcascade_frontalcatface_extended.xml"
_CASCADE_STD_PATH = MODELS_DIR / "haarcascade_frontalcatface.xml"
_PREDICTOR_DAT_PATH = MODELS_DIR / "predictor.dat"

_MIN_FACE_PX = 75
_ROTATIONS_DEG = (-20, 20, -35, 35)
_CLUSTER_IOU = 0.3
_MIN_VOTES = 2

_cascade_ext = None
_cascade_std = None
_predictor = None


def _load_cascade(path: Path) -> cv2.CascadeClassifier:
    cas = cv2.CascadeClassifier(str(path))
    if cas.empty():
        raise RuntimeError(f"failed to load cascade at {path}")
    return cas


def _ext():
    global _cascade_ext
    if _cascade_ext is None:
        _cascade_ext = _load_cascade(_CASCADE_EXT_PATH)
    return _cascade_ext


def _std():
    global _cascade_std
    if _cascade_std is None:
        _cascade_std = _load_cascade(_CASCADE_STD_PATH)
    return _cascade_std


def _shape_predictor():
    global _predictor
    if _predictor is None:
        _predictor = dlib.shape_predictor(str(_PREDICTOR_DAT_PATH))
    return _predictor


@dataclass(frozen=True)
class DetectedFace:
    box: tuple  # (x, y, w, h) in the original image
    landmarks: list  # 8 (x, y) points, pycatfd CatFaceLandmark order
    detector: str  # "haar", "haar-rotated", or "haar-loose" - which stage found it
    reliable: bool  # False when the box came from the loose stage


class NoCatFaceFound(Exception):
    pass


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


def _largest(boxes):
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return (int(x), int(y), int(w), int(h))


def _stage_strict(gray):
    boxes = _ext().detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(_MIN_FACE_PX, _MIN_FACE_PX))
    return _largest(boxes) if len(boxes) else None


def _rotate(gray, deg):
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    rotated = cv2.warpAffine(gray, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated, m


def _unrotate_box(box, m):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    ux, uy = cv2.invertAffineTransform(m) @ np.array([cx, cy, 1.0])
    return (int(ux - w / 2), int(uy - h / 2), int(w), int(h))


def _median_box(cluster):
    return tuple(int(v) for v in np.median(np.array(cluster), axis=0))


def _stage_rotated(gray):
    h, w = gray.shape
    candidates = []
    for cascade in (_ext(), _std()):
        for deg in _ROTATIONS_DEG:
            rotated, m = _rotate(gray, deg)
            for b in cascade.detectMultiScale(rotated, scaleFactor=1.05, minNeighbors=3, minSize=(_MIN_FACE_PX, _MIN_FACE_PX)):
                x, y, bw, bh = _unrotate_box(b, m)
                # A box poking outside the real image sits on the rotation's padded corners.
                if x < 0 or y < 0 or x + bw > w or y + bh > h:
                    continue
                candidates.append((x, y, bw, bh))

    # Cluster overlapping candidates; a real face is hit from several angles,
    # a warp artifact usually only once.
    clusters = []
    for box in candidates:
        for cluster in clusters:
            if _iou(box, _median_box(cluster)) >= _CLUSTER_IOU:
                cluster.append(box)
                break
        else:
            clusters.append([box])

    supported = [c for c in clusters if len(c) >= _MIN_VOTES]
    if not supported:
        return None
    best = max(supported, key=lambda c: (len(c), _median_box(c)[2] * _median_box(c)[3]))
    return _median_box(best)


def _stage_loose(gray):
    boxes = _ext().detectMultiScale(gray, scaleFactor=1.02, minNeighbors=2, minSize=(_MIN_FACE_PX, _MIN_FACE_PX))
    return _largest(boxes) if len(boxes) else None


def detect_cat_face(bgr) -> DetectedFace:
    """Find a cat face and its 8 landmarks in a BGR image (as loaded by cv2.imread)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    box, detector, reliable = None, "", False
    for stage_fn, name, trust in (
        (_stage_strict, "haar", True),
        (_stage_rotated, "haar-rotated", True),
        (_stage_loose, "haar-loose", False),
    ):
        box = stage_fn(gray)
        if box is not None:
            detector, reliable = name, trust
            break

    if box is None:
        raise NoCatFaceFound("no cat face found at any detection stage")

    x, y, w, h = box
    rect = dlib.rectangle(x, y, x + w, y + h)
    shape = _shape_predictor()(bgr, rect)
    points = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

    return DetectedFace(box=box, landmarks=points, detector=detector, reliable=reliable)
