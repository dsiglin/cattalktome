"""
Cat face detection and 8-point landmark extraction.

Detection: OpenCV's cat-face Haar cascade (classical, ~400KB, ships with
every OpenCV install) - fast and well-tested, tried first.
Fallback: pycatfd's own dlib FHOG detector, trained specifically to pair
with its shape predictor, tried only if the Haar cascade finds nothing.

Landmarks: pycatfd's dlib shape predictor (8 points), which works on any
detected rectangle regardless of which detector produced it.
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import dlib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_CASCADE_PATH = MODELS_DIR / "haarcascade_frontalcatface_extended.xml"
_DETECTOR_SVM_PATH = MODELS_DIR / "detector.svm"
_PREDICTOR_DAT_PATH = MODELS_DIR / "predictor.dat"

_cascade = None
_fhog_detector = None
_predictor = None


def _cascade_detector():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(str(_CASCADE_PATH))
        if _cascade.empty():
            raise RuntimeError(f"failed to load cascade at {_CASCADE_PATH}")
    return _cascade


def _fhog():
    global _fhog_detector
    if _fhog_detector is None:
        _fhog_detector = dlib.fhog_object_detector(str(_DETECTOR_SVM_PATH))
    return _fhog_detector


def _shape_predictor():
    global _predictor
    if _predictor is None:
        _predictor = dlib.shape_predictor(str(_PREDICTOR_DAT_PATH))
    return _predictor


@dataclass(frozen=True)
class DetectedFace:
    box: tuple  # (x, y, w, h) in the original image
    landmarks: list  # 8 (x, y) points, pycatfd CatFaceLandmark order
    detector: str  # "haar" or "fhog", for diagnostics


class NoCatFaceFound(Exception):
    pass


def detect_cat_face(bgr) -> DetectedFace:
    """Find a cat face and its 8 landmarks in a BGR image (as loaded by cv2.imread)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    boxes = _cascade_detector().detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(75, 75),
    )
    detector_name = "haar"

    rect = None
    if len(boxes) > 0:
        # Largest box - the most likely the primary subject, not a background cat.
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    else:
        detector_name = "fhog"
        dets = _fhog()(bgr, 1)
        if len(dets) > 0:
            rect = max(dets, key=lambda d: d.width() * d.height())

    if rect is None:
        raise NoCatFaceFound("no cat face found by either the Haar cascade or the FHOG detector")

    shape = _shape_predictor()(bgr, rect)
    points = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

    box = (rect.left(), rect.top(), rect.width(), rect.height())
    return DetectedFace(box=box, landmarks=points, detector=detector_name)
