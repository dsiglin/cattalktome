"""
End to end: photo bytes -> detected cat face -> geometry + eye signals
-> a feeling reading. Everything downstream of detect_cat_face is pure
and unit-tested; this module just wires it together.
"""
from dataclasses import dataclass
import numpy as np
import cv2

from .detect import detect_cat_face, DetectedFace
from .eyes import eye_signals, EyeSignals
from .geometry import landmarks_from_points, face_geometry, FaceGeometry
from .feelings import read_feeling, Reading


@dataclass(frozen=True)
class AnalysisResult:
    reading: Reading
    geometry: FaceGeometry
    eyes: EyeSignals
    face: DetectedFace


def analyse_bytes(image_bytes: bytes) -> AnalysisResult:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image bytes")
    return analyse_image(bgr)


def analyse_image(bgr: np.ndarray) -> AnalysisResult:
    face = detect_cat_face(bgr)
    lm = landmarks_from_points(face.landmarks)
    geometry = face_geometry(lm)
    eyes = eye_signals(bgr, lm, geometry.interocular_dist)
    reading = read_feeling(geometry, eyes, face_reliable=face.reliable)
    return AnalysisResult(reading=reading, geometry=geometry, eyes=eyes, face=face)
