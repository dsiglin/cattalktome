"""
End to end: photo bytes -> detected cat face -> geometry + photometrics
-> a feeling reading. Everything downstream of detect_cat_face is pure
and unit-tested; this module just wires it together.
"""
from dataclasses import dataclass
from typing import List
import numpy as np
import cv2

from .detect import detect_cat_face, DetectedFace, NoCatFaceFound
from .geometry import landmarks_from_points, face_geometry, FaceGeometry
from .photometrics import face_light, FaceLight
from .feelings import read_feeling, Reading


@dataclass(frozen=True)
class AnalysisResult:
    reading: Reading
    geometry: FaceGeometry
    light: FaceLight
    face: DetectedFace


def analyse_bytes(image_bytes: bytes) -> AnalysisResult:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image bytes")
    return analyse_image(bgr)


def analyse_image(bgr: np.ndarray) -> AnalysisResult:
    face = detect_cat_face(bgr)
    x, y, w, h = face.box
    crop = bgr[max(0, y):y + h, max(0, x):x + w]

    lm = landmarks_from_points(face.landmarks)
    geometry = face_geometry(lm)
    light = face_light(crop)
    reading = read_feeling(geometry, light)

    return AnalysisResult(reading=reading, geometry=geometry, light=light, face=face)
