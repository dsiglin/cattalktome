"""
Photometric features measured on a cropped face region (BGR uint8, as
OpenCV loads it) - never on the whole photo. Same honest method as the
browser app's features.ts, but scoped to just the cat's face, so a bright
window or dark sofa in the background no longer drives the reading.
"""
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass(frozen=True)
class FaceLight:
    brightness: float
    """Mean luminance of the face crop, 0=black, 1=white."""

    contrast: float
    """Spread of luminance in the face crop, 0=flat, 1=hard light."""

    sharpness: float
    """Edge energy in the face crop, 0=soft/blurred, 1=crisp - stands in
    for stillness against motion."""

    dark_ratio: float
    """Share of very dark pixels in the crop - stands in for deep shadow
    or, per the Feline Grimace Scale literature, wide pupils."""


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def face_light(bgr: np.ndarray) -> FaceLight:
    if bgr.size == 0 or bgr.shape[0] == 0 or bgr.shape[1] == 0:
        raise ValueError("face_light needs a non-empty crop")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    brightness = float(gray.mean())
    contrast = _clamp01(float(gray.std()) * 2)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    raw_sharpness = float(np.abs(lap).mean())
    # Calibrated against real face-crop photographs, not a synthetic
    # checkerboard (whose ~4.0 mean |Laplacian| no real photo ever
    # approaches - measured across several real cat photos, raw values
    # landed between 0.02 and 0.12; a 0.16 reference gives that range
    # real spread instead of clustering everything near zero).
    sharpness = _clamp01(raw_sharpness / 0.16)

    dark_ratio = float((gray < 0.2).mean())

    return FaceLight(
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        dark_ratio=dark_ratio,
    )
