"""
Eye measurements taken from the image around the two eye landmarks.

This exists because the loudest signal in a cat's face - how wide the
pupils are - is invisible to an 8-point landmark scheme (it has eye
*centres* and nothing else). Pupil dilation is a documented arousal cue:
wide pupils go with alert, excited, hunting, or frightened states; slit
pupils with a relaxed cat in ordinary light. It is measured on the cat's
own eyes, not on the room, which is the whole point.

Method: a disc around each eye centre, radius ~0.22 x interocular
distance (the eyeball, excluding surrounding fur). The fraction of that
disc darker than a black-ish threshold approximates how much of the
visible eye is pupil. Measured on real photos with correctly detected
faces: an alert cat with visibly huge pupils read 0.64 averaged across
both eyes; four calm cats read 0.15-0.38; a hissing cat read 0.57.

Known confounds, handled by refusing to answer rather than guessing:
  - Small faces. Below ~9px of eye radius the disc is a handful of pixels
    and the number is noise. `usable` goes False.
  - Retinal glow ("tapetum"). Flash or bright backlight makes dilated
    pupils reflect *bright*, reading as ~0.00 dark - the opposite of the
    truth. A blown-out eye disc sets `usable` False.
  - No pupil in view. A flash reflection, a squinted eye, or a landmark
    that missed the eye leaves a disc with no dark core at all. That used
    to read as "0.00 = narrow pupils" and turned frightened cats into
    calm ones. Now an eye with no dark core is unreadable, and if neither
    eye is readable `usable` goes False.
  - Ambient light. Pupils also dilate in dim rooms regardless of mood.
    This is a real physiological confound with no fix short of knowing
    the light level; it is disclosed rather than hidden.
"""
from dataclasses import dataclass
import cv2
import numpy as np

from .geometry import Landmarks

_EYE_RADIUS_FRACTION = 0.22
_DISC_FRACTION = 0.85  # keep the disc inside the eyeball, off the lid and fur
_MIN_EYE_RADIUS_PX = 9
_DARK_THRESHOLD = 70  # grayscale 0-255; below this reads as pupil
_GLOW_THRESHOLD = 200  # mean disc brightness above this = reflective glow
_NO_PUPIL_P5 = 80
"""If even the darkest 5% of the eye disc is brighter than this, there is
no pupil in view at all - the eye is reflecting a flash, squinted shut,
or the landmark missed the eye. Measured on real photos: every eye with
a visible pupil, even a slit, had a 5th percentile of 3-66; the eyes that
read a false "0.00 narrow" had 82-164. Such an eye is not "narrow", it is
unreadable, and saying so is the honest answer."""


@dataclass(frozen=True)
class EyeSignals:
    pupil_dilation: float
    """0..1: fraction of the eyeball disc that reads as pupil, averaged over
    both eyes. Only meaningful when `usable` is True; 0.5 (neutral) otherwise."""

    usable: bool
    """False when the face was too small to measure, or an eye was glowing."""

    left_raw: float
    right_raw: float
    """Per-eye dark fractions, for diagnostics. NaN when not measured."""


def _disc_dark_fraction(gray, centre, radius):
    cx, cy = int(centre[0]), int(centre[1])
    h, w = gray.shape
    y0, y1 = max(0, cy - radius), min(h, cy + radius)
    x0, x1 = max(0, cx - radius), min(w, cx + radius)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return float("nan"), float("nan")
    yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
    mask = (yy - crop.shape[0] / 2) ** 2 + (xx - crop.shape[1] / 2) ** 2 <= (radius * _DISC_FRACTION) ** 2
    px = crop[mask]
    if px.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float((px < _DARK_THRESHOLD).mean()), float(px.mean()), float(np.percentile(px, 5))


def _readable(dark, mean, p5) -> bool:
    """An eye counts only if it is in frame, not glowing, and shows a pupil."""
    return not np.isnan(dark) and mean <= _GLOW_THRESHOLD and p5 <= _NO_PUPIL_P5


def eye_signals(bgr, lm: Landmarks, interocular_dist: float) -> EyeSignals:
    radius = int(interocular_dist * _EYE_RADIUS_FRACTION)
    if radius < _MIN_EYE_RADIUS_PX:
        return EyeSignals(pupil_dilation=0.5, usable=False, left_raw=float("nan"), right_raw=float("nan"))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    left = _disc_dark_fraction(gray, lm.left_eye, radius)
    right = _disc_dark_fraction(gray, lm.right_eye, radius)

    # Use every readable eye; one good eye beats a guess from two bad ones.
    readable = [e[0] for e in (left, right) if _readable(*e)]
    if not readable:
        return EyeSignals(pupil_dilation=0.5, usable=False, left_raw=left[0], right_raw=right[0])

    return EyeSignals(
        pupil_dilation=float(np.mean(readable)),
        usable=True,
        left_raw=left[0],
        right_raw=right[0],
    )
