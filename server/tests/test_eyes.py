"""
Tests for pupil measurement around the eye landmarks.

Synthetic eyes: a disc of some brightness on a mid-grey "fur" background,
centred on the landmark. The measure should read a black disc as a wide
pupil, a light disc as a narrow one, refuse to answer on a tiny face, and
refuse on a blown-out (glowing) eye.
"""
import numpy as np
import cv2
from app.eyes import eye_signals
from app.geometry import Landmarks


def face_with_eyes(eye_value, iod=100, size=400, fur=128):
    """Grey square with two discs (the eyeballs) at a given brightness."""
    img = np.full((size, size, 3), fur, dtype=np.uint8)
    cy = size // 2
    lx, rx = size // 2 - iod // 2, size // 2 + iod // 2
    r = int(iod * 0.22)
    cv2.circle(img, (lx, cy), r, (eye_value,) * 3, -1)
    cv2.circle(img, (rx, cy), r, (eye_value,) * 3, -1)
    lm = Landmarks(
        chin=(size // 2, size - 40), left_eye=(lx, cy), right_eye=(rx, cy), nose=(size // 2, cy + 40),
        left_of_left_ear=(lx - 30, cy - 80), right_of_left_ear=(lx, cy - 100),
        left_of_right_ear=(rx, cy - 100), right_of_right_ear=(rx + 30, cy - 80),
    )
    return img, lm, float(iod)


def test_black_eyeballs_read_as_wide_pupils():
    img, lm, iod = face_with_eyes(eye_value=10)
    e = eye_signals(img, lm, iod)
    assert e.usable
    assert e.pupil_dilation > 0.9


def test_light_eyeballs_with_a_slit_pupil_read_as_narrow():
    img, lm, iod = face_with_eyes(eye_value=150)
    # a thin black slit in each light iris - a real narrow pupil
    cy = img.shape[0] // 2
    for x in (img.shape[1] // 2 - 50, img.shape[1] // 2 + 50):
        cv2.rectangle(img, (x - 1, cy - 12), (x + 1, cy + 12), (10, 10, 10), -1)
    e = eye_signals(img, lm, iod)
    assert e.usable
    assert e.pupil_dilation < 0.08


def test_eyeballs_with_no_dark_core_are_unreadable_not_narrow():
    # A flash reflection or a squinted-shut eye leaves no pupil in view.
    # This used to read "0.00 = narrow pupils" and turned frightened cats
    # into calm ones. It must say "could not read" instead.
    img, lm, iod = face_with_eyes(eye_value=150)
    e = eye_signals(img, lm, iod)
    assert e.usable is False
    assert e.pupil_dilation == 0.5


def test_one_readable_eye_is_enough():
    img, lm, iod = face_with_eyes(eye_value=10)
    # blow out the right eye only
    cy = img.shape[0] // 2
    cv2.circle(img, (img.shape[1] // 2 + 50, cy), int(iod * 0.22), (150, 150, 150), -1)
    e = eye_signals(img, lm, iod)
    assert e.usable
    assert e.pupil_dilation > 0.9


def test_partial_pupil_reads_in_between():
    img, lm, iod = face_with_eyes(eye_value=150)
    # paint a smaller black pupil inside each light iris
    cy = img.shape[0] // 2
    for x in (img.shape[1] // 2 - 50, img.shape[1] // 2 + 50):
        cv2.circle(img, (x, cy), int(iod * 0.22 * 0.5), (10, 10, 10), -1)
    e = eye_signals(img, lm, iod)
    assert e.usable
    assert 0.15 < e.pupil_dilation < 0.6


def test_too_small_a_face_is_declared_unusable_not_guessed():
    img, lm, iod = face_with_eyes(eye_value=10, iod=30, size=120)
    e = eye_signals(img, lm, iod)
    assert e.usable is False
    assert e.pupil_dilation == 0.5


def test_glowing_eyes_are_declared_unusable_not_read_as_narrow():
    # Retinal glow: dilated pupils that reflect bright. The naive dark-
    # fraction would read ~0 (narrow) - the opposite of the truth.
    img, lm, iod = face_with_eyes(eye_value=245)
    e = eye_signals(img, lm, iod)
    assert e.usable is False


def test_per_eye_raw_values_are_reported():
    img, lm, iod = face_with_eyes(eye_value=10)
    e = eye_signals(img, lm, iod)
    assert e.left_raw > 0.9 and e.right_raw > 0.9
