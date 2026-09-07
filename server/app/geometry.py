"""
Pure geometry math on 8-point cat facial landmarks (pycatfd scheme).

Every feature here names the real, documented feline cue it approximates,
and is honest about how loosely it approximates it. An 8-point landmark
scheme gives ear *position*, not full CatFACS action units (AU) - there is
no whisker point, so whisker change can never be measured this way. See
README.md in this directory for the full mapping to the Feline Grimace
Scale's five action units.

Landmark order (pycatfd's CatFaceLandmark enum):
    0 chin, 1 left_eye, 2 left_of_left_ear, 3 left_of_right_ear,
    4 nose, 5 right_eye, 6 right_of_left_ear, 7 right_of_right_ear.
Points are (x, y) in image pixel space, y increasing downward.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Landmarks:
    chin: tuple
    left_eye: tuple
    left_of_left_ear: tuple
    left_of_right_ear: tuple
    nose: tuple
    right_eye: tuple
    right_of_left_ear: tuple
    right_of_right_ear: tuple


def landmarks_from_dict(d: dict) -> Landmarks:
    return Landmarks(**{k: tuple(v) for k, v in d.items()})


def landmarks_from_points(points: list) -> Landmarks:
    """Build Landmarks from a flat list ordered per pycatfd's CatFaceLandmark enum."""
    if len(points) != 8:
        raise ValueError(f"expected 8 landmark points, got {len(points)}")
    (chin, left_eye, lol_ear, lor_ear, nose, right_eye, rol_ear, ror_ear) = points
    return Landmarks(
        chin=tuple(chin), left_eye=tuple(left_eye),
        left_of_left_ear=tuple(lol_ear), left_of_right_ear=tuple(lor_ear),
        nose=tuple(nose), right_eye=tuple(right_eye),
        right_of_left_ear=tuple(rol_ear), right_of_right_ear=tuple(ror_ear),
    )


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_deg(a, b):
    """Angle of the line a->b from horizontal, in degrees. Positive = tilts down to the right."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


@dataclass(frozen=True)
class FaceGeometry:
    interocular_dist: float
    """Distance between the eyes in pixels. The normalizing scale for every ratio below -
    standard practice in the facial-landmark literature (Normalized Mean Error)."""

    head_tilt_deg: float
    """Angle of the eye-to-eye line from horizontal. Approximates FGS 'head position' -
    a cat in pain often holds its head lower or tilted rather than level."""

    ear_splay_ratio: float
    """Distance between the two ears' outer points, divided by interocular distance.
    A relaxed, forward-eared cat reads differently here than one with ears pulled
    in and back - but this is head width, not ear rotation, so treat it as a coarse
    stand-in for FGS 'ear position', not a direct measurement of it."""

    left_ear_angle_deg: float
    right_ear_angle_deg: float
    """Angle of each ear's own base line (its two landmark points) from horizontal.
    A forward, upright ear's base tends toward flat; a flattened or swept-back ear
    tilts more steeply. This is the single closest proxy to FGS 'ear position' that
    an 8-point scheme allows - it still cannot see rotation toward or away from the
    camera, only the base's tilt in the image plane."""

    muzzle_ratio: float
    """Chin-to-eyeline distance, divided by interocular distance. A rough stand-in
    for FGS 'muzzle tension' - a tense, drawn-back muzzle tends to shorten this
    distance relative to a relaxed, open face. No whisker point exists in this
    scheme, so whisker change (a separate FGS action unit) cannot be measured."""

    nose_symmetry: float
    """1.0 when the nose sits exactly on the eye-line midpoint's perpendicular,
    falling toward 0 as it skews to one side. This is a face-orientation sanity
    check, not an emotional cue - a low value usually means the detector found
    a profile or poorly-cropped face, and the reading should be treated as less
    reliable."""


def face_geometry(lm: Landmarks) -> FaceGeometry:
    iod = _dist(lm.left_eye, lm.right_eye)
    if iod < 1e-6:
        raise ValueError("left and right eye landmarks coincide - cannot scale geometry")

    head_tilt = _angle_deg(lm.left_eye, lm.right_eye)

    ear_splay = _dist(lm.left_of_left_ear, lm.right_of_right_ear) / iod

    left_ear_angle = _angle_deg(lm.left_of_left_ear, lm.right_of_left_ear)
    right_ear_angle = _angle_deg(lm.left_of_right_ear, lm.right_of_right_ear)

    eye_mid = ((lm.left_eye[0] + lm.right_eye[0]) / 2, (lm.left_eye[1] + lm.right_eye[1]) / 2)
    muzzle_ratio = _dist(lm.chin, eye_mid) / iod

    nose_offset = abs(lm.nose[0] - eye_mid[0])
    nose_symmetry = max(0.0, 1.0 - (nose_offset / iod))

    return FaceGeometry(
        interocular_dist=iod,
        head_tilt_deg=head_tilt,
        ear_splay_ratio=ear_splay,
        left_ear_angle_deg=left_ear_angle,
        right_ear_angle_deg=right_ear_angle,
        muzzle_ratio=muzzle_ratio,
        nose_symmetry=nose_symmetry,
    )
