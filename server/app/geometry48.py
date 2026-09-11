"""
Geometry derived from the CatFLW 48-point scheme (Martvel et al., via Kaggle;
CC BY-NC 4.0 - training/README.md has the credit and licence details).

Index meaning below was NOT taken from the paper or any secondhand summary -
those disagreed with each other on whether eyelid points even exist. It was
read directly off the real label files by rendering every one of the 48
points on several real photos and checking the same index lands on the same
anatomical spot across different cats (see the training session log). What
that confirmed:

    per eye (7-8 pts): 1 outer corner, 1 inner corner, a 3-4 point upper
                       eyelid cluster, 1 lower eyelid point
    left eye:  4 outer, 5 inner, {6,36,37} upper lid, 3 upper-mid, 7 lower
    right eye: 8 outer, 9 inner, {10,39,40,41} upper lid, 1 upper-mid, 11 lower
    whisker pads (3 pts each): left {33, 42, 46}, right {34, 43, 47}
    ears (4-5 pts each, tip + base): left {22,23,24,25,26} (24=tip),
                                     right {27,28,29,30} (28=tip)

Left/right follow the same convention as app/geometry.py's 8-point scheme:
image-left and image-right, not the cat's own left/right.

Four points (31, 32, 35, 38) did not resolve to a clear region in that visual
check - likely cheek/temple contour - and are not used here.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

Point = Tuple[float, float]

LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_LOWER, LEFT_EYE_MID = 4, 5, 7, 3
LEFT_EYE_UPPER = (6, 36, 37)
RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_LOWER, RIGHT_EYE_MID = 8, 9, 11, 1
RIGHT_EYE_UPPER = (10, 39, 40, 41)

LEFT_WHISKER_PAD = (33, 42, 46)
RIGHT_WHISKER_PAD = (34, 43, 47)

N_POINTS = 48


def _dist(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _centroid(pts: List[Point]) -> Point:
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


@dataclass(frozen=True)
class EyeAperture:
    left: float
    """Eyelid gap / eye width for the left eye (image-left). ~0 = shut,
    rising with how round-open the eye is. Undefined (None) only if the
    eye's own width collapses to zero - should not happen on a real face."""

    right: float
    average: float


def eye_aperture(points: List[Optional[Point]]) -> Optional[EyeAperture]:
    """The human-face 'Eye Aspect Ratio' technique (eyelid-gap / eye-width),
    applied here with real cat eyelid points for the first time in this
    project - no prior art for cats was found when this was researched.
    Where EAR distinguishes a blink from an open eye, this is aimed at the
    same physical quantity the FGS calls 'orbital tightening': a
    frightened, wide-eyed cat opens the aperture past its resting width; a
    squinting or half-closed eye narrows it. Not yet calibrated against
    real photos - see training/README.md for what that calibration needs
    to show before this is wired into a reading.

    Returns None if any of the required points are missing (a landmark
    model that skips low-confidence points, unlike dlib's shape predictor
    which always returns all 48)."""
    def one_eye(outer_i, inner_i, upper_is, lower_i):
        idx = (outer_i, inner_i, lower_i, *upper_is)
        if any(points[i] is None for i in idx):
            return None
        outer, inner, lower = points[outer_i], points[inner_i], points[lower_i]
        upper = _centroid([points[i] for i in upper_is])
        width = _dist(outer, inner)
        if width < 1e-6:
            return None
        return _dist(upper, lower) / width

    left = one_eye(LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_UPPER, LEFT_EYE_LOWER)
    right = one_eye(RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER)
    if left is None and right is None:
        return None
    if left is None:
        return EyeAperture(left=right, right=right, average=right)
    if right is None:
        return EyeAperture(left=left, right=left, average=left)
    return EyeAperture(left=left, right=right, average=(left + right) / 2)


def whisker_pad_spread(points: List[Optional[Point]], interocular_dist: float) -> Optional[float]:
    """Distance between the two whisker pads' outermost points (46, 47),
    normalised by interocular distance.

    Marked experimental deliberately: these are whisker-FOLLICLE base
    points on the skin, not points along the whiskers themselves, and a
    whisker pushed forward or bunched is largely the whisker hair moving,
    which these points may not track at all. This may end up measuring
    muzzle width instead of anything whisker-specific. Do not feed this
    into a reading until it is checked against real photos where the
    whisker position is visibly different (e.g. the hugged-cat photo)
    and shown to actually move the way the app would need it to."""
    if points[46] is None or points[47] is None or interocular_dist < 1e-6:
        return None
    return _dist(points[46], points[47]) / interocular_dist
