"""
Synthetic tests for the CatFLW 48-point geometry (eye aperture, whisker
spread). No real photos here on purpose - these lock in the arithmetic
against hand-built point sets; test_pipeline_integration.py-style
real-photo calibration happens once the landmark model is trained and
its per-point accuracy is known (see training/README.md).
"""
from app.geometry48 import N_POINTS, eye_aperture, whisker_pad_spread


def blank_points():
    return [None] * N_POINTS


def make_eye(pts, outer, inner, upper_ids, lower, outer_xy, inner_xy, upper_y, lower_y):
    pts[outer] = outer_xy
    pts[inner] = inner_xy
    pts[lower] = (0.5 * (outer_xy[0] + inner_xy[0]), lower_y)
    for i in upper_ids:
        pts[i] = (0.5 * (outer_xy[0] + inner_xy[0]), upper_y)


def wide_open_face():
    pts = blank_points()
    # left eye: width 20 (x=0..20), aperture gap 10 (y=0 upper, y=10 lower)
    make_eye(pts, 4, 5, (6, 36, 37), 7, (0, 5), (20, 5), 0, 10)
    pts[3] = (10, 5)
    # right eye: same shape, mirrored
    make_eye(pts, 8, 9, (10, 39, 40, 41), 11, (60, 5), (40, 5), 0, 10)
    pts[1] = (50, 5)
    return pts


def test_wide_open_eyes_read_a_large_aperture():
    e = eye_aperture(wide_open_face())
    assert e is not None
    assert e.left == 0.5  # gap 10 / width 20
    assert e.right == 0.5
    assert e.average == 0.5


def test_narrowing_the_gap_lowers_the_ratio():
    pts = wide_open_face()
    # squeeze the left eye's gap from 10 to 2, width unchanged
    pts[7] = (10, 7)
    e = eye_aperture(pts)
    assert e.left < 0.5
    assert e.right == 0.5  # the other eye is untouched


def test_missing_one_eye_falls_back_to_the_other():
    pts = wide_open_face()
    pts[4] = None  # left eye's outer corner is missing
    e = eye_aperture(pts)
    assert e.left == e.right  # left couldn't be computed, so it mirrors right
    assert e.average == e.right


def test_missing_both_eyes_returns_none():
    pts = blank_points()
    assert eye_aperture(pts) is None


def test_zero_width_eye_does_not_explode():
    pts = wide_open_face()
    pts[4] = pts[5]  # collapse the left eye's width to zero
    e = eye_aperture(pts)
    assert e is not None
    assert e.left == e.right  # left was undefined, fell back to right


def test_whisker_spread_scales_with_interocular_distance():
    pts = blank_points()
    pts[46] = (0.0, 0.0)
    pts[47] = (50.0, 0.0)
    assert whisker_pad_spread(pts, interocular_dist=25.0) == 2.0
    assert whisker_pad_spread(pts, interocular_dist=50.0) == 1.0


def test_whisker_spread_missing_point_returns_none():
    pts = blank_points()
    pts[46] = (0.0, 0.0)
    assert whisker_pad_spread(pts, interocular_dist=25.0) is None


def test_whisker_spread_zero_interocular_distance_returns_none():
    pts = blank_points()
    pts[46], pts[47] = (0.0, 0.0), (1.0, 0.0)
    assert whisker_pad_spread(pts, interocular_dist=0.0) is None
