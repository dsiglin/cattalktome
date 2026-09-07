"""
Tests for pure geometry math on cat facial landmarks.

Landmark order matches pycatfd's CatFaceLandmark enum:
0 chin, 1 left_eye, 2 left_of_left_ear, 3 left_of_right_ear,
4 nose, 5 right_eye, 6 right_of_left_ear, 7 right_of_right_ear.

All points are (x, y) in image pixel space, y increasing downward.
"""
import math
import pytest
from app.geometry import (
    landmarks_from_dict,
    face_geometry,
)


def pts(chin, left_eye, left_of_left_ear, left_of_right_ear, nose, right_eye, right_of_left_ear, right_of_right_ear):
    return landmarks_from_dict({
        "chin": chin, "left_eye": left_eye,
        "left_of_left_ear": left_of_left_ear, "left_of_right_ear": left_of_right_ear,
        "nose": nose, "right_eye": right_eye,
        "right_of_left_ear": right_of_left_ear, "right_of_right_ear": right_of_right_ear,
    })


def upright_face(scale=100.0):
    """A symmetric, perfectly upright, forward-facing synthetic cat face."""
    return pts(
        chin=(50 * scale / 100, 80 * scale / 100),
        left_eye=(30 * scale / 100, 40 * scale / 100),
        right_eye=(70 * scale / 100, 40 * scale / 100),
        nose=(50 * scale / 100, 55 * scale / 100),
        left_of_left_ear=(10 * scale / 100, 10 * scale / 100),
        right_of_left_ear=(30 * scale / 100, 5 * scale / 100),
        left_of_right_ear=(70 * scale / 100, 5 * scale / 100),
        right_of_right_ear=(90 * scale / 100, 10 * scale / 100),
    )


class TestInterocularDistance:
    def test_measures_eye_separation(self):
        g = face_geometry(upright_face())
        assert g.interocular_dist == pytest.approx(40.0, abs=0.01)

    def test_scales_with_face_size(self):
        small = face_geometry(upright_face(scale=50))
        large = face_geometry(upright_face(scale=200))
        assert large.interocular_dist == pytest.approx(small.interocular_dist * 4, rel=0.01)

    def test_rejects_coincident_eyes(self):
        bad = pts(
            chin=(50, 80), left_eye=(50, 40), right_eye=(50, 40),
            nose=(50, 55), left_of_left_ear=(10, 10), right_of_left_ear=(30, 5),
            left_of_right_ear=(70, 5), right_of_right_ear=(90, 10),
        )
        with pytest.raises(ValueError):
            face_geometry(bad)


class TestHeadTilt:
    def test_is_near_zero_for_a_level_head(self):
        g = face_geometry(upright_face())
        assert abs(g.head_tilt_deg) < 1.0

    def test_detects_a_tilted_head(self):
        tilted = pts(
            chin=(50, 80), left_eye=(30, 30), right_eye=(70, 50),
            nose=(50, 55), left_of_left_ear=(10, 10), right_of_left_ear=(30, 5),
            left_of_right_ear=(70, 15), right_of_right_ear=(90, 25),
        )
        g = face_geometry(tilted)
        assert g.head_tilt_deg > 10

    def test_sign_flips_with_tilt_direction(self):
        right_down = pts(
            chin=(50, 80), left_eye=(30, 30), right_eye=(70, 50),
            nose=(50, 55), left_of_left_ear=(10, 10), right_of_left_ear=(30, 5),
            left_of_right_ear=(70, 15), right_of_right_ear=(90, 25),
        )
        left_down = pts(
            chin=(50, 80), left_eye=(30, 50), right_eye=(70, 30),
            nose=(50, 55), left_of_left_ear=(10, 15), right_of_left_ear=(30, 25),
            left_of_right_ear=(70, 5), right_of_right_ear=(90, 10),
        )
        g1 = face_geometry(right_down)
        g2 = face_geometry(left_down)
        assert g1.head_tilt_deg * g2.head_tilt_deg < 0


class TestEarSplay:
    def test_wide_set_ears_score_higher_than_narrow(self):
        wide = face_geometry(upright_face())
        narrow = pts(
            chin=(50, 80), left_eye=(30, 40), right_eye=(70, 40),
            nose=(50, 55),
            left_of_left_ear=(35, 10), right_of_left_ear=(45, 5),
            left_of_right_ear=(55, 5), right_of_right_ear=(65, 10),
        )
        g_narrow = face_geometry(narrow)
        assert wide.ear_splay_ratio > g_narrow.ear_splay_ratio

    def test_is_normalized_by_interocular_distance(self):
        small = face_geometry(upright_face(scale=50))
        large = face_geometry(upright_face(scale=200))
        assert small.ear_splay_ratio == pytest.approx(large.ear_splay_ratio, rel=0.02)


class TestEarBaseAngle:
    def test_flat_ear_base_reads_near_zero(self):
        # left ear base perfectly horizontal
        flat = pts(
            chin=(50, 80), left_eye=(30, 40), right_eye=(70, 40),
            nose=(50, 55),
            left_of_left_ear=(10, 10), right_of_left_ear=(30, 10),
            left_of_right_ear=(70, 10), right_of_right_ear=(90, 10),
        )
        g = face_geometry(flat)
        assert abs(g.left_ear_angle_deg) < 1.0
        assert abs(g.right_ear_angle_deg) < 1.0

    def test_swept_back_ear_has_a_steeper_angle(self):
        swept = pts(
            chin=(50, 80), left_eye=(30, 40), right_eye=(70, 40),
            nose=(50, 55),
            left_of_left_ear=(10, 30), right_of_left_ear=(30, 5),
            left_of_right_ear=(70, 5), right_of_right_ear=(90, 30),
        )
        g = face_geometry(swept)
        assert abs(g.left_ear_angle_deg) > 30
        assert abs(g.right_ear_angle_deg) > 30


class TestMuzzleLength:
    def test_measures_chin_to_eyeline_distance(self):
        g = face_geometry(upright_face())
        # chin is 40px below the eye line in the fixture (y=80 vs y=40)
        assert g.muzzle_ratio == pytest.approx(1.0, rel=0.05)

    def test_normalized_by_interocular_distance(self):
        small = face_geometry(upright_face(scale=50))
        large = face_geometry(upright_face(scale=200))
        assert small.muzzle_ratio == pytest.approx(large.muzzle_ratio, rel=0.02)


class TestFaceSymmetry:
    def test_symmetric_face_scores_near_one(self):
        g = face_geometry(upright_face())
        assert g.nose_symmetry > 0.9

    def test_off_center_nose_lowers_symmetry(self):
        skewed = pts(
            chin=(50, 80), left_eye=(30, 40), right_eye=(70, 40),
            nose=(65, 55),
            left_of_left_ear=(10, 10), right_of_left_ear=(30, 5),
            left_of_right_ear=(70, 5), right_of_right_ear=(90, 10),
        )
        g = face_geometry(skewed)
        assert g.nose_symmetry < 0.7


class TestAllFeaturesFinite:
    def test_no_nans_or_infinities(self):
        g = face_geometry(upright_face())
        for name, value in g.__dict__.items():
            assert math.isfinite(value), f"{name} is not finite"
