"""
Photometric features measured on a cropped face region only - not the
whole photo. This is the fix for the flaw where a sunlit room or dark
sofa drove the reading instead of the cat's own face.
"""
import numpy as np
import pytest
from app.photometrics import face_light


def solid(h, w, gray):
    return np.full((h, w, 3), gray, dtype=np.uint8)


class TestFaceLight:
    def test_black_crop_is_dark_and_flat(self):
        f = face_light(solid(40, 40, 0))
        assert f.brightness == pytest.approx(0.0, abs=0.02)
        assert f.contrast == pytest.approx(0.0, abs=0.02)
        assert f.sharpness == pytest.approx(0.0, abs=0.02)

    def test_white_crop_is_bright(self):
        f = face_light(solid(40, 40, 255))
        assert f.brightness == pytest.approx(1.0, abs=0.02)

    def test_half_black_half_white_is_high_contrast(self):
        img = solid(40, 40, 0)
        img[:20, :, :] = 255
        f = face_light(img)
        assert f.brightness == pytest.approx(0.5, abs=0.05)
        assert f.contrast > 0.6

    def test_checkerboard_is_sharper_than_flat(self):
        h = w = 32
        board = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                if (x + y) % 2 == 0:
                    board[y, x] = 255
        flat = solid(h, w, 128)
        assert face_light(board).sharpness > face_light(flat).sharpness

    def test_dark_ratio_reflects_shadow_coverage(self):
        assert face_light(solid(20, 20, 10)).dark_ratio > 0.9
        assert face_light(solid(20, 20, 240)).dark_ratio < 0.1

    def test_rejects_empty_crop(self):
        with pytest.raises(ValueError):
            face_light(np.zeros((0, 0, 3), dtype=np.uint8))
