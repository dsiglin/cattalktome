"""
Whole-cat detection (NanoDet, OpenCV model zoo) on real fixtures.
"""
from pathlib import Path

import cv2
import numpy as np

from app import catdet

FIXTURES = Path(__file__).resolve().parent.parent.parent / "test" / "fixtures"


def load(name):
    return cv2.imread(str(FIXTURES / name))


def test_model_is_present():
    assert catdet.available(), f"expected {catdet._MODEL}"


def test_finds_the_lounging_cat_and_the_box_contains_its_face():
    cats = catdet.find_cats(load("cat.jpg"))
    assert cats, "no cat found in cat.jpg"
    x, y, w, h = cats[0].box
    fx, fy, fw, fh = 784, 215, 158, 158  # the strict-stage face box
    assert x <= fx + fw / 2 <= x + w and y <= fy + fh / 2 <= y + h
    assert cats[0].score > 0.5


def test_finds_the_cat_in_profile():
    # The face is not readable in profile, but the cat is plainly there.
    cats = catdet.find_cats(load("orange-cat-profile-1400.jpg"))
    assert cats and cats[0].score > 0.5


def test_finds_the_wide_eyed_tabby():
    cats = catdet.find_cats(load("wide-eyed-tabby-1400.jpg"))
    assert cats and cats[0].score > 0.5


def test_no_cat_in_a_blank_image():
    blank = np.full((600, 400, 3), 128, dtype=np.uint8)
    assert catdet.find_cats(blank) == []


def test_inside_any_uses_the_box_centre():
    cats = [catdet.CatBox(box=(100, 100, 200, 200), score=0.9)]
    assert catdet.inside_any((150, 150, 50, 50), cats)
    assert not catdet.inside_any((400, 400, 50, 50), cats)
    assert catdet.inside_any((400, 400, 50, 50), [])  # no cat knowledge: no veto
