"""
Integration test: run the whole pipeline against a real cat photograph,
the same fixture the browser app's tests use.
"""
from pathlib import Path
import pytest
from app.pipeline import analyse_bytes
from app.feelings import FEELINGS
from app.detect import NoCatFaceFound

FIXTURE = Path(__file__).resolve().parent.parent.parent / "test" / "fixtures" / "cat.jpg"


def test_fixture_exists():
    assert FIXTURE.exists(), f"expected fixture at {FIXTURE}"


def test_detects_a_real_cat_face_and_reads_a_feeling():
    result = analyse_bytes(FIXTURE.read_bytes())

    print(f"\n  Face box: {result.face.box}  (detector: {result.face.detector})")
    print(f"  Geometry: tilt={result.geometry.head_tilt_deg:.1f} "
          f"splay={result.geometry.ear_splay_ratio:.2f} "
          f"ears=({result.geometry.left_ear_angle_deg:.1f},{result.geometry.right_ear_angle_deg:.1f}) "
          f"muzzle={result.geometry.muzzle_ratio:.2f} "
          f"symmetry={result.geometry.nose_symmetry:.2f}")
    print(f"  Light: brightness={result.light.brightness:.2f} contrast={result.light.contrast:.2f} "
          f"sharpness={result.light.sharpness:.2f}")
    print(f"  Reading: {result.reading.feeling.emoji} {result.reading.feeling.label} "
          f"(confidence {result.reading.confidence}, reliable={result.reading.reliable})")
    print(f"  Runner-up: {result.reading.runner_up.label}")
    print(f"  Evidence: {result.reading.evidence}\n")

    assert result.reading.feeling.id in {f.id for f in FEELINGS}
    assert 0.3 <= result.reading.confidence <= 0.9
    assert len(result.reading.evidence) >= 2
    # A real photo of a cat should have a face box that's a real fraction of the frame.
    x, y, w, h = result.face.box
    assert w > 40 and h > 40


def test_reads_the_same_photo_the_same_way_twice():
    a = analyse_bytes(FIXTURE.read_bytes())
    b = analyse_bytes(FIXTURE.read_bytes())
    assert a.reading.feeling.id == b.reading.feeling.id
    assert a.reading.confidence == b.reading.confidence


def test_a_photo_with_no_cat_raises_no_cat_face_found():
    import numpy as np
    import cv2
    blank = np.full((200, 200, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok
    with pytest.raises(NoCatFaceFound):
        analyse_bytes(buf.tobytes())
