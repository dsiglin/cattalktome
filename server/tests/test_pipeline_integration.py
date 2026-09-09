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

# A defensive Sphynx cat, ears back, hissing at a dog from her owner's
# shoulder - CC BY-SA, Trilobite2, commons.wikimedia.org/wiki/File:Angry_Sphynx.jpg.
# The default Haar cascade settings (scaleFactor=1.05, minNeighbors=3)
# missed this entirely - a real user-reported failure. Regression fixture
# for the looser settings in detect.py.
HARD_POSE_FIXTURE = Path(__file__).resolve().parent.parent.parent / "test" / "fixtures" / "angry-sphynx.jpg"


def test_fixture_exists():
    assert FIXTURE.exists(), f"expected fixture at {FIXTURE}"
    assert HARD_POSE_FIXTURE.exists(), f"expected fixture at {HARD_POSE_FIXTURE}"


def test_detects_a_defensive_off_angle_cat_the_default_cascade_settings_missed():
    """
    Regression guard for a real user report: two photos in a row failed
    with 'no cat face found.' Investigation found the Haar cascade's
    default-ish settings (scaleFactor=1.05, minNeighbors=3) detected only
    6 of 9 real test photos - missing profile angles, mid-hiss open
    mouths, and this photo specifically. A finer scale step and lower
    neighbor count (1.02 / 2) recovered 8 of 9, with zero new false
    positives measured on blank and random-noise images, at a real but
    small cost (~70ms slower per request).
    """
    result = analyse_bytes(HARD_POSE_FIXTURE.read_bytes())
    x, y, w, h = result.face.box
    assert w > 40 and h > 40


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


def test_sharpness_does_not_cluster_near_zero_on_a_real_photo():
    """
    Regression guard for a real bug: sharpness was normalized against a
    synthetic checkerboard's mean |Laplacian| (~4.0), which no real
    photograph ever approaches. Measured across several real cat photos,
    raw (pre-normalization) values landed between 0.02 and 0.12 - so
    every real photo scored sharpness under 0.1 regardless of actual
    content. That silently starved every feeling needing real sharpness
    variation (Locked on, Curious can never win) and let Sleepy / Fully
    loafed win by default through their (1 - sharpness) terms, no matter
    what the photo showed.

    A synthetic checkerboard can't stand in for this test - it has far
    more edge energy than any real photo even at a coarse tile size, so
    it saturates the sharpness scale regardless of which calibration is
    in use. Only a real photograph (and a genuinely blurred copy of it)
    actually exercises the bug.
    """
    import cv2
    from app.photometrics import face_light

    bgr = cv2.imread(str(FIXTURE))
    result = analyse_bytes(FIXTURE.read_bytes())
    x, y, w, h = result.face.box
    crop = bgr[y:y + h, x:x + w]

    sharp = face_light(crop)
    assert sharp.sharpness > 0.3, (
        f"sharpness={sharp.sharpness} - a real, in-focus photo should land "
        "with real headroom above zero, not clustered near it the way every "
        "real photo did under the old checkerboard-normalized calibration"
    )

    blurred = cv2.GaussianBlur(crop, (15, 15), 0)
    soft = face_light(blurred)
    assert sharp.sharpness - soft.sharpness > 0.15, (
        "a clearly blurred version of the same photo should read "
        "meaningfully softer - the old calibration crushed both into "
        "~0.01-0.09, a gap too small to tell apart in practice"
    )
