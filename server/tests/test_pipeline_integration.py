"""
Integration tests: the whole pipeline against real cat photographs.

Three fixtures, each earning its place with a real failure:
  cat.jpg                    - the original fixture; an ordinary lounging cat.
  angry-sphynx.jpg           - a defensive Sphynx in profile. Strict detection
                               misses it; only the loose stage finds it. A
                               user-reported "no cat face found."
  alert-tabby-windowsill.png - a user's own cat, wide-eyed, head pitched up
                               ~10 degrees. Strict detection missed the face;
                               a single loosened pass found only the cat's
                               EYE and fitted a whole face inside it, and the
                               light-based scorer called the result
                               "Unimpressed" at 74%. Regression fixture for
                               the staged detector and the eyes-not-light
                               scorer both.
"""
from pathlib import Path
import pytest
from app.pipeline import analyse_bytes
from app.feelings import FEELINGS
from app.detect import NoCatFaceFound

FIXTURES = Path(__file__).resolve().parent.parent.parent / "test" / "fixtures"
LOUNGING = FIXTURES / "cat.jpg"
SPHYNX = FIXTURES / "angry-sphynx.jpg"
ALERT_TABBY = FIXTURES / "alert-tabby-windowsill.png"


def test_fixtures_exist():
    for f in (LOUNGING, SPHYNX, ALERT_TABBY):
        assert f.exists(), f"expected fixture at {f}"


def test_lounging_cat_resolves_at_the_strict_stage():
    r = analyse_bytes(LOUNGING.read_bytes())
    assert r.face.detector == "haar"
    assert r.face.reliable is True
    assert r.reading.feeling.id in {f.id for f in FEELINGS}
    # Narrow pupils (0.29), ears wide (2.51), muzzle loose (1.35), head
    # level, nose symmetry 0.95: a relaxed cat looking at you.
    assert r.reading.feeling.id == "trusting"
    assert 0.3 <= r.reading.confidence <= 0.9
    assert len(r.reading.evidence) >= 2


def test_reads_the_same_photo_the_same_way_twice():
    a = analyse_bytes(LOUNGING.read_bytes())
    b = analyse_bytes(LOUNGING.read_bytes())
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


def test_defensive_sphynx_is_found_by_the_loose_stage_and_flagged():
    r = analyse_bytes(SPHYNX.read_bytes())
    assert r.face.detector == "haar-loose"
    assert r.face.reliable is False
    assert r.reading.reliable is False, "a loose-stage box must make the whole reading hedge"


def test_alert_tabby_face_is_found_by_rotation_not_an_eye():
    r = analyse_bytes(ALERT_TABBY.read_bytes())
    x, y, w, h = r.face.box
    # The real face spans roughly x 300-690, y 40-480 in this 740x970 crop;
    # the false positive that used to win was a 90px box on the right eye.
    assert r.face.detector == "haar-rotated"
    assert r.face.reliable is True
    assert w > 250 and h > 250, f"box {r.face.box} is too small to be the face - is it an eye again?"
    cx, cy = x + w / 2, y + h / 2
    assert 350 < cx < 650 and 100 < cy < 450, f"box centre ({cx:.0f},{cy:.0f}) is not on the face"


def test_alert_tabby_reads_as_aroused_not_unimpressed():
    r = analyse_bytes(ALERT_TABBY.read_bytes())
    assert r.eyes.usable, "the pupils in this photo are huge and clearly visible"
    assert r.eyes.pupil_dilation > 0.5
    assert r.reading.feeling.id in {"curious", "focused"}, (
        f"read as {r.reading.feeling.id!r} - the old light-based model said Unimpressed"
    )
    joined = " ".join(r.reading.evidence).lower()
    assert "swept back" not in joined
    assert "light" not in joined and "focus" not in joined


def test_evidence_never_mentions_the_room():
    for fixture in (LOUNGING, SPHYNX, ALERT_TABBY):
        r = analyse_bytes(fixture.read_bytes())
        joined = " ".join(r.reading.evidence).lower()
        for banned in ("light", "bright", "dim", "focus", "sharp", "blur"):
            assert banned not in joined, f"{fixture.name}: evidence mentions the room: {banned!r}"
