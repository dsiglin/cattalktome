"""
Tests for the feeling catalogue and scorer.

Inputs are face geometry (from real landmarks) and eye signals (pupil
dilation measured on the cat's own eyes). Nothing about the room's light
is an input any more - see feelings.py for why.

The scenario tests below use measurements taken from real photos with
correctly detected faces, not invented values.
"""
import pytest
from app.geometry import FaceGeometry
from app.eyes import EyeSignals
from app.feelings import FEELINGS, read_feeling


def geo(**over):
    base = dict(
        interocular_dist=100.0, head_tilt_deg=0.0, ear_splay_ratio=2.2,
        left_ear_angle_deg=-42.0, right_ear_angle_deg=42.0,
        muzzle_ratio=1.3, nose_symmetry=0.97,
    )
    base.update(over)
    return FaceGeometry(**base)


def eyes(dilation=0.25, usable=True):
    return EyeSignals(pupil_dilation=dilation, usable=usable, left_raw=dilation, right_raw=dilation)


class TestCatalogue:
    def test_has_the_seven_feelings_the_face_can_support(self):
        assert {f.id for f in FEELINGS} == {
            "curious", "focused", "frightened", "cautious", "irritated", "trusting", "unimpressed",
        }

    def test_ids_are_unique(self):
        ids = [f.id for f in FEELINGS]
        assert len(set(ids)) == len(ids)

    def test_every_feeling_has_label_emoji_blurb_cue(self):
        for f in FEELINGS:
            assert len(f.label) > 0
            assert len(f.label) <= 24
            assert len(f.emoji) > 0
            assert len(f.blurb) > 10
            assert len(f.cue) > 10


class TestReadFeeling:
    def test_deterministic(self):
        a = read_feeling(geo(), eyes())
        b = read_feeling(geo(), eyes())
        assert a.feeling.id == b.feeling.id
        assert a.confidence == b.confidence

    def test_returns_a_known_feeling(self):
        assert read_feeling(geo(), eyes()).feeling.id in {f.id for f in FEELINGS}

    def test_runner_up_differs_from_winner(self):
        r = read_feeling(geo(), eyes())
        assert r.runner_up.id != r.feeling.id

    def test_confidence_is_bounded(self):
        r = read_feeling(geo(), eyes())
        assert 0.3 <= r.confidence <= 0.9

    def test_gives_evidence_about_the_face_only(self):
        r = read_feeling(geo(), eyes())
        assert len(r.evidence) >= 2
        joined = " ".join(r.evidence).lower()
        for banned in ("light", "bright", "dim", "focus", "sharp", "blur", "swept back"):
            assert banned not in joined, f"evidence mentions the room, not the cat: {banned!r}"

    def test_flags_low_reliability_on_poor_symmetry(self):
        assert read_feeling(geo(nose_symmetry=0.95), eyes()).reliable is True
        assert read_feeling(geo(nose_symmetry=0.3), eyes()).reliable is False

    def test_flags_low_reliability_when_the_detector_was_unsure(self):
        assert read_feeling(geo(), eyes(), face_reliable=False).reliable is False

    def test_an_unreliable_reading_never_looks_sure(self):
        # The same relaxed face that scores 0.90 when the detector was sure
        # must cap at a coin flip when the box came from the loose stage -
        # that stage has been seen to pick a patch of chest fur.
        sure = read_feeling(geo(ear_splay_ratio=2.5, muzzle_ratio=1.35), eyes(dilation=0.29))
        unsure = read_feeling(geo(ear_splay_ratio=2.5, muzzle_ratio=1.35), eyes(dilation=0.29), face_reliable=False)
        assert sure.confidence > 0.5
        assert unsure.confidence <= 0.5
        assert unsure.feeling.id == sure.feeling.id

    # --- scenarios grounded in real measurements -------------------------

    def test_alert_wide_eyed_cat_reads_curious_or_focused(self):
        # A grey tabby on a windowsill looking straight up at the camera:
        # pupils visibly huge, ears out, face loose, head level. The old
        # light-based model called this one "Unimpressed" at 74%.
        r = read_feeling(
            geo(head_tilt_deg=0.0, ear_splay_ratio=2.23, muzzle_ratio=1.30, nose_symmetry=0.97),
            eyes(dilation=0.64),
        )
        assert r.feeling.id in {"curious", "focused"}

    def test_calm_narrow_pupil_cat_facing_you_reads_trusting(self):
        # Four calm portrait cats measured pupil-dark 0.15-0.38, spread
        # 1.8-2.4, muzzle 1.24-1.49, tilt under 8 degrees, nose symmetry
        # 0.95-0.99. Narrow pupils on a relaxed face pointed at you = trust.
        r = read_feeling(
            geo(head_tilt_deg=-4.2, ear_splay_ratio=2.27, muzzle_ratio=1.28, nose_symmetry=0.99),
            eyes(dilation=0.15),
        )
        assert r.feeling.id == "trusting"
        assert any("points at the camera" in line for line in r.evidence)

    def test_calm_cat_facing_the_lens_at_0_90_symmetry_is_still_trusting(self):
        # A Savannah cat looking straight at the camera measured nose
        # symmetry 0.90 with narrow pupils (0.17) and ears up. A boundary
        # at 0.945 called it Unimpressed; it is plainly facing you.
        r = read_feeling(
            geo(head_tilt_deg=0.8, ear_splay_ratio=1.86, muzzle_ratio=1.24, nose_symmetry=0.90),
            eyes(dilation=0.17),
        )
        assert r.feeling.id == "trusting"

    def test_calm_cat_looking_away_reads_unimpressed(self):
        # Same relaxed face, but the nose has drifted toward one eye - the
        # head is turned part-way from the camera (real turned heads scored
        # 0.87-0.89). Unimpressed is about looking away, not about pupils.
        r = read_feeling(
            geo(head_tilt_deg=-2.0, ear_splay_ratio=2.2, muzzle_ratio=1.28, nose_symmetry=0.86),
            eyes(dilation=0.15),
        )
        assert r.feeling.id == "unimpressed"
        assert any("turned away" in line for line in r.evidence)

    def test_narrow_pupils_on_a_tense_face_never_read_as_trusting(self):
        # A scared tabby on a real photo: pupils measured narrow (0.00),
        # ears pulled in (1.63), muzzle tight (0.85), head near level.
        # Narrow pupils alone must not make this cat "Trusting".
        r = read_feeling(
            geo(head_tilt_deg=7.4, ear_splay_ratio=1.63, muzzle_ratio=0.85, nose_symmetry=0.98),
            eyes(dilation=0.00),
        )
        assert r.feeling.id in {"irritated", "cautious"}
        assert r.runner_up.id != "trusting" or r.confidence > 0.5

    def test_hissing_cat_reads_irritated(self):
        # A hissing cat on a real photo: narrow pupils (0.04), ears back
        # (1.50), muzzle tight (0.81), head tilted 11.7, facing the camera.
        # The chart's "Angry": ears rotated back, pupils to slits.
        r = read_feeling(
            geo(head_tilt_deg=11.7, ear_splay_ratio=1.50, muzzle_ratio=0.81, nose_symmetry=0.91),
            eyes(dilation=0.04),
        )
        assert r.feeling.id == "irritated"

    def test_frightened_crouching_cat_reads_cautious_or_frightened(self):
        # A cat crouching from a dog: ears pulled in (spread 1.39), chin
        # tucked (muzzle 0.72), head down (tilt 17 degrees). Its eyes were
        # glowing so the pupils could not be read - the reading must still
        # land in the fear family on geometry alone.
        r = read_feeling(
            geo(head_tilt_deg=17.3, ear_splay_ratio=1.39, muzzle_ratio=0.72, nose_symmetry=0.90),
            eyes(dilation=0.5, usable=False),
        )
        assert r.feeling.id in {"cautious", "frightened"}

    def test_unreadable_pupils_still_produce_a_reading(self):
        r = read_feeling(geo(), eyes(usable=False))
        assert r.feeling.id in {f.id for f in FEELINGS}
        assert any("could not read the pupils" in line for line in r.evidence)

    def test_every_feeling_is_reachable(self):
        seen = set()
        for tilt in [0, 8, 18]:
            for splay in [1.3, 1.8, 2.3]:
                for muzzle in [0.7, 1.0, 1.4]:
                    for sym in [0.5, 0.86, 0.97]:
                        for dilation, usable in [(0.1, True), (0.35, True), (0.65, True), (0.5, False)]:
                            r = read_feeling(
                                geo(head_tilt_deg=tilt, ear_splay_ratio=splay, muzzle_ratio=muzzle, nose_symmetry=sym),
                                eyes(dilation=dilation, usable=usable),
                            )
                            seen.add(r.feeling.id)
        missing = [f.id for f in FEELINGS if f.id not in seen]
        assert missing == [], f"unreachable: {missing}"
