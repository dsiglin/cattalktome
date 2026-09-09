"""
Tests for the feeling catalogue and scorer.

Inputs are geometry (from real landmarks) plus photometrics (measured on
the cropped face only, not the whole photo - fixing the flaw where
background light/shadow drove the reading instead of the cat).
"""
import pytest
from app.geometry import FaceGeometry
from app.photometrics import FaceLight
from app.feelings import FEELINGS, read_feeling


def geo(**over):
    base = dict(
        interocular_dist=40.0, head_tilt_deg=0.0, ear_splay_ratio=2.0,
        left_ear_angle_deg=0.0, right_ear_angle_deg=0.0,
        muzzle_ratio=1.0, nose_symmetry=0.95,
    )
    base.update(over)
    return FaceGeometry(**base)


def light(**over):
    base = dict(brightness=0.5, contrast=0.5, sharpness=0.4, dark_ratio=0.2)
    base.update(over)
    return FaceLight(**base)


class TestCatalogue:
    def test_has_at_least_eight_feelings(self):
        assert len(FEELINGS) >= 8

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
        a = read_feeling(geo(), light())
        b = read_feeling(geo(), light())
        assert a.feeling.id == b.feeling.id
        assert a.confidence == b.confidence

    def test_returns_a_known_feeling(self):
        r = read_feeling(geo(), light())
        assert r.feeling.id in {f.id for f in FEELINGS}

    def test_runner_up_differs_from_winner(self):
        r = read_feeling(geo(), light())
        assert r.runner_up.id != r.feeling.id

    def test_confidence_is_bounded(self):
        r = read_feeling(geo(), light())
        assert 0.3 <= r.confidence <= 0.9

    def test_gives_evidence(self):
        r = read_feeling(geo(), light())
        assert len(r.evidence) >= 2

    def test_flags_low_reliability_on_poor_symmetry(self):
        good = read_feeling(geo(nose_symmetry=0.95), light())
        poor = read_feeling(geo(nose_symmetry=0.3), light())
        assert poor.reliable is False
        assert good.reliable is True

    def test_flat_ears_and_low_head_read_as_wary_or_startled(self):
        # ears swept back hard past the relaxed ~42-degree baseline, head
        # tilted low, tight muzzle. 63/-60 mirrors a real scared cat's
        # measured angles (a cat crouching from a dog, detected live).
        r = read_feeling(
            geo(head_tilt_deg=22, left_ear_angle_deg=63, right_ear_angle_deg=-60,
                muzzle_ratio=0.7, ear_splay_ratio=1.3),
            light(brightness=0.3, contrast=0.7, dark_ratio=0.5),
        )
        assert r.feeling.id in {"wary", "startled"}

    def test_forward_ears_and_relaxed_muzzle_read_as_calm(self):
        # 42/-42 is the relaxed baseline this landmark scheme actually
        # produces for an upright ear - not 0 - measured across four
        # real calm-cat photos (42.3-44.6 degrees).
        r = read_feeling(
            geo(left_ear_angle_deg=42, right_ear_angle_deg=-42, muzzle_ratio=1.3, ear_splay_ratio=2.3),
            light(brightness=0.7, contrast=0.15, sharpness=0.1),
        )
        assert r.feeling.id in {"sun-drunk", "content-loaf", "sleepy"}

    def test_every_feeling_is_reachable(self):
        seen = set()
        axis = [0.0, 0.25, 0.5, 0.75, 1.0]
        # Centered on the ~42-degree relaxed baseline (see _RELAXED_EAR_ANGLE
        # in feelings.py), spanning well past it in both directions so
        # genuinely flat-eared inputs are exercised too.
        angles = [20, 35, 42, 55, 70]
        for head_tilt in [-15, 0, 15]:
            for splay in [1.2, 1.8, 2.4]:
                for la in angles:
                    for ra in angles:
                        for muzzle in [0.6, 1.0, 1.4]:
                            for sym in [0.5, 0.95]:
                                for brightness in axis:
                                    for contrast in axis:
                                        for sharpness in [0.05, 0.4, 0.8]:
                                            for dark in [0.0, 0.3, 0.6]:
                                                r = read_feeling(
                                                    geo(head_tilt_deg=head_tilt, ear_splay_ratio=splay,
                                                        left_ear_angle_deg=la, right_ear_angle_deg=ra,
                                                        muzzle_ratio=muzzle, nose_symmetry=sym),
                                                    light(brightness=brightness, contrast=contrast,
                                                          sharpness=sharpness, dark_ratio=dark),
                                                )
                                                seen.add(r.feeling.id)
        missing = [f.id for f in FEELINGS if f.id not in seen]
        assert missing == [], f"unreachable: {missing}"
