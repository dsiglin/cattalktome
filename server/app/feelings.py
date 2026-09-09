"""
Maps real cat-face geometry plus face-crop photometrics onto a feeling.

This is a considered guess, not a diagnosis. It reuses the sticker
catalogue from the browser app (feelings.ts) so labels match across the
instant, on-device read and this deeper, server-side read - but the
inputs here are real facial geometry (ear-base angle, head tilt, muzzle
ratio) instead of whole-photo brightness/contrast, which is a genuine
improvement: the signal now comes from the cat's face, not the room.

Still honest about its limits: an 8-point landmark scheme has no whisker
point, so whisker change (one of the Feline Grimace Scale's five action
units) can never be measured this way. See geometry.py's docstrings for
the exact FGS mapping and its caveats.
"""
from dataclasses import dataclass
from typing import Callable, List

from .geometry import FaceGeometry
from .photometrics import FaceLight


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _near(value: float, target: float, tolerance: float) -> float:
    return pow(2.718281828, -(((value - target) / tolerance) ** 2))


def _blend(*terms: tuple) -> float:
    total = 0.0
    weight = 0.0
    for w, v in terms:
        total += w * _clamp01(v)
        weight += w
    return 0.0 if weight == 0 else total / weight


@dataclass(frozen=True)
class Feeling:
    id: str
    label: str
    emoji: str
    blurb: str
    cue: str
    score: Callable[[FaceGeometry, FaceLight], float]


_RELAXED_EAR_ANGLE = 42.0
"""The base-angle a relaxed, upright ear reads as in this landmark scheme -
not zero. The two ear-base points sit at different heights on a normally
shaped ear's outline (outer-lower corner, inner-upper corner), so even a
fully relaxed ear produces a diagonal ~40-45 degree line; that's the
landmark convention, not the cat's mood. Measured across four unrelated
real photos of calm cats (42.3-44.6 degrees on the more reliable side).
Recentering here - instead of on 0 degrees - is what lets this feature
actually separate "relaxed" from "flattened," rather than reading every
calm cat as already most of the way to "ears back.\""""

_EAR_ANGLE_SPREAD = 15.0
"""How many degrees past the relaxed baseline reads as fully flat. Set
from one real contrasting data point - a cat crouching from a dog, which
measured 50-63 degrees - so treat the exact value as a first calibration,
not a precise measurement."""


def _flat_ears(g: FaceGeometry) -> float:
    """0..1: how far the ears' base angle has swept back from the relaxed
    baseline. Larger deviation -> flatter/back-swept -> higher score."""
    avg = (abs(g.left_ear_angle_deg) + abs(g.right_ear_angle_deg)) / 2
    return _clamp01((avg - _RELAXED_EAR_ANGLE) / _EAR_ANGLE_SPREAD)


def _forward_ears(g: FaceGeometry) -> float:
    return 1.0 - _flat_ears(g)


def _tight_muzzle(g: FaceGeometry) -> float:
    """0..1: how drawn-back the muzzle reads. Lower ratio -> tighter."""
    return _clamp01((1.15 - g.muzzle_ratio) / 0.6)


def _relaxed_muzzle(g: FaceGeometry) -> float:
    return 1.0 - _tight_muzzle(g)


def _low_head(g: FaceGeometry) -> float:
    return _clamp01(abs(g.head_tilt_deg) / 25.0)


def _wide_splay(g: FaceGeometry) -> float:
    return _clamp01((g.ear_splay_ratio - 1.2) / 1.4)


FEELINGS: List[Feeling] = [
    Feeling(
        id="sun-drunk", label="Sun-drunk", emoji="\U0001F31E",
        blurb="Warm, soft light, ears forward, muzzle loose. This is a cat melting into a sunbeam.",
        cue="Relaxed cats in warm light hold their ears neutral, muzzle loose, and pupils narrow.",
        score=lambda g, p: _blend((3, p.brightness), (2, 1 - p.contrast), (1, 1 - p.sharpness), (2, _forward_ears(g))),
    ),
    Feeling(
        id="sleepy", label="Sleepy", emoji="\U0001F634",
        blurb="Low light, soft edges, ears at ease. This is the slow-blink end of the day.",
        cue="A drowsy cat softens its edges, half-closes its eyes, and stops tracking the room.",
        score=lambda g, p: _blend((1.5, 1 - p.sharpness), (2, 1 - p.contrast), (2, _near(p.brightness, 0.32, 0.35)), (1.5, _forward_ears(g))),
    ),
    Feeling(
        id="locked-on", label="Locked on", emoji="\U0001F3AF",
        blurb="Ears forward and level, head steady, sharp focus. Something has this cat's full attention.",
        cue="A hunting cat fixes its head, points its ears and whiskers forward, and stops moving.",
        score=lambda g, p: _blend((1.5, p.sharpness), (2.5, _forward_ears(g)), (2, 1 - _low_head(g)), (1.5, p.contrast)),
    ),
    Feeling(
        id="curious", label="Curious", emoji="\U0001F440",
        blurb="Wide-set ears, bright and crisp. This cat wants to know what that was.",
        cue="Curiosity pushes the ears and whiskers forward and opens the eyes wide.",
        score=lambda g, p: _blend((1.5, p.sharpness), (2, p.brightness), (2, _wide_splay(g)), (2, _forward_ears(g))),
    ),
    Feeling(
        id="startled", label="Startled", emoji="\U0001F633",
        blurb="Hard light, ears swept back suddenly. Something just happened.",
        cue="A startled cat dilates its pupils fully and flattens its ears in one motion.",
        score=lambda g, p: _blend((2.5, p.dark_ratio), (2.5, p.contrast), (2.5, _flat_ears(g)), (0.75, p.sharpness)),
    ),
    Feeling(
        id="wary", label="Wary", emoji="\U0001FAE3",
        blurb="Dim, ears back, muzzle tight, head held low. This cat is keeping an exit in view.",
        cue="A wary cat holds still in shade, turns its ears back, tightens its muzzle, and watches the room.",
        score=lambda g, p: _blend((3, 1 - p.brightness), (2.5, _flat_ears(g)), (2, _tight_muzzle(g)), (1.5, _low_head(g))),
    ),
    Feeling(
        id="demanding", label="Demanding", emoji="\U0001F37D️",
        blurb="Head level, ears forward, filling the frame. This is a request, not a pose.",
        cue="A cat asking for something walks straight at you, ears forward, and holds eye contact.",
        score=lambda g, p: _blend((3, g.nose_symmetry), (2.5, _forward_ears(g)), (2, _near(p.brightness, 0.6, 0.35)), (0.5, p.sharpness)),
    ),
    Feeling(
        id="content-loaf", label="Fully loafed", emoji="\U0001F35E",
        blurb="Soft light, ears relaxed, muzzle loose. Paws tucked, nothing owed to anyone.",
        cue="A cat that tucks its paws under itself feels safe enough to stop being ready.",
        score=lambda g, p: _blend((3, 1 - p.contrast), (1, 1 - p.sharpness), (2.5, _relaxed_muzzle(g)), (1.5, _forward_ears(g))),
    ),
    Feeling(
        id="unimpressed", label="Unimpressed", emoji="\U0001F611",
        blurb="Head level, ears upright, face still. This cat has considered you and moved on.",
        cue="A neutral cat holds its ears upright and its face still. That stillness is the message.",
        score=lambda g, p: _blend((3, 1 - _low_head(g)), (2, _near(p.contrast, 0.5, 0.35)), (1, _near(p.sharpness, 0.45, 0.35)), (1.5, g.nose_symmetry)),
    ),
    Feeling(
        id="mischief", label="Plotting something", emoji="\U0001F63C",
        blurb="Crisp, ears wide and forward, muzzle tight with focus. The decision is already made.",
        cue="Before a pounce a cat lowers its body, widens its ear stance, and locks its gaze.",
        score=lambda g, p: _blend((1.25, p.sharpness), (2, _wide_splay(g)), (2, _tight_muzzle(g)), (2, _forward_ears(g))),
    ),
]

_RELIABILITY_FLOOR = 0.55
"""Below this nose-symmetry score, the detected face is likely a profile
or a poor crop, and the reading is flagged as unreliable rather than hidden -
the app should still show something, but say plainly that this one is shaky."""


@dataclass(frozen=True)
class Reading:
    feeling: Feeling
    runner_up: Feeling
    confidence: float
    evidence: List[str]
    reliable: bool


def _evidence(g: FaceGeometry, p: FaceLight) -> List[str]:
    lines = []
    if p.brightness > 0.65:
        lines.append("The light on the face is bright.")
    elif p.brightness < 0.35:
        lines.append("The light on the face is low.")
    else:
        lines.append("The light on the face is even.")

    if abs(g.left_ear_angle_deg) > 30 or abs(g.right_ear_angle_deg) > 30:
        lines.append("The ears read swept back, not upright.")
    else:
        lines.append("The ears read upright and forward.")

    if g.muzzle_ratio < 0.85:
        lines.append("The muzzle reads drawn back and tight.")
    elif g.muzzle_ratio > 1.15:
        lines.append("The muzzle reads open and relaxed.")

    if abs(g.head_tilt_deg) > 15:
        lines.append("The head is tilted rather than level.")

    if p.sharpness > 0.5:
        lines.append("The face is in crisp focus, so the cat held still.")
    elif p.sharpness < 0.2:
        lines.append("The face is soft-edged, so the cat was relaxed or moving.")

    return lines


def read_feeling(g: FaceGeometry, p: FaceLight) -> Reading:
    ranked = sorted(
        ((f, f.score(g, p)) for f in FEELINGS),
        key=lambda pair: (-pair[1], FEELINGS.index(pair[0])),
    )
    margin = ranked[0][1] - ranked[1][1]
    confidence = round(_clamp(0.3 + 2.5 * margin, 0.3, 0.9), 2)

    return Reading(
        feeling=ranked[0][0],
        runner_up=ranked[1][0],
        confidence=confidence,
        evidence=_evidence(g, p),
        reliable=g.nose_symmetry >= _RELIABILITY_FLOOR,
    )
