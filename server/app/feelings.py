"""
Maps what can actually be measured on a cat's face onto a feeling.

Everything here is about the cat - pupil dilation, how far apart the ears
sit, how tucked the muzzle is, how level the head is. Nothing about the
room: no brightness, no contrast, no photo sharpness. Those used to be in
the formulas and they produced confident nonsense (an alert, wide-eyed
cat called "Unimpressed" at 74% because the contrast happened to land on
a sweet spot). They are gone.

Removing them had an honest consequence. Five of the old ten feelings -
Sun-drunk, Sleepy, Fully loafed, Demanding, Plotting - were being told
apart *only* by light and sharpness. With those gone, nothing measurable
separates them, so they are gone too. Five remain, each pinned to real
measurements:

    Curious     wide pupils, ears spread, muzzle relaxed
    Locked on   wide pupils, ears spread, muzzle tight, head level
    Startled    wide pupils, ears pulled in, muzzle tight
    Wary        ears pulled in, muzzle tight, head tilted or low
    Unimpressed narrow pupils, ears spread, muzzle relaxed, head level

Ear *base angle* (the old "ears swept back" signal) is not used anywhere.
Across 10 real photos with correctly detected faces it varied with camera
angle and landmark placement, not with the cat, and it was the source of
a visibly false "ears swept back" line on a cat whose ears were straight
up. Ear *spread* (outer-ear span over interocular distance) does track
the real cue: a frightened cat with ears pinned read 1.39 against
1.8-2.4 for everything else.

Still a considered guess, not a diagnosis. And still blind to two things
that matter: eye aperture (a half-closed, sleepy eye is unmeasurable
without eyelid points) and mouth (open-mouth detection was tried and
fooled by pink fur). Body posture would need a pose model this service
does not have.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional

from .eyes import EyeSignals
from .geometry import FaceGeometry


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _blend(*terms: tuple) -> float:
    total = 0.0
    weight = 0.0
    for w, v in terms:
        if v is None:  # a signal that could not be measured on this photo
            continue
        total += w * _clamp01(v)
        weight += w
    return 0.0 if weight == 0 else total / weight


# --- derived cues, each 0..1, each named for the real thing it stands in for ---

def _arousal(e: EyeSignals) -> Optional[float]:
    """Pupil dilation stretched over the range seen on real photos:
    0.20 dark-fraction (calm, slit pupils) -> 0, 0.60 (wide) -> 1.
    None when the eyes could not be measured."""
    if not e.usable:
        return None
    return _clamp01((e.pupil_dilation - 0.20) / 0.40)


def _pinned_ears(g: FaceGeometry) -> float:
    """Ear spread pulled in toward the head. 2.0+ interocular widths -> 0
    (ears out, relaxed or alert); 1.3 -> 1 (pinned). A frightened cat
    measured 1.39; calm and alert cats 1.8-2.4."""
    return _clamp01((2.0 - g.ear_splay_ratio) / 0.7)


def _tense_muzzle(g: FaceGeometry) -> float:
    """Chin drawn up toward the eyes. 1.15+ -> 0 (relaxed, open face);
    0.70 -> 1 (tucked). A crouching, frightened cat measured 0.72."""
    return _clamp01((1.15 - g.muzzle_ratio) / 0.45)


def _tilted_head(g: FaceGeometry) -> float:
    """Head off level. Up to 5 degrees reads as level (0); 20 -> 1.
    The frightened cat measured 17 degrees; everything else under 8."""
    return _clamp01((abs(g.head_tilt_deg) - 5.0) / 15.0)


def _inv(x: Optional[float]) -> Optional[float]:
    return None if x is None else 1.0 - x


@dataclass(frozen=True)
class Feeling:
    id: str
    label: str
    emoji: str
    blurb: str
    cue: str
    score: Callable[[FaceGeometry, EyeSignals], float]


FEELINGS: List[Feeling] = [
    Feeling(
        id="curious", label="Curious", emoji="\U0001F440",
        blurb="Wide pupils, ears out, face loose. This cat wants to know what that was.",
        cue="Curiosity opens the pupils, pushes the ears out and forward, and leaves the mouth soft.",
        score=lambda g, e: _blend((4, _arousal(e)), (1.5, 1 - _pinned_ears(g)), (1.5, 1 - _tense_muzzle(g)), (1, _tilted_head(g))),
    ),
    Feeling(
        id="locked-on", label="Locked on", emoji="\U0001F3AF",
        blurb="Wide pupils, ears out, mouth set, head dead level. Something has this cat's full attention.",
        cue="A hunting cat fixes its head, opens its pupils, points its ears, and closes its mouth tight.",
        score=lambda g, e: _blend((4, _arousal(e)), (1.5, 1 - _pinned_ears(g)), (1.5, _tense_muzzle(g)), (1, 1 - _tilted_head(g))),
    ),
    Feeling(
        id="startled", label="Startled", emoji="\U0001F633",
        blurb="Pupils blown wide, ears pulled in, face tight. Something just happened.",
        cue="A startled cat dilates its pupils fully and pulls its ears in and back in one motion.",
        score=lambda g, e: _blend((4, _arousal(e)), (2.5, _pinned_ears(g)), (2, _tense_muzzle(g)), (1, _tilted_head(g))),
    ),
    Feeling(
        id="wary", label="Wary", emoji="\U0001FAE3",
        blurb="Ears pulled in, muzzle tight, head held low. This cat is keeping an exit in view.",
        cue="A wary cat draws its ears in, tightens its muzzle, and drops or tilts its head to watch.",
        score=lambda g, e: _blend((2.5, _pinned_ears(g)), (2, _tense_muzzle(g)), (2, _tilted_head(g))),
    ),
    Feeling(
        id="unimpressed", label="Unimpressed", emoji="\U0001F611",
        blurb="Narrow pupils, ears out, face loose, head level. This cat has considered you and moved on.",
        cue="A settled cat keeps its pupils narrow, its ears out, its mouth soft, and its head level.",
        score=lambda g, e: _blend((4, _inv(_arousal(e))), (1.5, 1 - _pinned_ears(g)), (1.5, 1 - _tense_muzzle(g)), (1, 1 - _tilted_head(g))),
    ),
]

_SYMMETRY_FLOOR = 0.55
"""Below this nose-symmetry score the face is likely a profile or a poor
crop. The reading is flagged rather than hidden - the app should still show
something, but say plainly that this one is shaky."""


@dataclass(frozen=True)
class Reading:
    feeling: Feeling
    runner_up: Feeling
    confidence: float
    evidence: List[str]
    reliable: bool


def _evidence(g: FaceGeometry, e: EyeSignals) -> List[str]:
    """Plain statements about the cat's face only. Every line is something
    that was measured on the cat, never on the room."""
    lines = []

    if e.usable:
        a = _arousal(e)
        if a >= 0.7:
            lines.append("The pupils are wide open.")
        elif a <= 0.25:
            lines.append("The pupils are narrow.")
        else:
            lines.append("The pupils are partway open.")
    else:
        lines.append("I could not read the pupils in this photo.")

    if _pinned_ears(g) > 0.5:
        lines.append("The ears are pulled in toward the head.")
    else:
        lines.append("The ears sit out wide.")

    if _tense_muzzle(g) > 0.5:
        lines.append("The muzzle reads drawn up and tight.")
    elif _tense_muzzle(g) < 0.15:
        lines.append("The muzzle reads loose and open.")

    if _tilted_head(g) > 0.5:
        lines.append("The head is tilted or held low, not level.")
    else:
        lines.append("The head is level.")

    return lines


def read_feeling(g: FaceGeometry, e: EyeSignals, face_reliable: bool = True) -> Reading:
    ranked = sorted(
        ((f, f.score(g, e)) for f in FEELINGS),
        key=lambda pair: (-pair[1], FEELINGS.index(pair[0])),
    )
    margin = ranked[0][1] - ranked[1][1]
    confidence = round(_clamp(0.3 + 2.5 * margin, 0.3, 0.9), 2)

    return Reading(
        feeling=ranked[0][0],
        runner_up=ranked[1][0],
        confidence=confidence,
        evidence=_evidence(g, e),
        reliable=face_reliable and g.nose_symmetry >= _SYMMETRY_FLOOR,
    )
