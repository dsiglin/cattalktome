"""
Maps what can actually be measured on a cat's face onto a feeling.

Everything here is about the cat - pupil dilation, how far apart the ears
sit, how tucked the muzzle is, how level the head is, and which way the
face points. Nothing about the room: no brightness, no contrast, no photo
sharpness. Those used to be in the formulas and they produced confident
nonsense (an alert, wide-eyed cat called "Unimpressed" at 74% because the
contrast happened to land on a sweet spot). They are gone.

The catalogue follows the standard cat facial-expression chart (Happy,
Angry, Frightened, Playful, Content) and the body-language poster
(Interested, Friendly, Cautious, Trusting, Irritated, Focus, ...), kept
to the entries the five measurable cues can actually tell apart:

                 pupils   ears      muzzle   head      face
    Curious      wide     out       loose    tilted    -
    Focused      wide     out       tight    level     -
    Frightened   wide     pinned    tight    tilted    -
    Cautious     (any)    pinned    tight    tilted    -
    Irritated    narrow   pinned    tight    -         toward you
    Trusting     narrow   out       loose    level     toward you
    Unimpressed  narrow   out       -        -         turned away

Two things the chart has that this cannot offer, said plainly:
"Content" needs eye aperture (half-closed lids) and the 8-point scheme
has no eyelid points; everything on the body poster (tail, crouch,
rolling over) needs a pose model this service does not have.

Narrow pupils are NOT enough on their own to call a cat Trusting. On a
real photo a scared tabby measured narrow pupils with pinned ears and a
tight muzzle - that is Irritated or Cautious, never Trusting. So Trusting
is gated: pinned ears or a tight muzzle scale its score down hard.

Ear *base angle* (the old "ears swept back" signal) is not used anywhere.
Across 10 real photos with correctly detected faces it varied with camera
angle and landmark placement, not with the cat. Ear *spread* does track
the real cue: a frightened cat with ears pinned read 1.39 against 1.8-2.4
for everything else.

Still a considered guess, not a diagnosis.
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

def _arousal(e: EyeSignals) -> float:
    """Pupil dilation stretched over the range seen on real photos:
    0.20 dark-fraction (calm, slit pupils) -> 0, 0.60 (wide) -> 1.

    0.5 (neutral) when the eyes could not be measured. Neutral, not
    dropped: if the term simply vanished, Irritated would lose its
    "narrow pupils" requirement and become "ears in, muzzle tight" -
    the same thing as Cautious - and Focused would win on a tight muzzle
    alone. An unreadable pupil must push no feeling either way."""
    if not e.usable:
        return 0.5
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


def _turned_away(g: FaceGeometry) -> float:
    """Face turned from the camera: the nose drifts toward one eye. On real
    photos (offset measured along the eye line) cats facing the lens scored
    nose symmetry 0.90-0.99; heads turned part-way scored 0.87-0.89. The
    separation is thin - a frontal detector only finds faces that mostly
    face the lens - so this is a weak cue. 0.91+ -> 0 (facing); 0.85 -> 1."""
    return _clamp01((0.91 - g.nose_symmetry) / 0.06)


def _relaxed_gate(g: FaceGeometry) -> float:
    """1 when both ears and muzzle are relaxed, falling to 0 as either
    tightens. Multiplies the Trusting score so narrow pupils on a tense
    face can never read as trust."""
    return 1.0 - max(_pinned_ears(g), _tense_muzzle(g))


def _inv(x: float) -> float:
    return 1.0 - x


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
        cue="Interest opens the pupils, pushes the ears out and forward, and leaves the mouth soft.",
        score=lambda g, e: _blend((4, _arousal(e)), (1.5, 1 - _pinned_ears(g)), (1.5, 1 - _tense_muzzle(g)), (1, _tilted_head(g))),
    ),
    Feeling(
        id="focused", label="Focused", emoji="\U0001F3AF",
        blurb="Wide pupils, ears out, mouth set, head dead level. Something has this cat's full attention.",
        cue="A hunting cat fixes its head, opens its pupils, points its ears, and closes its mouth tight.",
        score=lambda g, e: _blend((4, _arousal(e)), (1.5, 1 - _pinned_ears(g)), (1.5, _tense_muzzle(g)), (1, 1 - _tilted_head(g))),
    ),
    Feeling(
        id="frightened", label="Frightened", emoji="\U0001F633",
        blurb="Pupils blown wide, ears pulled flat, face tight. Something just scared this cat.",
        cue="A frightened cat dilates its pupils fully and flattens its ears out sideways in one motion.",
        score=lambda g, e: _blend((4, _arousal(e)), (2.5, _pinned_ears(g)), (2, _tense_muzzle(g)), (1, _tilted_head(g))),
    ),
    Feeling(
        id="cautious", label="Cautious", emoji="\U0001FAE3",
        blurb="Ears pulled in, muzzle tight, head held low. This cat is keeping an exit in view.",
        cue="A cautious cat draws its ears in, tightens its muzzle, and drops or tilts its head to watch.",
        score=lambda g, e: _blend((2.5, _pinned_ears(g)), (2, _tense_muzzle(g)), (2, _tilted_head(g))),
    ),
    Feeling(
        id="irritated", label="Irritated", emoji="\U0001F63E",
        blurb="Narrow pupils, ears swung back, muzzle tight, eyes on you. This cat would like you to stop.",
        cue="An angry cat narrows its pupils to slits, rotates its ears back, and tightens its face while it stares.",
        score=lambda g, e: _blend((2.5, _pinned_ears(g)), (2, _tense_muzzle(g)), (2, _inv(_arousal(e))), (1, 1 - _turned_away(g))),
    ),
    Feeling(
        id="trusting", label="Trusting", emoji="\U0001F60C",
        blurb="Soft, narrow pupils, ears easy, face loose and turned to you. This cat is comfortable with you.",
        cue="A cat that trusts you keeps its pupils narrow and soft, its ears out, its face loose, and looks right at you.",
        score=lambda g, e: _relaxed_gate(g) * _blend(
            (2, _inv(_arousal(e))), (2, 1 - _pinned_ears(g)), (2, 1 - _tense_muzzle(g)),
            (1, 1 - _tilted_head(g)), (1, 1 - _turned_away(g)),
        ),
    ),
    Feeling(
        id="unimpressed", label="Unimpressed", emoji="\U0001F611",
        blurb="Face turned away, pupils narrow, ears easy. This cat has considered you and moved on.",
        cue="A cat that is over it looks away, keeps its pupils narrow, and leaves its ears where they were.",
        score=lambda g, e: _blend((3, _turned_away(g)), (2, _inv(_arousal(e))), (1.5, 1 - _pinned_ears(g)), (1, 1 - _tense_muzzle(g))),
    ),
]

_UNRELIABLE_CONFIDENCE_CAP = 0.5
"""An unreliable reading never shows more than a coin flip's confidence."""

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

    if _turned_away(g) > 0.5:
        lines.append("The face is turned away from the camera.")
    else:
        lines.append("The face points at the camera.")

    return lines


def read_feeling(g: FaceGeometry, e: EyeSignals, face_reliable: bool = True) -> Reading:
    ranked = sorted(
        ((f, f.score(g, e)) for f in FEELINGS),
        key=lambda pair: (-pair[1], FEELINGS.index(pair[0])),
    )
    margin = ranked[0][1] - ranked[1][1]
    confidence = round(_clamp(0.3 + 2.5 * margin, 0.3, 0.9), 2)

    reliable = face_reliable and g.nose_symmetry >= _SYMMETRY_FLOOR
    if not reliable:
        # The box may not be a face at all (the loose detection stage has
        # been seen to pick a patch of chest fur), or it is a profile the
        # landmarks were never trained on. Whatever the geometry says, the
        # reading must not present itself as sure.
        confidence = min(confidence, _UNRELIABLE_CONFIDENCE_CAP)

    return Reading(
        feeling=ranked[0][0],
        runner_up=ranked[1][0],
        confidence=confidence,
        evidence=_evidence(g, e),
        reliable=reliable,
    )
