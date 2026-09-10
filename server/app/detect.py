"""
Cat face detection and 8-point landmark extraction.

Detection is staged, most-trustworthy first. This shape came out of a real
failure: on an ordinary phone photo of an alert cat with its head pitched
up ~10 degrees, a single loosened Haar pass found *only* a 90px false
positive on the cat's eye - and the landmark predictor then fitted an
entire face inside that eyeball, so every downstream measurement was
garbage delivered with 74% confidence. Meanwhile the strict pass found
nothing, which at least was honest.

Measured across 11 real photos (see tests/test_pipeline_integration.py):

  Stage 1 - strict Haar, unrotated (scaleFactor 1.05, minNeighbors 3).
            Found the correct face on every ordinary photo (IoU >= 0.98).
            Its only misses are pitched/rotated/extreme heads. Largest box
            wins here - on this stage, that rule was never wrong.
  Stage 2 - both cat cascades on rotated copies (+/-20, +/-35 degrees),
            still strict. Recovers pitched heads. A box needs >= 2 agreeing
            detections (clustered by IoU), because a single rotated hit is
            often a warp artifact. Rotation uses black borders and rejects
            boxes touching the padded corners for the same reason.
  Stage 3 - loose Haar, unrotated (1.02 / 2). Recovers extreme poses
            (a hissing Sphynx in profile) but also produces false positives,
            so anything found here is flagged `reliable=False` and the app
            hedges instead of asserting.

pycatfd's own FHOG detector was dropped: it found nothing on any of the
11 photos, at any upsampling, so it was only adding latency.

Landmarks: pycatfd's dlib shape predictor (8 points), which fits *some*
face inside any box it is handed - which is exactly why box quality has
to be settled before the predictor ever runs.
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import dlib
import numpy as np

from . import catdet, facedet

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_CASCADE_EXT_PATH = MODELS_DIR / "haarcascade_frontalcatface_extended.xml"
_CASCADE_STD_PATH = MODELS_DIR / "haarcascade_frontalcatface.xml"
_PREDICTOR_DAT_PATH = MODELS_DIR / "predictor.dat"

_MIN_FACE_PX = 75
_ROTATIONS_DEG = (-20, 20, -35, 35)
_CLUSTER_IOU = 0.3
_MIN_VOTES = 2

_cascade_ext = None
_cascade_std = None
_predictor = None


def _load_cascade(path: Path) -> cv2.CascadeClassifier:
    cas = cv2.CascadeClassifier(str(path))
    if cas.empty():
        raise RuntimeError(f"failed to load cascade at {path}")
    return cas


def _ext():
    global _cascade_ext
    if _cascade_ext is None:
        _cascade_ext = _load_cascade(_CASCADE_EXT_PATH)
    return _cascade_ext


def _std():
    global _cascade_std
    if _cascade_std is None:
        _cascade_std = _load_cascade(_CASCADE_STD_PATH)
    return _cascade_std


def _shape_predictor():
    global _predictor
    if _predictor is None:
        _predictor = dlib.shape_predictor(str(_PREDICTOR_DAT_PATH))
    return _predictor


@dataclass(frozen=True)
class DetectedFace:
    box: tuple  # (x, y, w, h) in the original image
    landmarks: list  # 8 (x, y) points, pycatfd CatFaceLandmark order
    detector: str  # "yolox-face", "haar", "haar-rotated", "haar-contained", or "haar-loose" - which stage found it
    reliable: bool  # False when the box came from the loose stage


class NoCatFaceFound(Exception):
    """No cat face anywhere in the photo."""


class CatFoundButNoFace(NoCatFaceFound):
    """A whole-cat detector sees a cat, but no stage found a readable face -
    typically a profile, a cat facing away, or a face hidden behind
    something. Different message for the user; same handling otherwise."""


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


def _largest(boxes):
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return (int(x), int(y), int(w), int(h))


def _stage_strict(gray):
    boxes = _ext().detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(_MIN_FACE_PX, _MIN_FACE_PX))
    return _largest(boxes) if len(boxes) else None


def _rotate(gray, deg):
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    rotated = cv2.warpAffine(gray, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated, m


def _unrotate_box(box, m):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    ux, uy = cv2.invertAffineTransform(m) @ np.array([cx, cy, 1.0])
    return (int(ux - w / 2), int(uy - h / 2), int(w), int(h))


def _median_box(cluster):
    return tuple(int(v) for v in np.median(np.array(cluster), axis=0))


def _stage_rotated(gray):
    h, w = gray.shape
    candidates = []
    for cascade in (_ext(), _std()):
        for deg in _ROTATIONS_DEG:
            rotated, m = _rotate(gray, deg)
            for b in cascade.detectMultiScale(rotated, scaleFactor=1.05, minNeighbors=3, minSize=(_MIN_FACE_PX, _MIN_FACE_PX)):
                x, y, bw, bh = _unrotate_box(b, m)
                # A box poking outside the real image sits on the rotation's padded corners.
                if x < 0 or y < 0 or x + bw > w or y + bh > h:
                    continue
                candidates.append((x, y, bw, bh))

    # Cluster overlapping candidates; a real face is hit from several angles,
    # a warp artifact usually only once.
    clusters = []
    for box in candidates:
        for cluster in clusters:
            if _iou(box, _median_box(cluster)) >= _CLUSTER_IOU:
                cluster.append(box)
                break
        else:
            clusters.append([box])

    supported = [c for c in clusters if len(c) >= _MIN_VOTES]
    if not supported:
        return None
    best = max(supported, key=lambda c: (len(c), _median_box(c)[2] * _median_box(c)[3]))
    return _median_box(best)


def _stage_loose(gray):
    boxes = _ext().detectMultiScale(gray, scaleFactor=1.02, minNeighbors=2, minSize=(_MIN_FACE_PX, _MIN_FACE_PX))
    return _largest(boxes) if len(boxes) else None


_DETECT_MAX_SIDE = 1400
"""Detection runs on a copy no larger than this on its longest side.

Haar cascades are trained on 24x24 patches and get noisy on large, finely
textured photos. On a 1450x2576 phone photo of a wide-eyed tabby, every
stage missed the 550px face at full resolution and the loose stage
settled on a 168px patch of fur; downscaled to anywhere between 500 and
1600px, the rotated stage found the same face box every time. On a
2576px photo of a cat in profile, full resolution produced a *false*
face on the chest fur, and 1400px correctly found nothing.

1400 is also exactly the long side the browser app uploads (MAX_RENDER_SIDE
in src/main.ts), so for photos from the app the server measures the very
pixels the phone sent, with no second resampling in between.

The box is found on the downscaled copy and mapped back; the landmarks
and the pupil measurement still use the full-resolution pixels.

Known limit, measured and accepted: fluffy chest fur can produce a
Haar "face" that no cheap check separates from a real one (its level
weight, neighbour count, and eye-region contrast all overlap with real
faces found by the loose stage). Such a box only ever comes from the
loose stage, which is flagged unreliable, and the scorer caps the
confidence of any unreliable reading at 0.5."""


_CONTAINER_AREA_RATIO = 2.5
_CONTAINER_SLACK = 0.2  # the small box may poke out of the big one by this fraction of its size


def _contains(outer, inner) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    sx, sy = iw * _CONTAINER_SLACK, ih * _CONTAINER_SLACK
    return ox <= ix + sx and oy <= iy + sy and ox + ow >= ix + iw - sx and oy + oh >= iy + ih - sy


def _containing_face(gray, box):
    """A face *part* mistaken for a face: the strict stage has picked a
    cat's muzzle (a Savannah, 166px) and a cat's eye (a tabby, 90px) and
    reported them as whole faces with a straight face. The signature is
    specific: a much larger box, found by the extended cascade at loose
    settings and confirmed by the standard cascade, that *contains* the
    small one. Across 25 real photos this fires exactly once, on the
    Savannah, and lands on its actual face. A tail or a patch of fur next
    to a real face does not contain it, so this cannot swap a correct
    small face for a wrong big one."""
    ext = [tuple(int(v) for v in b) for b in _ext().detectMultiScale(gray, scaleFactor=1.02, minNeighbors=2, minSize=(_MIN_FACE_PX, _MIN_FACE_PX))]
    std = [tuple(int(v) for v in b) for b in _std().detectMultiScale(gray, scaleFactor=1.02, minNeighbors=2, minSize=(_MIN_FACE_PX, _MIN_FACE_PX))]
    area = box[2] * box[3]
    best = None
    for c in ext:
        if c[2] * c[3] < _CONTAINER_AREA_RATIO * area or not _contains(c, box):
            continue
        if max((_iou(c, s) for s in std), default=0.0) < _CLUSTER_IOU:
            continue
        if best is None or c[2] * c[3] > best[2] * best[3]:
            best = c
    return best


def detect_cat_face(bgr) -> DetectedFace:
    """Find a cat face and its 8 landmarks in a BGR image (as loaded by cv2.imread).

    Order of trust:
      0. learned YOLOX cat-face detector, if its model file is present
         (facedet.py) - trained on faces at every angle; skipped when absent.
      1-3. the Haar stages above, each candidate vetoed if it sits outside
         every whole-cat box NanoDet found (catdet.py) - that removes leaf
         patches and people's faces without touching anything on the cat.
      If nothing is found and NanoDet did see a cat, raise CatFoundButNoFace
      so the app can say "I see a cat, but not its face".
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h_full, w_full = gray.shape
    scale = min(1.0, _DETECT_MAX_SIDE / max(h_full, w_full))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    small_bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr

    cats = catdet.find_cats(small_bgr)

    box, detector, reliable = None, "", False

    faces = facedet.find_faces(small_bgr)
    faces = [f for f in faces if catdet.inside_any(f.box, cats)]
    if faces:
        box, detector, reliable = facedet.to_predictor_framing(faces[0].box), "yolox-face", True

    if box is None:
        for stage_fn, name, trust in (
            (_stage_strict, "haar", True),
            (_stage_rotated, "haar-rotated", True),
            (_stage_loose, "haar-loose", False),
        ):
            candidate = stage_fn(small)
            if candidate is not None and catdet.inside_any(candidate, cats):
                box, detector, reliable = candidate, name, trust
                break

        if box is not None and reliable:
            container = _containing_face(small, box)
            if container is not None:
                box, detector = container, "haar-contained"

    if box is None:
        if cats:
            raise CatFoundButNoFace("a cat is in the photo but its face is not readable")
        raise NoCatFaceFound("no cat face found at any detection stage")

    # Map the box back onto the full-resolution image.
    box = tuple(int(round(v / scale)) for v in box)
    x, y, w, h = box
    rect = dlib.rectangle(x, y, x + w, y + h)
    shape = _shape_predictor()(bgr, rect)
    points = [(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)]

    return DetectedFace(box=box, landmarks=points, detector=detector, reliable=reliable)
