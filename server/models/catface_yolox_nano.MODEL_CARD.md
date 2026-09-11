# catface_yolox_nano.onnx - model card

Trained 2026-09-11 locally on an Apple M4 Max (MPS) with
training/train_catface.py, finishing all 30 scheduled epochs. (An earlier
attempt crashed the machine at epoch 14/30 via a kernel panic from
sustained GPU load - see training/README.md. That epoch-13 checkpoint
was deployed first; this epoch-30 checkpoint replaces it, having been
resumed safely in three short, watched bursts.)

- Task: one class, "cat_face", axis-aligned box.
- Architecture: YOLOX-Nano (depth 0.33, width 0.25, depthwise), input
  416x416 letterboxed (pad 114), BGR 0-255, no normalisation.
- Output: (1, N, 6) = cx, cy, w, h, objectness, class score, already
  decoded to input-pixel coordinates.
- Fine-tuned from Megvii's COCO checkpoint yolox_nano.pth (Apache-2.0).
- Data: the original Zhang et al. 2008 "Cat Head Detection" dataset
  (archive.org), cleaned via zylamarek/cat-dataset's MIT-licensed list,
  11,279 training images after rotation augmentation (+-15..45 deg on
  60%), boxes derived from the 9 original landmarks.
- Size: 4.1 MB.

## Evaluation (COCO AP)
| Split | AP (50-95) | AP50 |
|---|---|---|
| val | 0.909 | 1.000 |
| val, rotated 20-40 deg | 0.890 | 0.990 |
| test (held out) | 0.908 | 0.990 |

Box precision improved measurably over the epoch-13 checkpoint (AP
0.871 on val and test) while AP50 - does it find the face at all -
was already near its ceiling either way. cv2.dnn vs PyTorch max abs
diff on box/objectness columns: 4e-4.

## A real fragility this exposed, and how it's handled

Box-framing recalibration (`tools/calibrate_face_box.py`, 1,292/1,295
test-split photos matched) confirmed this model's boxes are still
identity-within-noise against the framing the 8-point dlib predictor
expects (scale 1.010, dx 0.004, dy 0.007). But on one real photo (an
extreme close-up, cat looking straight up), a ~10px difference from the
epoch-13 checkpoint's box was enough to put that predictor's right-eye
landmark on the cat's forehead instead of its eye.

Measured broadly (1,295 held-out photos): this is not unique to this
checkpoint. The *currently-deployed* epoch-13 model showed an
almost-identical rate (8.9% vs this model's 8.5% of confident detections
producing two individually-plausible-looking eye readings that disagree
by more than 0.5) - a property of the 8-point predictor's sensitivity to
fine box-framing details, not of either detector specifically. Padding
the box more generously was tried and made the aggregate rate worse
(13.0% at 1.20x), not better - ruled out.

The fix that shipped: `server/app/eyes.py` now treats two
individually-plausible eyes that disagree this much as an unreadable
pair rather than guessing which one is right (`_MAX_TRUSTED_DISAGREEMENT`),
the same "refuse to answer rather than guess" rule already used for a
glowing or squinted eye. This benefits both this checkpoint and the
previous one equally.

## Licenses / attribution
- Zhang et al. 2008 CAT dataset (archive.org): no stated license
  (academic release; same caveat pycatfd already carries).
- zylamarek/cat-dataset cleaning list: MIT.
- YOLOX code and pretrained weights: Apache-2.0 -
  https://github.com/Megvii-BaseDetection/YOLOX
