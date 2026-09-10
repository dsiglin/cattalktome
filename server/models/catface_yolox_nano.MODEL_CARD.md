# catface_yolox_nano.onnx - model card

Trained 2026-09-10 locally on an Apple M4 Max (MPS) with
training/train_catface.py; stopped at epoch 13/30 after a kernel panic
during epoch 14 (unrelated to model quality - see training/README.md
and the session log for the root cause: sustained GPU load wedged
WindowServer). Epoch 13 was already the best checkpoint.

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

## Evaluation (COCO AP), epoch 13 checkpoint
| Split | AP (50-95) | AP50 |
|---|---|---|
| val | 0.871 | 1.000 |
| val, rotated 20-40 deg | 0.836 | 0.990 |
| test (held out) | 0.871 | 0.990 |

cv2.dnn vs PyTorch max abs diff on box/objectness columns: 4e-4.

## Licenses / attribution
- Zhang et al. 2008 CAT dataset (archive.org): no stated license
  (academic release; same caveat pycatfd already carries).
- zylamarek/cat-dataset cleaning list: MIT.
- YOLOX code and pretrained weights: Apache-2.0 -
  https://github.com/Megvii-BaseDetection/YOLOX
