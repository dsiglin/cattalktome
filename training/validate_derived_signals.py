"""
The question that matters isn't raw per-point NME, it's: does the trained
model's PREDICTED eye aperture / whisker spread track the GROUND-TRUTH
value on real held-out photos? Point error can partially cancel in a ratio,
or compound - only measuring the actual derived signal answers it.

    python validate_derived_signals.py [split]   # default: test
"""
import sys
from pathlib import Path

import cv2
import dlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
from app.geometry48 import eye_aperture, whisker_pad_spread, LEFT_EYE_OUTER, RIGHT_EYE_OUTER  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_catflw import load_xml, DATA, RUNS  # noqa: E402


def main(split="test"):
    predictor = dlib.shape_predictor(str(RUNS / "catflw48.dat"))
    items = load_xml(DATA / f"{split}.xml")
    gt_ap, pred_ap, gt_ws, pred_ws = [], [], [], []
    for file, (x, y, w, h), gt in items:
        img = cv2.imread(file)
        if img is None:
            continue
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(img, rect)
        pred = [(shape.part(i).x, shape.part(i).y) for i in range(48)]

        gt_e, pred_e = eye_aperture(gt), eye_aperture(pred)
        if gt_e and pred_e:
            gt_ap.append(gt_e.average); pred_ap.append(pred_e.average)

        iod_gt = np.hypot(gt[LEFT_EYE_OUTER][0] - gt[RIGHT_EYE_OUTER][0], gt[LEFT_EYE_OUTER][1] - gt[RIGHT_EYE_OUTER][1])
        iod_pred = np.hypot(pred[LEFT_EYE_OUTER][0] - pred[RIGHT_EYE_OUTER][0], pred[LEFT_EYE_OUTER][1] - pred[RIGHT_EYE_OUTER][1])
        gt_w, pred_w = whisker_pad_spread(gt, iod_gt), whisker_pad_spread(pred, iod_pred)
        if gt_w and pred_w:
            gt_ws.append(gt_w); pred_ws.append(pred_w)

    gt_ap, pred_ap = np.array(gt_ap), np.array(pred_ap)
    gt_ws, pred_ws = np.array(gt_ws), np.array(pred_ws)

    print(f"split={split}  n={len(items)}")
    print(f"\neye aperture: n={len(gt_ap)}")
    print(f"  ground truth: mean={gt_ap.mean():.3f} std={gt_ap.std():.3f} range=[{gt_ap.min():.3f}, {gt_ap.max():.3f}]")
    print(f"  predicted:    mean={pred_ap.mean():.3f} std={pred_ap.std():.3f} range=[{pred_ap.min():.3f}, {pred_ap.max():.3f}]")
    r = np.corrcoef(gt_ap, pred_ap)[0, 1]
    mae = np.abs(gt_ap - pred_ap).mean()
    print(f"  Pearson r={r:.3f}   MAE={mae:.3f}  (MAE as a fraction of GT range: {mae/(gt_ap.max()-gt_ap.min()):.2%})")

    print(f"\nwhisker pad spread: n={len(gt_ws)}")
    print(f"  ground truth: mean={gt_ws.mean():.3f} std={gt_ws.std():.3f} range=[{gt_ws.min():.3f}, {gt_ws.max():.3f}]")
    print(f"  predicted:    mean={pred_ws.mean():.3f} std={pred_ws.std():.3f} range=[{pred_ws.min():.3f}, {pred_ws.max():.3f}]")
    r2 = np.corrcoef(gt_ws, pred_ws)[0, 1]
    mae2 = np.abs(gt_ws - pred_ws).mean()
    print(f"  Pearson r={r2:.3f}   MAE={mae2:.3f}  (MAE as a fraction of GT range: {mae2/(gt_ws.max()-gt_ws.min()):.2%})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
