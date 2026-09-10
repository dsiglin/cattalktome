"""
Measure how the learned face detector's boxes relate to the framing the
dlib landmark predictor expects (Haar-cascade style), and print FRAMING
constants for app/facedet.py.

Two data sources, either works, first available wins:

1. Real photos where BOTH the learned detector and the strict Haar stage
   fire on the same face (IoU >= 0.3) - the most direct comparison, but
   needs enough real photos where strict Haar succeeds, which is rare by
   construction (that's why the learned detector exists).

       python calibrate_face_box.py ../test/fixtures /path/to/more/photos

2. The training pipeline's held-out TEST split (../training/data/catface,
   1,295 real photos never used in training), comparing the learned
   detector's predicted box against the ground-truth box that
   prepare_catface_dataset.py derived from the original 9 landmarks - the
   same framing (ear tips to chin, padded) the dlib predictor was
   calibrated against. Far more data; used automatically when no photo
   directories are given, or when they yield too few matches.

       python calibrate_face_box.py

Medians become FRAMING. Paste the printed dict into facedet.FRAMING, then
re-run the integration tests and the photo survey to confirm readings.
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import detect as d  # noqa: E402
from app import facedet  # noqa: E402

MIN_MATCHES = 20  # below this, prefer the test-split comparison


def from_photos(dirs):
    ratios, dxs, dys = [], [], []
    for folder in dirs:
        for p in sorted(glob.glob(f"{folder}/*")):
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            s = min(1.0, d._DETECT_MAX_SIDE / max(h, w))
            small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
            small_bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
            haar = d._stage_strict(small)
            faces = facedet.find_faces(small_bgr)
            if haar is None or not faces:
                continue
            f = faces[0].box
            iou = d._iou(haar, f)
            if iou < 0.3:
                continue
            ratios.append(np.sqrt((haar[2] * haar[3]) / (f[2] * f[3])))
            dxs.append(((haar[0] + haar[2] / 2) - (f[0] + f[2] / 2)) / f[2])
            dys.append(((haar[1] + haar[3] / 2) - (f[1] + f[3] / 2)) / f[3])
            print(f"{Path(p).name[:30]:30s} ratio={ratios[-1]:.2f} dx={dxs[-1]:+.2f} dy={dys[-1]:+.2f} iou={iou:.2f}")
    return ratios, dxs, dys


def from_test_split():
    """Compare against the held-out test split's ground-truth boxes (derived
    from real landmarks, framed like the extended Haar cascade - see
    training/prepare_catface_dataset.py box_from_landmarks)."""
    root = Path(__file__).resolve().parent.parent.parent / "training" / "data" / "catface"
    ann = json.load(open(root / "annotations" / "test.json"))
    gt_by_image = {a["image_id"]: a["bbox"] for a in ann["annotations"]}
    ratios, dxs, dys, matched, total = [], [], [], 0, 0
    for im in ann["images"]:
        gt = gt_by_image.get(im["id"])
        if gt is None:
            continue
        total += 1
        bgr = cv2.imread(str(root / "test" / im["file_name"]))
        if bgr is None:
            continue
        faces = facedet.find_faces(bgr)
        if not faces:
            continue
        f = faces[0].box
        iou = d._iou(tuple(gt), f)
        if iou < 0.3:
            continue
        matched += 1
        ratios.append(np.sqrt((gt[2] * gt[3]) / (f[2] * f[3])))
        dxs.append(((gt[0] + gt[2] / 2) - (f[0] + f[2] / 2)) / f[2])
        dys.append(((gt[1] + gt[3] / 2) - (f[1] + f[3] / 2)) / f[3])
    print(f"test split: {matched}/{total} images matched (IoU >= 0.3 against ground truth)")
    return ratios, dxs, dys


def main(dirs):
    if not facedet.available():
        print("no learned model at", facedet._MODEL)
        return 1
    ratios, dxs, dys = from_photos(dirs) if dirs else ([], [], [])
    if len(ratios) < MIN_MATCHES:
        if ratios:
            print(f"only {len(ratios)} photo matches - falling back to the test split for more data")
        ratios, dxs, dys = from_test_split()
    if not ratios:
        print("no matches found in either source; cannot calibrate")
        return 1
    print(f"\n{len(ratios)} matched boxes")
    print(f"scale: median {np.median(ratios):.3f}  p25 {np.percentile(ratios,25):.3f}  p75 {np.percentile(ratios,75):.3f}")
    print(f"dx:    median {np.median(dxs):.3f}  p25 {np.percentile(dxs,25):.3f}  p75 {np.percentile(dxs,75):.3f}")
    print(f"dy:    median {np.median(dys):.3f}  p25 {np.percentile(dys,25):.3f}  p75 {np.percentile(dys,75):.3f}")
    print("FRAMING = {" + f'"scale": {np.median(ratios):.3f}, "dx": {np.median(dxs):.3f}, "dy": {np.median(dys):.3f}' + "}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
