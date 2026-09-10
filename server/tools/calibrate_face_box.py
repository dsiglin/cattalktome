"""
Measure how the learned face detector's boxes relate to the Haar boxes the
landmark predictor was trained on, and print FRAMING constants for
app/facedet.py.

    python tools/calibrate_face_box.py ../test/fixtures /path/to/more/photos

For every photo where both the learned detector and the strict Haar stage
fire on the same face (IoU >= 0.3), it records size ratio and centre offset
(as fractions of the learned box). Medians become FRAMING. Run it once the
model exists; paste the printed dict into facedet.FRAMING; re-run the
integration tests and the photo survey to confirm readings did not drift.
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import detect as d  # noqa: E402
from app import facedet  # noqa: E402


def main(dirs):
    if not facedet.available():
        print("no learned model at", facedet._MODEL)
        return 1
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
            haar = d._stage_strict(small)
            faces = facedet.find_faces(cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr)
            if haar is None or not faces:
                print(f"{Path(p).name[:30]:30s} haar={haar} learned={[f.box for f in faces[:2]]}")
                continue
            f = faces[0].box
            iou = d._iou(haar, f)
            if iou < 0.3:
                print(f"{Path(p).name[:30]:30s} disagree: haar={haar} learned={f} iou={iou:.2f}")
                continue
            ratios.append(np.sqrt((haar[2] * haar[3]) / (f[2] * f[3])))
            dxs.append(((haar[0] + haar[2] / 2) - (f[0] + f[2] / 2)) / f[2])
            dys.append(((haar[1] + haar[3] / 2) - (f[1] + f[3] / 2)) / f[3])
            print(f"{Path(p).name[:30]:30s} ratio={ratios[-1]:.2f} dx={dxs[-1]:+.2f} dy={dys[-1]:+.2f} iou={iou:.2f}")
    if not ratios:
        print("no photo had both detectors agreeing; cannot calibrate")
        return 1
    print(f"\n{len(ratios)} agreeing photos")
    print("FRAMING = {" + f'"scale": {np.median(ratios):.3f}, "dx": {np.median(dxs):.3f}, "dy": {np.median(dys):.3f}' + "}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["../test/fixtures"]))
