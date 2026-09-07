"""One-off script: draw the detected face box, 8 landmarks, and the
resulting feeling sticker onto the real cat fixture, for visual proof."""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.pipeline import analyse_image

FIXTURE = Path(__file__).resolve().parent.parent / "test" / "fixtures" / "cat.jpg"
OUT = Path(__file__).resolve().parent / "debug_output.jpg"

bgr = cv2.imread(str(FIXTURE))
result = analyse_image(bgr)

x, y, w, h = result.face.box
cv2.rectangle(bgr, (x, y), (x + w, y + h), (60, 200, 255), 3)

names = ["chin", "left_eye", "LoL_ear", "LoR_ear", "nose", "right_eye", "RoL_ear", "RoR_ear"]
for (px, py), name in zip(result.face.landmarks, names):
    cv2.circle(bgr, (px, py), 5, (60, 60, 255), -1)
    cv2.putText(bgr, name, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.putText(bgr, name, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)

label = f"{result.reading.feeling.emoji} {result.reading.feeling.label}"
(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
sticker_x, sticker_y = x, y + h + 40
cv2.rectangle(bgr, (sticker_x - 15, sticker_y - th - 20), (sticker_x + tw + 15, sticker_y + 15), (230, 246, 255), -1)
cv2.rectangle(bgr, (sticker_x - 15, sticker_y - th - 20), (sticker_x + tw + 15, sticker_y + 15), (61, 163, 232), 3)
cv2.putText(bgr, label, (sticker_x, sticker_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (28, 43, 61), 3)

cv2.imwrite(str(OUT), bgr)
print(f"Detector: {result.face.detector}, box: {result.face.box}")
print(f"Reading: {result.reading.feeling.label} (confidence {result.reading.confidence}, reliable={result.reading.reliable})")
print(f"Saved: {OUT}")
