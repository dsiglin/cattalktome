"""
Build the cat-face detection dataset from its original public source.

Source: the Zhang et al. 2008 "Cat Head Detection" dataset on archive.org
(CAT_DATASET_01.zip, CAT_DATASET_02.zip; ~10,000 images, 9 landmarks each:
two eyes, mouth, three points per ear). Cleaning follows zylamarek/cat-dataset
(MIT, third_party/zylamarek/config.json): one corrupt label file replaced,
427 duplicates and images without exactly one visible cat face removed,
split CAT_00-04 train / CAT_05 val / CAT_06 test.

A face box is derived from the 9 landmarks: the tight box around them,
extended below the mouth to include the chin (0.45 x inter-ocular distance),
padded 8% on the sides and 5% on top, then made square. That framing is
close to the OpenCV extended cat-face cascade the landmark predictor was
calibrated against; tools/calibrate_face_box.py measures the residual.

Rotation augmentation: 60% of training images also get a rotated copy
(+-15..45 degrees, boxes recomputed from the rotated landmarks, which is
tighter than rotating the box). The validation split also gets a rotated
copy (20..40 degrees) so rotation robustness is measured, not assumed.

    python prepare_catface_dataset.py            # uses training/data
"""
import json
import math
import os
import random
import shutil
import sys
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ORIG = DATA / "original"
OUT = DATA / "catface"
CFG = json.load(open(HERE / "third_party" / "zylamarek" / "config.json"))

SPLITS = {"train": CFG["split"]["training"]["subdirs"],
          "val": CFG["split"]["validation"]["subdirs"],
          "test": CFG["split"]["test"]["subdirs"]}
ROT_TRAIN_SHARE = 0.60
random.seed(7)


def extract():
    ORIG.mkdir(parents=True, exist_ok=True)
    for name in ("CAT_DATASET_01.zip", "CAT_DATASET_02.zip"):
        marker = ORIG / f".{name}.extracted"
        if marker.exists():
            continue
        print("extracting", name)
        with zipfile.ZipFile(DATA / name) as z:
            z.extractall(ORIG)
        marker.touch()
    # the zips may nest a folder; find the CAT_0x dirs
    found = {p.name: p for p in ORIG.rglob("CAT_0*") if p.is_dir()}
    assert len(found) == 7, f"expected CAT_00..CAT_06, found {sorted(found)}"
    return found


def apply_fixes(dirs):
    # one label file in the original is wrong; zylamarek publishes the corrected one
    for rep in CFG["replace"]:
        url, md5 = CFG["dataset_urls"][rep["filename_to"]]["url"], None
        target = dirs[rep["dir"]] / rep["filename_to"]
        bad = dirs[rep["dir"]] / rep["filename_from"]
        if bad.exists():
            bad.unlink()
        def looks_like_label(path):
            try:
                vals = open(path).read().split()
                return len(vals) >= 19 and all(v.lstrip("-").isdigit() for v in vals[:19])
            except Exception:
                return False
        for attempt in range(8):
            if target.exists() and looks_like_label(target):
                break
            print("fetching corrected label", rep["filename_to"], f"(attempt {attempt + 1})")
            subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(target), url], check=True)
            import time; time.sleep(2)
        assert looks_like_label(target), f"could not fetch a valid {rep['filename_to']} (archive.org throttling?)"
    removed = 0
    for subdir, files in CFG["remove"].items():
        for fn in files:
            for p in (dirs[subdir] / fn, dirs[subdir] / (fn + ".cat")):
                if p.exists():
                    p.unlink()
                    removed += 1
    print("removed", removed // 2, "images flagged by zylamarek (duplicates / not one clear cat face)")


def read_cat(path):
    vals = [int(float(v)) for v in open(path).read().split()]
    n = vals[0]
    pts = np.array(vals[1:1 + 2 * n], dtype=np.float32).reshape(-1, 2)
    return pts


def box_from_landmarks(pts, w, h):
    """pts: 9x2 (left eye, right eye, mouth, 3 left-ear, 3 right-ear)."""
    iod = float(np.linalg.norm(pts[0] - pts[1]))
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    chin = pts[2, 1] + 0.45 * iod
    y1 = max(y1, chin)
    bw, bh = x1 - x0, y1 - y0
    x0 -= 0.08 * bw; x1 += 0.08 * bw
    y0 -= 0.05 * bh
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = cx - side / 2, cy - side / 2
    x1, y1 = cx + side / 2, cy + side / 2
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w), x1), min(float(h), y1)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def rotate(img, pts, deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REPLICATE)
    p = np.hstack([pts, np.ones((len(pts), 1), np.float32)]) @ m.T
    return out, p.astype(np.float32)


def build_split(name, dirs, rotated_share=0.0, rot_range=(15, 45), rotate_all_into=None):
    img_dir = OUT / name
    img_dir.mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    iid = aid = 1
    rot_images, rot_anns = [], []
    for subdir in dirs:
        for cat_file in sorted(Path(SPLITS_DIRS[subdir]).glob("*.jpg.cat")):
            jpg = cat_file.with_suffix("")  # strip .cat -> .jpg
            if not jpg.exists():
                continue
            img = cv2.imread(str(jpg))
            if img is None:
                continue
            h, w = img.shape[:2]
            try:
                pts = read_cat(cat_file)
            except ValueError:
                print("  skipping unparsable label", cat_file.name)
                continue
            if len(pts) != 9:
                continue
            fname = f"{subdir}_{jpg.name}"
            dst = img_dir / fname
            if not dst.exists():
                shutil.copyfile(jpg, dst)
            b = box_from_landmarks(pts, w, h)
            if b[2] < 8 or b[3] < 8:
                continue
            images.append({"id": iid, "file_name": fname, "width": w, "height": h})
            anns.append({"id": aid, "image_id": iid, "category_id": 1, "bbox": b, "area": b[2] * b[3], "iscrowd": 0})
            iid += 1; aid += 1
            if rotated_share and random.random() < rotated_share:
                deg = random.choice([-1, 1]) * random.uniform(*rot_range)
                rimg, rpts = rotate(img, pts, deg)
                rname = f"rot{deg:+.0f}_{fname}"
                cv2.imwrite(str(img_dir / rname), rimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rb = box_from_landmarks(rpts, rimg.shape[1], rimg.shape[0])
                images.append({"id": iid, "file_name": rname, "width": rimg.shape[1], "height": rimg.shape[0]})
                anns.append({"id": aid, "image_id": iid, "category_id": 1, "bbox": rb, "area": rb[2] * rb[3], "iscrowd": 0})
                iid += 1; aid += 1
            if rotate_all_into:
                rdir = OUT / rotate_all_into
                rdir.mkdir(exist_ok=True)
                deg = random.choice([-1, 1]) * random.uniform(20, 40)
                rimg, rpts = rotate(img, pts, deg)
                cv2.imwrite(str(rdir / fname), rimg, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rb = box_from_landmarks(rpts, rimg.shape[1], rimg.shape[0])
                rot_images.append({"id": len(rot_images) + 1, "file_name": fname, "width": rimg.shape[1], "height": rimg.shape[0]})
                rot_anns.append({"id": len(rot_anns) + 1, "image_id": len(rot_images), "category_id": 1, "bbox": rb, "area": rb[2] * rb[3], "iscrowd": 0})
    cats = [{"id": 1, "name": "cat_face", "supercategory": "cat"}]
    (OUT / "annotations").mkdir(exist_ok=True)
    json.dump({"images": images, "annotations": anns, "categories": cats}, open(OUT / "annotations" / f"{name}.json", "w"))
    print(f"{name}: {len(images)} images, {len(anns)} boxes")
    if rotate_all_into:
        json.dump({"images": rot_images, "annotations": rot_anns, "categories": cats}, open(OUT / "annotations" / f"{rotate_all_into}.json", "w"))
        print(f"{rotate_all_into}: {len(rot_images)} images")
    return images, anns


if __name__ == "__main__":
    dirs = extract()
    apply_fixes(dirs)
    SPLITS_DIRS = {k: str(v) for k, v in dirs.items()}
    _, train_anns = build_split("train", SPLITS["train"], rotated_share=ROT_TRAIN_SHARE)
    build_split("val", SPLITS["val"], rotate_all_into="val_rot")
    build_split("test", SPLITS["test"])
    ws = np.array([a["bbox"][2] for a in train_anns])
    print(f"train face width px: min {ws.min():.0f} p5 {np.percentile(ws, 5):.0f} median {np.median(ws):.0f} p95 {np.percentile(ws, 95):.0f}")
    # a contact sheet of a few boxes for a human check
    tr = json.load(open(OUT / "annotations" / "train.json"))
    by = {a["image_id"]: a for a in tr["annotations"]}
    tiles = []
    for im in random.sample(tr["images"], 12):
        img = cv2.imread(str(OUT / "train" / im["file_name"]))
        x, y, w, h = map(int, by[im["id"]]["bbox"])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), max(2, img.shape[1] // 300))
        tiles.append(cv2.resize(img, (300, 300)))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)])
    cv2.imwrite(str(DATA / "contact_sheet.jpg"), sheet)
    print("wrote", DATA / "contact_sheet.jpg")
