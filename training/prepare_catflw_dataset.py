"""
Build dlib training XML from the CatFLW dataset (Martvel et al., CC BY-NC 4.0;
downloaded by hand from Kaggle - https://www.kaggle.com/datasets/georgemartvel/catflw,
account required, no permissive/no-account mirror exists - checked GitHub
releases, Hugging Face Hub and Zenodo).

Ground truth, read directly from the label JSON files (not assumed from
secondhand paper summaries, which disagreed with each other and with the
actual data): 48 (x, y) points and a [x0, y0, x1, y1] box per image.
Region mapping, confirmed visually (training/data/catflw_points_*.jpg, made
while building this script - see the session log):

    per eye (7-8 pts):  1 outer corner, 1 inner corner, a 3-4-point upper
                        eyelid cluster, 1 lower eyelid point
    left eye:  4 outer, 5 inner, 6/36/37 upper lid, 3 upper-mid, 7 lower
    right eye: 9 inner, 8 outer, 10/39/40/41 upper lid, 1 upper-mid, 11 lower
    ears (4-5 pts each): tip + several base points along the pinna edge
    left ear:  22,23,24(tip),25,26     right ear: 27,28(tip),29,30
    nose/mouth/whiskers: 12,13,14,44,45,15 (nose bridge/nostril), 16,17,0
                        (philtrum/chin midline), 20,18,2,19 (lower lip/chin),
                        21 (chin tip), 33/46 (left whisker pad), 34/43 (right
                        whisker pad)

This gives real eyelid points (contrary to one secondhand summary that
guessed there were none) - eye aperture (eyelid gap / eye width) becomes
measurable - and whisker-pad points, though not whisker orientation itself
(no points along the whiskers, only at their base).

Split: by cat identity, not by image, so no cat appears in more than one
split. Filenames mix two conventions (e.g. "00000001_000.png" and
"CAT_01_00000142_003.png") - both encode an 8-digit id; that id is the
group key. 339 identities, capped at 15 images each.

    python prepare_catflw_dataset.py
"""
import json
import random
import re
from pathlib import Path
from xml.sax.saxutils import escape

import cv2

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "catflw_raw" / "CatFLW dataset"
OUT = HERE / "data" / "catflw"
N_POINTS = 48
BOX_PAD = 0.06  # the given box sometimes just touches a point; pad so dlib never clips one
random.seed(7)

ID_RE = re.compile(r"(\d{8})_\d+$")


def group_key(stem: str) -> str:
    m = ID_RE.search(stem)
    if not m:
        raise ValueError(f"unexpected filename stem: {stem}")
    return m.group(1)


def load_all():
    items = []
    for label_path in sorted((RAW / "labels").glob("*.json")):
        stem = label_path.stem
        img_path = RAW / "images" / f"{stem}.png"
        if not img_path.exists():
            continue
        d = json.load(open(label_path))
        pts = d["labels"]
        box = d["bounding_boxes"]
        if len(pts) != N_POINTS:
            print(f"skipping {stem}: {len(pts)} points, expected {N_POINTS}")
            continue
        items.append((stem, img_path, pts, box))
    return items


def padded_box(pts, box, w, h):
    x0, y0, x1, y1 = box
    # box must contain every point, plus BOX_PAD of the box's own size as slack
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, y0 = min(x0, min(xs)), min(y0, min(ys))
    x1, y1 = max(x1, max(xs)), max(y1, max(ys))
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * BOX_PAD; x1 += bw * BOX_PAD
    y0 -= bh * BOX_PAD; y1 += bh * BOX_PAD
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w), x1), min(float(h), y1)
    return x0, y0, x1 - x0, y1 - y0


def write_xml(path, items):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<dataset>", "<images>"]
    for stem, img_path, pts, box in items:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        x, y, bw, bh = padded_box(pts, box, w, h)
        lines.append(f'  <image file="{escape(str(img_path))}">')
        # dlib's XML loader parses box/part coordinates as integers - a
        # value like "0.0" fails its cast ("invalid string = '0.0'"),
        # discovered by running this against the real trainer.
        lines.append(f'    <box top="{int(round(y))}" left="{int(round(x))}" width="{int(round(bw))}" height="{int(round(bh))}">')
        for i, (px, py) in enumerate(pts):
            lines.append(f'      <part name="{i:02d}" x="{int(round(px))}" y="{int(round(py))}"/>')
        lines.append("    </box>")
        lines.append("  </image>")
    lines += ["</images>", "</dataset>"]
    OUT.mkdir(parents=True, exist_ok=True)
    open(path, "w").write("\n".join(lines))


if __name__ == "__main__":
    items = load_all()
    print(f"{len(items)} labelled images")

    by_group = {}
    for it in items:
        by_group.setdefault(group_key(it[0]), []).append(it)
    groups = list(by_group.keys())
    random.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(n * 0.15))
    n_val = max(1, int(n * 0.15))
    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test:n_test + n_val])
    train_groups = set(groups[n_test + n_val:])

    train = [it for g in train_groups for it in by_group[g]]
    val = [it for g in val_groups for it in by_group[g]]
    test = [it for g in test_groups for it in by_group[g]]
    print(f"{len(train_groups)} cat identities / {len(train)} images -> train")
    print(f"{len(val_groups)} cat identities / {len(val)} images -> val")
    print(f"{len(test_groups)} cat identities / {len(test)} images -> test")
    assert not (train_groups & val_groups & test_groups)

    write_xml(OUT / "train.xml", train)
    write_xml(OUT / "val.xml", val)
    write_xml(OUT / "test.xml", test)
    print("wrote", OUT / "train.xml", OUT / "val.xml", OUT / "test.xml")
