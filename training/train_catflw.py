"""
Train a dlib 48-point cat facial-landmark shape predictor on CatFLW.

dlib's shape predictor is an ensemble of regression trees (Kazemi & Sullivan
2014) - CPU only, no GPU/Metal involved, unlike the YOLOX face detector.
That matters after this session's kernel panic: this cannot repeat it.

    python train_catflw.py                    # full training
    python train_catflw.py --smoke            # quick correctness check (~1 min)
    python train_catflw.py --eval-only MODEL   # evaluate an existing model
"""
import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import dlib
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "catflw"
RUNS = HERE / "runs" / "catflw"
N_POINTS = 48

# Outer corner of each eye (left=4, right=8 - see prepare_catflw_dataset.py's
# docstring). A stable width to normalise landmark error by, the same role
# interocular distance plays in server/app/geometry.py.
LEFT_OUTER, RIGHT_OUTER = 4, 8


def load_xml(path):
    """Return [(image_path, box_xywh, points[48])], parsed from dlib's own
    training XML format (so evaluation reads exactly what training read)."""
    root = ET.parse(path).getroot()
    items = []
    for img_el in root.find("images"):
        file = img_el.get("file")
        box_el = img_el.find("box")
        box = (float(box_el.get("left")), float(box_el.get("top")),
               float(box_el.get("width")), float(box_el.get("height")))
        pts = [None] * N_POINTS
        for part in box_el.findall("part"):
            pts[int(part.get("name"))] = (float(part.get("x")), float(part.get("y")))
        items.append((file, box, pts))
    return items


def train(args):
    RUNS.mkdir(parents=True, exist_ok=True)
    options = dlib.shape_predictor_training_options()
    options.oversampling_amount = args.oversampling
    options.nu = args.nu
    options.tree_depth = args.tree_depth
    options.cascade_depth = args.cascade_depth
    options.feature_pool_size = args.feature_pool_size
    options.num_threads = args.threads
    options.be_verbose = True
    options.num_test_splits = 20

    train_xml = str(DATA / ("smoke_train.xml" if args.smoke else "train.xml"))
    out_path = str(RUNS / ("smoke.dat" if args.smoke else "catflw48.dat"))
    print(f"training on {train_xml} -> {out_path}")
    print(f"options: oversampling={options.oversampling_amount} nu={options.nu} "
          f"tree_depth={options.tree_depth} cascade_depth={options.cascade_depth} "
          f"feature_pool_size={options.feature_pool_size} threads={options.num_threads}")
    t0 = time.time()
    dlib.train_shape_predictor(train_xml, out_path, options)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    evaluate(out_path, splits=("val",) if args.smoke else ("train", "val", "test"))


def evaluate(model_path, splits=("train", "val", "test")):
    predictor = dlib.shape_predictor(model_path)
    for split in splits:
        xml_path = DATA / f"{split}.xml"
        if not xml_path.exists():
            continue
        items = load_xml(xml_path)
        errors, per_point = [], [[] for _ in range(N_POINTS)]
        for file, (x, y, w, h), gt in items:
            img = cv2.imread(file)
            if img is None:
                continue
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(img, rect)
            pred = [(shape.part(i).x, shape.part(i).y) for i in range(N_POINTS)]
            iod = np.hypot(gt[LEFT_OUTER][0] - gt[RIGHT_OUTER][0], gt[LEFT_OUTER][1] - gt[RIGHT_OUTER][1])
            if iod < 1e-3:
                continue
            for i in range(N_POINTS):
                d = np.hypot(pred[i][0] - gt[i][0], pred[i][1] - gt[i][1]) / iod
                per_point[i].append(d)
                errors.append(d)
        errors = np.array(errors)
        print(f"{split:6s} n={len(items):4d}  NME (outer-canthal-normalised): "
              f"mean={errors.mean()*100:.2f}%  median={np.median(errors)*100:.2f}%  "
              f"p90={np.percentile(errors,90)*100:.2f}%")
        worst = sorted(range(N_POINTS), key=lambda i: -np.mean(per_point[i]))[:5]
        best = sorted(range(N_POINTS), key=lambda i: np.mean(per_point[i]))[:5]
        print(f"        worst points: {[(i, round(np.mean(per_point[i])*100,2)) for i in worst]}")
        print(f"        best points:  {[(i, round(np.mean(per_point[i])*100,2)) for i in best]}")


def make_smoke_xml():
    """A ~120-image slice of train.xml, so a full-pipeline correctness check
    takes about a minute instead of the full run's time."""
    import random
    random.seed(0)
    items = load_xml(DATA / "train.xml")
    random.shuffle(items)
    subset = items[:120]
    root = ET.Element("dataset")
    images = ET.SubElement(root, "images")
    for file, (x, y, w, h), pts in subset:
        img_el = ET.SubElement(images, "image", file=file)
        box_el = ET.SubElement(img_el, "box", top=str(int(round(y))), left=str(int(round(x))),
                               width=str(int(round(w))), height=str(int(round(h))))
        for i, (px, py) in enumerate(pts):
            ET.SubElement(box_el, "part", name=f"{i:02d}", x=str(int(round(px))), y=str(int(round(py))))
    ET.ElementTree(root).write(DATA / "smoke_train.xml")
    print("wrote", DATA / "smoke_train.xml", f"({len(subset)} images)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="quick correctness check on ~120 images, few trees")
    ap.add_argument("--eval-only", type=str, default=None)
    ap.add_argument("--oversampling", type=int, default=None)
    ap.add_argument("--nu", type=float, default=0.1)
    ap.add_argument("--tree-depth", type=int, default=4)
    ap.add_argument("--cascade-depth", type=int, default=None)
    ap.add_argument("--feature-pool-size", type=int, default=400)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.oversampling is None:
        a.oversampling = 20 if a.smoke else 300
    if a.cascade_depth is None:
        a.cascade_depth = 3 if a.smoke else 15
    if a.eval_only:
        evaluate(a.eval_only)
    else:
        if a.smoke and not (DATA / "smoke_train.xml").exists():
            make_smoke_xml()
        train(a)
