"""
Fine-tune YOLOX-Nano (Apache-2.0) on the cat-face dataset built by
prepare_catface_dataset.py, on whatever device this machine has (Apple MPS,
CUDA, or CPU), then evaluate and export ONNX.

YOLOX's own Trainer hardcodes CUDA, so this is a compact, device-agnostic
loop around YOLOX's model, mosaic data pipeline, optimiser, LR schedule and
EMA - the same recipe, minus the distributed and CUDA-only plumbing.

    PYTHONPATH=YOLOX python train_catface.py --epochs 30 --batch 32
    PYTHONPATH=YOLOX python train_catface.py --eval-only runs/catface/best.pth
    PYTHONPATH=YOLOX python train_catface.py --export runs/catface/best.pth
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "YOLOX"))

from yolox.exp import Exp as BaseExp  # noqa: E402
from yolox.utils import ModelEMA, load_ckpt, postprocess, xyxy2xywh  # noqa: E402

DATA = HERE / "data" / "catface"
RUNS = HERE / "runs" / "catface"
PRETRAINED_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"


class Exp(BaseExp):
    def __init__(self, epochs=30, input_size=416):
        super().__init__()
        self.depth, self.width, self.depthwise = 0.33, 0.25, True
        self.input_size = self.test_size = (input_size, input_size)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.num_classes = 1
        self.data_dir = str(DATA)
        self.train_ann, self.val_ann = "train.json", "val.json"
        self.max_epoch = epochs
        self.no_aug_epochs = max(2, epochs // 6)
        self.warmup_epochs = 1
        self.data_num_workers = 8
        self.flip_prob, self.hsv_prob, self.degrees = 0.5, 1.0, 0.0
        self.eval_interval = 1
        self.exp_name = "catface_nano"

    def get_model(self, sublinear=False):
        import torch.nn as nn
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
        if getattr(self, "model", None) is None:
            ch = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=ch, act=self.act, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=ch, act=self.act, depthwise=True)
            self.model = YOLOX(backbone, head)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(data_dir=self.data_dir, json_file=self.train_ann, name="train", img_size=self.input_size,
                           preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                           cache=False)

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        return COCODataset(data_dir=self.data_dir, json_file=getattr(self, "eval_ann", self.val_ann),
                           name=getattr(self, "eval_name", "val"), img_size=self.test_size,
                           preproc=ValTransform(legacy=False))

    # YOLOX 0.3.0's loaders build COCODataset themselves with hardcoded folder
    # names (train2017 / val2017) and pin_memory=True (CUDA-only). Override both
    # so our folder names and any eval split work, on any device.
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (TrainTransform, YoloBatchSampler, DataLoader, InfiniteSampler,
                                MosaicDetection, worker_init_reset_seed)
        dataset = self.get_dataset()
        dataset = MosaicDetection(
            dataset, mosaic=not no_aug, img_size=self.input_size,
            preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            degrees=self.degrees, translate=self.translate, mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale, shear=self.shear, enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob, mixup_prob=self.mixup_prob,
        )
        self.dataset = dataset
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug)
        return DataLoader(self.dataset, num_workers=self.data_num_workers, pin_memory=False,
                          batch_sampler=batch_sampler, worker_init_fn=worker_init_reset_seed)

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        valdataset = self.get_eval_dataset(legacy=legacy)
        sampler = torch.utils.data.SequentialSampler(valdataset)
        return torch.utils.data.DataLoader(valdataset, num_workers=self.data_num_workers, pin_memory=False,
                                           sampler=sampler, batch_size=batch_size)


def random_resize(exp, loader, epoch):
    """YOLOX's Exp.random_resize, minus the .cuda() it uses for a distributed
    broadcast. Multi-scale training: pick a new input size every 10 iters,
    in steps of 32 within exp.random_size, keeping the aspect ratio."""
    import random
    if epoch >= exp.max_epoch - exp.no_aug_epochs:
        return exp.test_size
    size_factor = exp.input_size[1] * 1.0 / exp.input_size[0]
    size = random.randint(*exp.random_size)
    return (int(32 * size), 32 * int(size * size_factor))


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(exp, model, device, ann="val.json", name="val", batch=32, quiet=False):
    """COCO AP on a split. Returns (AP50:95, AP50)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    exp.eval_ann, exp.eval_name = ann, name
    loader = exp.get_eval_loader(batch, is_distributed=False)
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, _, info, ids in loader:
            out = model(imgs.to(device))
            out = postprocess(out, exp.num_classes, 0.001, 0.65)
            for o, ih, iw, iid in zip(out, info[0], info[1], ids):
                if o is None:
                    continue
                o = o.cpu().numpy()
                scale = min(exp.test_size[0] / float(ih), exp.test_size[1] / float(iw))
                boxes = xyxy2xywh(torch.from_numpy(o[:, :4] / scale)).numpy()
                for b, s in zip(boxes, o[:, 4] * o[:, 5]):
                    results.append({"image_id": int(iid), "category_id": 1, "bbox": [float(v) for v in b], "score": float(s)})
    model.train()
    if not results:
        return 0.0, 0.0
    gt = COCO(str(DATA / "annotations" / ann))
    dt = gt.loadRes(results)
    ev = COCOeval(gt, dt, "bbox")
    if quiet:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate(); ev.accumulate(); ev.summarize()
    else:
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def bench(args):
    device = pick_device()
    exp = Exp(epochs=30, input_size=args.input)
    exp.data_num_workers = args.workers
    loader = exp.get_data_loader(args.batch, is_distributed=False)
    it = iter(loader)
    next(it)
    n = 20
    t = time.time()
    for _ in range(n):
        next(it)
    print(f"loader only ({args.workers} workers, batch {args.batch}): {n * args.batch / (time.time() - t):.0f} img/s")
    model = exp.get_model().to(device)
    opt = exp.get_optimizer(args.batch)
    for g in opt.param_groups:
        g["lr"] = 1e-3
    for size in (416, 640):
        inps = (torch.rand(args.batch, 3, size, size) * 255).to(device)
        targets = torch.zeros(args.batch, 50, 5)
        targets[:, 0] = torch.tensor([0, size / 2, size / 2, size / 4, size / 4])
        targets = targets.to(device)
        for i in range(8):
            if i == 3:
                if device.type == "mps":
                    torch.mps.synchronize()
                t = time.time()
            loss = model(inps, targets)["total_loss"]
            opt.zero_grad(); loss.backward(); opt.step()
        if device.type == "mps":
            torch.mps.synchronize()
        print(f"GPU step only at {size}px (batch {args.batch}): {5 * args.batch / (time.time() - t):.0f} img/s")


def train(args):
    device = pick_device()
    torch.manual_seed(0)
    exp = Exp(epochs=args.epochs, input_size=args.input)
    exp.data_num_workers = args.workers
    RUNS.mkdir(parents=True, exist_ok=True)
    model = exp.get_model().to(device)

    pre = HERE / "yolox_nano.pth"
    if not pre.exists():
        import subprocess
        print("downloading COCO-pretrained yolox_nano.pth")
        subprocess.run(["curl", "-sL", "-o", str(pre), PRETRAINED_URL], check=True)
    ckpt = torch.load(pre, map_location="cpu", weights_only=False)
    model = load_ckpt(model, ckpt["model"])  # skips the 80-class head tensors that no longer match

    loader = exp.get_data_loader(args.batch, is_distributed=False, no_aug=False)
    iters_per_epoch = len(loader)
    optimizer = exp.get_optimizer(args.batch)
    scheduler = exp.get_lr_scheduler(exp.basic_lr_per_img * args.batch, iters_per_epoch)
    ema = ModelEMA(model, 0.9998)
    ema.updates = 0

    print(f"device={device} epochs={args.epochs} batch={args.batch} iters/epoch={iters_per_epoch} input={exp.input_size}")
    best_ap50, log = 0.0, []
    it_total = 0
    t_start = time.time()
    for epoch in range(args.epochs):
        if epoch == args.epochs - exp.no_aug_epochs:
            print("closing mosaic and enabling L1 loss for the final epochs")
            loader.close_mosaic()
            model.head.use_l1 = True
        model.train()
        it = iter(loader)
        t_ep = time.time()
        for i in range(iters_per_epoch):
            inps, targets, _, _ = next(it)
            inps, targets = inps.to(device), targets.to(device)
            inps, targets = exp.preprocess(inps, targets, exp.input_size)
            outputs = model(inps, targets)
            loss = outputs["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            lr = scheduler.update_lr(it_total + 1)
            for g in optimizer.param_groups:
                g["lr"] = lr
            it_total += 1
            if i % 20 == 0:
                el = time.time() - t_ep
                rate = (i + 1) * args.batch / max(el, 1e-6)
                print(f"ep {epoch + 1}/{args.epochs} it {i}/{iters_per_epoch} loss {loss.item():.3f} "
                      f"iou {outputs['iou_loss'].item():.3f} conf {outputs['conf_loss'].item():.3f} "
                      f"lr {lr:.2e} {rate:.0f} img/s", flush=True)
            if (i + 1) % 10 == 0:
                exp.input_size = random_resize(exp, loader, epoch)
        exp.input_size = exp.test_size if epoch >= args.epochs - exp.no_aug_epochs else exp.input_size
        ap, ap50 = evaluate(exp, ema.ema, device, batch=args.batch, quiet=True)
        log.append({"epoch": epoch + 1, "ap": ap, "ap50": ap50, "minutes": (time.time() - t_start) / 60})
        print(f"== epoch {epoch + 1}: val AP {ap:.3f} AP50 {ap50:.3f}  ({(time.time() - t_ep) / 60:.1f} min)", flush=True)
        torch.save({"model": ema.ema.state_dict(), "epoch": epoch + 1, "ap50": ap50}, RUNS / "last.pth")
        if ap50 > best_ap50:
            best_ap50 = ap50
            torch.save({"model": ema.ema.state_dict(), "epoch": epoch + 1, "ap50": ap50}, RUNS / "best.pth")
        json.dump(log, open(RUNS / "log.json", "w"), indent=1)
    print(f"done in {(time.time() - t_start) / 60:.0f} min; best val AP50 {best_ap50:.3f}")


def eval_only(path, args):
    device = pick_device()
    exp = Exp(input_size=args.input)
    model = exp.get_model().to(device)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    for ann, name in (("val.json", "val"), ("val_rot.json", "val_rot"), ("test.json", "test")):
        ap, ap50 = evaluate(exp, model, device, ann, name, batch=args.batch, quiet=True)
        print(f"{name:8s} AP {ap:.3f}  AP50 {ap50:.3f}")


def export(path, args):
    import onnx
    import onnxsim
    exp = Exp(input_size=args.input)
    model = exp.get_model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    model.eval()
    model.head.decode_in_inference = True
    dummy = torch.randn(1, 3, args.input, args.input)
    out = RUNS / "catface_yolox_nano.onnx"
    torch.onnx.export(model, dummy, str(out), input_names=["images"], output_names=["output"], opset_version=11, dynamo=False)
    m, ok = onnxsim.simplify(onnx.load(str(out)))
    assert ok, "onnxsim failed"
    onnx.save(m, str(out))
    print("exported", out, f"{out.stat().st_size / 1e6:.1f} MB")
    # prove OpenCV can run it and agrees with torch
    import cv2
    net = cv2.dnn.readNetFromONNX(str(out))
    x = (torch.rand(1, 3, args.input, args.input) * 255).round()
    with torch.no_grad():
        t = model(x)[0].numpy()
    net.setInput(x.numpy())
    c = net.forward()[0]
    print("cv2.dnn vs torch max abs diff on box/obj columns:", float(np.abs(t[:, :5] - c[:, :5]).max()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--input", type=int, default=416)
    ap.add_argument("--eval-only", type=str, default=None)
    ap.add_argument("--export", type=str, default=None)
    ap.add_argument("--bench-loader", action="store_true", help="measure data-loader and GPU-step throughput, then exit")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.bench_loader:
        bench(a)
    elif a.eval_only:
        eval_only(a.eval_only, a)
    elif a.export:
        export(a.export, a)
    else:
        train(a)
