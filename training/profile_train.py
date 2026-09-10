"""Profile the real training loop for a few dozen iterations under different
settings, to find what limits throughput on this device."""
import sys
import time

import torch

from train_catface import Exp, pick_device, random_resize
from yolox.utils import ModelEMA, load_ckpt


def run(label, multiscale, ema_on, iters=30, batch=32, workers=8):
    device = pick_device()
    exp = Exp(epochs=30)
    exp.data_num_workers = workers
    exp.model = None
    model = exp.get_model().to(device)
    ck = torch.load("yolox_nano.pth", map_location="cpu", weights_only=False)
    model = load_ckpt(model, ck["model"])
    loader = exp.get_data_loader(batch, is_distributed=False)
    opt = exp.get_optimizer(batch)
    for g in opt.param_groups:
        g["lr"] = 1e-3
    ema = ModelEMA(model, 0.9998) if ema_on else None
    it = iter(loader)
    for _ in range(3):  # warm up
        inps, targets, _, _ = next(it)
        inps, targets = inps.to(device), targets.to(device)
        loss = model(inps, targets)["total_loss"]; opt.zero_grad(); loss.backward(); opt.step()
    if device.type == "mps":
        torch.mps.synchronize()
    t = time.time()
    for i in range(iters):
        inps, targets, _, _ = next(it)
        inps, targets = inps.to(device), targets.to(device)
        inps, targets = exp.preprocess(inps, targets, exp.input_size)
        loss = model(inps, targets)["total_loss"]
        opt.zero_grad(); loss.backward(); opt.step()
        if ema is not None:
            ema.update(model)
        if multiscale and (i + 1) % 10 == 0:
            exp.input_size = random_resize(exp, loader, 0)
    if device.type == "mps":
        torch.mps.synchronize()
    print(f"{label:40s} {iters * batch / (time.time() - t):.0f} img/s", flush=True)


def timed(workers=4, iters=25, batch=32):
    device = pick_device()
    exp = Exp(epochs=30); exp.data_num_workers = workers; exp.model = None
    model = exp.get_model().to(device)
    model = load_ckpt(model, torch.load("yolox_nano.pth", map_location="cpu", weights_only=False)["model"])
    loader = exp.get_data_loader(batch, is_distributed=False)
    opt = exp.get_optimizer(batch)
    for g in opt.param_groups: g["lr"] = 1e-3
    it = iter(loader)
    for _ in range(3):
        inps, targets, _, _ = next(it)
        loss = model(inps.to(device), targets.to(device))["total_loss"]; opt.zero_grad(); loss.backward(); opt.step()
    torch.mps.synchronize()
    t_next = t_copy = t_fwd = t_bwd = t_opt = 0.0
    ngt = []
    for i in range(iters):
        t0 = time.time(); inps, targets, _, _ = next(it); t1 = time.time()
        ngt.append(int((targets.sum(dim=2) > 0).sum().item()))
        inps, targets = inps.to(device), targets.to(device); torch.mps.synchronize(); t2 = time.time()
        out = model(inps, targets); loss = out["total_loss"]; torch.mps.synchronize(); t3 = time.time()
        opt.zero_grad(); loss.backward(); torch.mps.synchronize(); t4 = time.time()
        opt.step(); torch.mps.synchronize(); t5 = time.time()
        t_next += t1 - t0; t_copy += t2 - t1; t_fwd += t3 - t2; t_bwd += t4 - t3; t_opt += t5 - t4
    n = iters
    print(f"per batch: next(it) {t_next/n*1000:.0f} ms | to(mps) {t_copy/n*1000:.0f} ms | fwd+loss {t_fwd/n*1000:.0f} ms | backward {t_bwd/n*1000:.0f} ms | step {t_opt/n*1000:.0f} ms | GT boxes/batch avg {sum(ngt)/len(ngt):.0f} | dtype {inps.dtype}")


if __name__ == "__main__":
    timed()
