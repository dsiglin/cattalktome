# Training the cat-face detector

Fine-tunes **YOLOX-Nano** (Megvii, Apache-2.0) to find a cat's face at any head
angle. It replaces the OpenCV Haar cascades in `server/app/detect.py`, which
miss tilted faces and fire on fur. Output: `runs/catface/catface_yolox_nano.onnx`
(~4 MB), loaded by `server/app/facedet.py` when copied to `server/models/`.

## Data - no accounts needed

The original public source: the Zhang et al. 2008 "Cat Head Detection" dataset
on archive.org (`CAT_DATASET_01.zip` 1.2 GB, `CAT_DATASET_02.zip` 0.95 GB;
~10,000 images with 9 landmarks: eyes, mouth, three points per ear).

- `fetch_dataset.py` downloads both zips in parallel range chunks (archive.org
  throttles single connections to ~150 KB/s) and verifies their MD5 against
  archive.org's own values.
- `prepare_catface_dataset.py` extracts them, applies zylamarek/cat-dataset's
  MIT-licensed cleaning list (`third_party/zylamarek/config.json`: one corrupt
  label replaced, 427 duplicates / not-one-clear-face images removed), derives
  a square face box from the 9 landmarks (ear tips to chin, framed like the
  extended Haar cascade), and writes COCO JSON:
  `train` 11,279 images (7,008 originals + rotated copies of 60% at ±15-45°,
  boxes recomputed from the rotated landmarks), `val` 1,267, `val_rot` 1,267
  (rotated 20-40°, to measure rotation robustness), `test` 1,295.
  It also writes `data/contact_sheet.jpg` for a human check of the boxes.

## Training - on this Mac's GPU

`train_catface.py` is a compact device-agnostic loop (Apple MPS, CUDA, or CPU)
around YOLOX's model, mosaic data pipeline, optimiser, schedule and EMA.
YOLOX's own Trainer hardcodes CUDA. It needs the YOLOX 0.3.0 checkout in
`YOLOX/` with `yolox_mps.patch` applied:

```bash
cd training
/usr/local/bin/python3.13 -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy "opencv-python-headless==4.14.0.94" loguru tqdm thop ninja tabulate psutil tensorboard pycocotools onnx onnxsim onnxruntime
git clone --depth 1 --branch 0.3.0 https://github.com/Megvii-BaseDetection/YOLOX.git
(cd YOLOX && git apply ../yolox_mps.patch)
curl -sL -o yolox_nano.pth https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
python fetch_dataset.py && python prepare_catface_dataset.py
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=YOLOX python train_catface.py --epochs 30 --batch 32 --workers 4
PYTHONPATH=YOLOX python train_catface.py --eval-only runs/catface/best.pth   # val / val_rot / test AP
PYTHONPATH=YOLOX python train_catface.py --export runs/catface/best.pth      # ONNX + cv2.dnn agreement check
```

What `yolox_mps.patch` changes and why (measured, not guessed):
- `.type(tensor.type())` casts use a type *string* that MPS has no name for;
  replaced with `.to(tensor.device, tensor.dtype)`.
- SimOTA label assignment runs on the CPU when the model is not on CUDA. Its
  per-image loop is hundreds of tiny ops; on MPS each costs ~0.3-0.5 ms of
  dispatch, which made forward+loss 938 ms per batch. On CPU: 425 ms. The math
  is identical; only the device differs. YOLOX already had this code path as
  an out-of-memory fallback, ending in `.cuda()`; that became `.to(device)`.

Throughput on an M4 Max: ~50 img/s in the real loop, ~3.6 min/epoch,
30 epochs ≈ 1.8 h. After **one** epoch the smoke run already measured
val AP50 0.892, rotated-val AP50 0.789, test AP50 0.892.

`profile_train.py` times each phase of an iteration; `train_catface.py
--bench-loader` measures the loader and the bare GPU step.

## Unattended run

`run_pipeline.sh` does verify → prepare → train → evaluate → export, logging to
`runs/pipeline.log`, keeping the Mac awake with `caffeinate`. It was scheduled
once via a one-shot LaunchAgent (`~/Library/LaunchAgents/com.cattalktome.train-catface.plist`,
17:30 local) that the script removes when it finishes.

## Colab alternative

`catface_detector_colab.ipynb` does the same on a free T4 GPU using the
Roboflow-hosted copy of the same data (CC BY 4.0, needs a Roboflow API key).
Kept as a fallback; the local pipeline above needs no accounts.

## Licenses
| Thing | License | Credit |
|---|---|---|
| Zhang et al. 2008 CAT dataset (archive.org) | not stated (academic release; Flickr photos) - same caveat pycatfd carries | README |
| zylamarek/cat-dataset cleaning list | MIT | `third_party/zylamarek/LICENSE` |
| YOLOX code and COCO-pretrained `yolox_nano.pth` | Apache-2.0 | README |
| NanoDet whole-cat model (OpenCV model zoo) | Apache-2.0 | README |
| Roboflow "cat-face-data" (notebook path only) | CC BY 4.0 | footer + README if used |

Non-goals: no "cat emotion" dataset is used anywhere. Every one we examined
was self-labelled with no expert review.
