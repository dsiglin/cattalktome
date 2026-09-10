# Training the cat-face detector

`catface_detector_colab.ipynb` fine-tunes **YOLOX-Nano** (Megvii, Apache-2.0) to
find a cat's face at any head angle. It replaces the OpenCV Haar cascades in
`server/app/detect.py`, which miss tilted faces and fire on fur.

Open it in Colab:
https://colab.research.google.com/github/dsiglin/cattalktome/blob/main/training/catface_detector_colab.ipynb

Steps: pick a T4 GPU runtime, run the cells top to bottom, paste your Roboflow
API key when the hidden prompt asks (it is used once and never stored), wait
~45-70 minutes, and download `catface_yolox_nano.onnx` + `MODEL_CARD.md`.
Put both in `server/models/`. The server picks the model up automatically;
then run `python server/tools/calibrate_face_box.py` and the tests.

## Licenses
| Thing | License | Credit |
|---|---|---|
| Dataset "cat-face-data" (Roboflow Universe, workspace `cat-face`, 8,153 images) | CC BY 4.0 | required: footer + README |
| YOLOX code and COCO-pretrained `yolox_nano.pth` | Apache-2.0 | README |
| NanoDet whole-cat model (`server/models/object_detection_nanodet_2022nov.onnx`, OpenCV model zoo) | Apache-2.0 | README |

Non-goals: no "cat emotion" dataset is used anywhere. Every one we examined
was self-labelled with no expert review.
