#!/bin/zsh
# One-shot, unattended cat-face detector training pipeline.
#   verify dataset zips -> prepare dataset -> train (Apple GPU) -> evaluate -> export ONNX
# Logs to training/runs/pipeline.log. Keeps the Mac awake while it runs.
# Scheduled by ~/Library/LaunchAgents/com.cattalktome.train-catface.plist (removed at the end).
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin
T=/Users/david.siglin/Git/emma/cattalktome/training
LOG=$T/runs/pipeline.log
mkdir -p $T/runs
exec >>$LOG 2>&1
echo "=== pipeline start $(date) ==="

run() {
  echo "--- $1 ($(date +%H:%M:%S))"; shift
  caffeinate -is "$@"
  rc=$?; echo "--- exit $rc"; return $rc
}

cd $T
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=$T/YOLOX

run "verify dataset zips" python fetch_dataset.py || { echo "ABORT: dataset zips not good"; exit 1; }

if [ ! -f data/catface/annotations/val_rot.json ]; then
  run "prepare dataset" python prepare_catface_dataset.py || { echo "ABORT: dataset preparation failed"; exit 1; }
else
  echo "--- dataset already prepared"
fi

run "train 30 epochs" python train_catface.py --epochs 30 --batch 32 --workers 4 || echo "WARN: training exited non-zero (best.pth may still exist)"

if [ -f runs/catface/best.pth ]; then
  run "evaluate val / val_rot / test" python train_catface.py --eval-only runs/catface/best.pth
  run "export ONNX" python train_catface.py --export runs/catface/best.pth
else
  echo "ABORT: no best.pth produced"
fi

echo "=== pipeline end $(date) ==="
# one-shot: remove the scheduler entry
launchctl bootout gui/$(id -u)/com.cattalktome.train-catface 2>/dev/null
rm -f ~/Library/LaunchAgents/com.cattalktome.train-catface.plist
echo "PIPELINE_DONE"
