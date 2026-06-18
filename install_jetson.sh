#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv --system-site-packages .venv
fi

sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' .venv/pyvenv.cfg

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
python -m pip install numpy==1.26.4 cffi pycparser psutil 'polars==1.41.2' 'polars-runtime-32==1.41.2' 'ultralytics-thop>=2.0.18' --no-deps
python -m pip install 'onnx>=1.16.1,<2.0.0' 'onnxslim>=0.1.82' onnxruntime
python -m pip install 'ultralytics>=8.3,<9' --no-deps

python - <<'PY'
import cv2
import torch
from ultralytics import YOLO

print(f"cv2={cv2.__version__}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
print("ultralytics=ok")
PY
