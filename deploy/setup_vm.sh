#!/usr/bin/env bash
# Установка окружения на VM команды (Tesla T4). Запускается НА VM:
#   bash ~/shlif/deploy/setup_vm.sh
set -euo pipefail
cd ~/shlif

python3 -m venv venv 2>/dev/null || true
./venv/bin/pip install -q --upgrade pip

# GPU-torch (T4, CUDA) — с обычного PyPI
./venv/bin/pip install -q torch torchvision

./venv/bin/pip install -q \
    numpy pandas matplotlib pillow tqdm scikit-learn \
    opencv-python-headless scikit-image scipy lightgbm \
    streamlit plotly reportlab "pyvips[binary]" \
    segmentation-models-pytorch albumentations

echo "=== versions ==="
./venv/bin/python - <<'EOF'
import torch, cv2, pyvips
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
print("cv2", cv2.__version__, "| vips", pyvips.version(0), pyvips.version(1))
EOF

# данные кейса (если загружен zip)
if [ -f ~/shlif/data/*.zip ] 2>/dev/null || ls ~/shlif/data/*.zip >/dev/null 2>&1; then
  if [ ! -d ~/shlif/data/raw/shlif ]; then
    mkdir -p ~/shlif/data/raw
    unzip -q -o ~/shlif/data/*.zip -d ~/shlif/data/raw/
    mv "$(find ~/shlif/data/raw -maxdepth 1 -type d -name 'Задача*' | head -1)" ~/shlif/data/raw/shlif
  fi
fi
echo "setup done"
