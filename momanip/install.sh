#!/bin/bash
set -e

ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install lerobot from local source first
pip install -e ./lerobot

if [ "$ARCH" = "aarch64" ]; then
    echo "Jetson (ARM64) detected — installing Jetson PyTorch..."

    # JetPack 6.x — PyTorch 2.5.0 for Jetson Orin
    pip install --no-cache \
        https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

    # torchvision is not hosted by NVIDIA — build from source (matches torch 2.5 / torchvision 0.20)
    echo "Building torchvision from source (this may take 10-20 minutes)..."
    sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
    pip install --no-cache Pillow
    git clone --branch v0.20.0 https://github.com/pytorch/vision.git /tmp/torchvision
    cd /tmp/torchvision && python setup.py install && cd -
    rm -rf /tmp/torchvision
else
    echo "x86_64 detected — installing standard PyTorch with CUDA..."
    pip install torch==2.7.1 torchvision==0.22.1
fi

# Install remaining dependencies
pip install -r requirements.txt

echo ""
echo "Installation complete. Verifying..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print('SmolVLAPolicy: OK')
"
