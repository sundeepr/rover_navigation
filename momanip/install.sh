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

    # Locate libcusparseLt — required by the Jetson PyTorch wheel
    CUSPARSELT=$(find /usr/local/cuda* /usr/lib/aarch64-linux-gnu -name "libcusparseLt.so*" 2>/dev/null | head -1)
    if [ -n "$CUSPARSELT" ]; then
        CUSPARSELT_DIR=$(dirname "$CUSPARSELT")
        echo "Found libcusparseLt at $CUSPARSELT_DIR"
    else
        echo "libcusparseLt not found — installing from CUDA toolkit repo..."
        # Try the versioned package name used in CUDA 12.x on Jetson
        sudo apt-get install -y cuda-cusparselt-12-6 || \
        sudo apt-get install -y cuda-cusparselt-12-2 || \
        echo "WARNING: Could not install libcusparseLt — torch may fail to import"
        CUSPARSELT_DIR=/usr/local/cuda/lib64
    fi

    # Ensure CUDA libraries are on the path
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${CUSPARSELT_DIR}:${LD_LIBRARY_PATH}
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${CUSPARSELT_DIR}:\$LD_LIBRARY_PATH" >> ~/.bashrc

    # Jetson PyTorch wheel requires NumPy < 2
    pip install "numpy<2"

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
