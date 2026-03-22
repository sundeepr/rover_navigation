#!/bin/bash
# MoManip Navigation - Debug Script
# Run this and share the output when reporting issues.

echo "========================================"
echo " MoManip Navigation - Debug Report"
echo " $(date)"
echo "========================================"

echo ""
echo "--- System ---"
uname -a
echo "Arch: $(uname -m)"

echo ""
echo "--- Python ---"
python3 --version 2>&1

echo ""
echo "--- Virtual environment ---"
if [ -d ".venv" ]; then
    echo "venv exists"
    ls .venv/bin/pip .venv/bin/python 2>&1
else
    echo "ERROR: .venv not found — run ./install.sh first"
fi

echo ""
echo "--- Activate venv and check packages ---"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "torch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
    echo "numpy: $(python -c 'import numpy; print(numpy.__version__)' 2>&1)"
    echo "transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>&1)"
    echo "lerobot: $(python -c 'import lerobot; print(lerobot.__version__)' 2>&1)"
    echo "flask: $(python -c 'import flask; print(flask.__version__)' 2>&1)"
    echo "scipy: $(python -c 'import scipy; print(scipy.__version__)' 2>&1)"
    echo "cv2: $(python -c 'import cv2; print(cv2.__version__)' 2>&1)"
    echo "serial: $(python -c 'import serial; print(serial.__version__)' 2>&1)"
else
    echo "Cannot activate venv"
fi

echo ""
echo "--- CUDA / GPU ---"
echo "CUDA libraries:"
find /usr/local/cuda* /usr/lib/aarch64-linux-gnu -name "libcuda*" -o -name "libcusparseLt*" 2>/dev/null | head -10
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB')
" 2>&1
fi

echo ""
echo "--- SmolVLA model cache ---"
ls ~/.cache/huggingface/hub/ 2>/dev/null | grep smolvla || echo "SmolVLA not cached — will download on first run"

echo ""
echo "--- Serial ports (Roomba) ---"
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "No serial ports found"
echo "Current user: $(whoami)"
echo "Groups: $(groups)"
if groups | grep -q dialout; then
    echo "dialout group: YES (serial access OK)"
else
    echo "dialout group: NO — fix with: sudo usermod -a -G dialout \$USER then re-login"
fi

echo ""
echo "--- Camera ---"
ls /dev/video* 2>/dev/null || echo "No camera devices found"

echo ""
echo "--- lerobot SmolVLA import test ---"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
print('SmolVLAPolicy import: OK')
" 2>&1
fi

echo ""
echo "========================================"
echo " End of debug report"
echo "========================================"
