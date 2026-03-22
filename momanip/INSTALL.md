# Installation

## Requirements
- Python 3.10+
- JetPack 6.x (on Jetson) or CUDA 12.x (on x86)
- Internet access for first run (downloads SmolVLA weights ~865 MB)

## Steps

Run the install script — it auto-detects x86 vs Jetson (ARM64) and installs the correct PyTorch:

```bash
chmod +x install.sh
./install.sh
```

This will:
1. Create a `.venv` virtual environment
2. Install the platform-appropriate PyTorch:
   - **x86_64**: `torch==2.7.1` from PyPI with CUDA
   - **Jetson (aarch64)**: NVIDIA's Jetson PyTorch wheel for JetPack 6.x + torchvision built from source
3. Install lerobot from local source
4. Install all remaining dependencies
5. Verify the installation and confirm GPU is available

## Known Issues & Fixes (Jetson Orin Nano)

### 1. pip not found in venv
```
/bin/bash: .venv/bin/pip: cannot execute: required file not found
```
**Fix:** Install missing system packages then recreate venv:
```bash
sudo apt install python3-pip python3-venv
python3 -m venv --clear .venv
```

### 2. libcusparseLt not found
```
ImportError: libcusparseLt.so.0: cannot open shared object file
```
**Fix:** Find the library and add it to the path:
```bash
find /usr/local/cuda* -name "libcusparseLt.so*" 2>/dev/null
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
If not found, install full JetPack SDK:
```bash
sudo apt install nvidia-jetpack
```

### 3. torchvision 404 error
```
ERROR: HTTP error 404 while getting https://.../torchvision-...whl
```
**Fix:** NVIDIA does not host torchvision — the install script builds it from source automatically (takes 10–20 min).

### 4. NumPy 2.x incompatibility
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```
**Fix:**
```bash
pip install "numpy<2"
```

### 5. CUDA OOM when loading model
```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp
```
**Fix:** Already applied in `lerobot/src/lerobot/policies/pretrained.py` — weights load to CPU first then move to GPU.

### 6. Permission denied on serial port
```
[Errno 13] Permission denied: '/dev/ttyUSB0'
```
**Fix:**
```bash
sudo usermod -a -G dialout $USER
# Then log out and log back in
```

### 7. Roomba not found / wrong port
```
Warning: Could not connect to Roomba
```
**Fix:** Check which port the Roomba is on:
```bash
ls /dev/ttyUSB* /dev/ttyACM*
```
Then pass the correct port: `--port /dev/ttyACM0`

## Running

### With real Roomba
```bash
source .venv/bin/activate
python momanip_navigation.py --model lerobot/smolvla_base --port /dev/ttyUSB0
```

### Dry run (no Roomba, real VLA inference)
```bash
python momanip_navigation.py --model lerobot/smolvla_base --dry-run --no-display
```

### Mock VLA (no model, for quick testing)
```bash
python momanip_navigation.py --mock-vla --dry-run
```

## Notes
- SmolVLA weights (~865 MB) download automatically from HuggingFace on first run
- Web streaming UI available at `http://localhost:5001` while running
- The model uses bfloat16 on CUDA — Jetson Orin Nano's Ampere GPU supports this
