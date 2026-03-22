# Installation

## Requirements
- Python 3.10+
- CUDA-capable GPU (recommended)

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
   - **Jetson (aarch64)**: NVIDIA's Jetson PyTorch wheel for JetPack 6.x
3. Install lerobot from local source
4. Install all remaining dependencies
5. Verify the installation and confirm GPU is available

## Running

### With real Roomba
```bash
python momanip_navigation.py --model lerobot/smolvla_base --port /dev/ttyUSB0
```

### Dry run (no Roomba, real VLA inference)
```bash
python momanip_navigation.py --model lerobot/smolvla_base --dry-run --no-display
```

### Mock VLA (no model, for testing only)
```bash
python momanip_navigation.py --mock-vla --dry-run
```

## Notes
- SmolVLA model weights (~865 MB) are downloaded automatically from HuggingFace on first run
- Check available serial ports with: `ls /dev/ttyUSB*`
- Web streaming UI available at `http://localhost:5001` while running
