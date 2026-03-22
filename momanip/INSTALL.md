# Installation

## Requirements
- Python 3.10+
- CUDA-capable GPU (recommended)

## Steps

### 1. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install lerobot (local source)
```bash
pip install -e ./lerobot
```

### 3. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify installation
```bash
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print('OK')"
```

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
