#!/usr/bin/env bash
# init.sh — Reproducible environment setup for Singapore Weather Diffusion POC
# Run this once (or after a fresh clone) to create the conda environment.
# Safe to re-run: skips steps that are already complete.
set -euo pipefail

ENV_NAME="sg-weather"
PYTHON_VERSION="3.11"

echo "=== Singapore Weather Diffusion — Environment Init ==="

# ── 1. Check prerequisites ────────────────────────────────────────────────────
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found. Install Miniconda first."; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "WARNING: nvidia-smi not found. GPU may not be available."; }

# ── 2. Create / update conda environment ─────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "[SKIP] Conda env '${ENV_NAME}' already exists."
else
  echo "[CREATE] Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

# ── 3. Install packages ────────────────────────────────────────────────────────
echo "[INSTALL] Installing requirements into '${ENV_NAME}'..."
conda run -n "${ENV_NAME}" pip install --upgrade pip
conda run -n "${ENV_NAME}" pip install -r requirements.txt

# ── 4. Install PyTorch with CUDA 12.x (NVIDIA recommended) ────────────────────
# Overrides any CPU-only torch from requirements.txt
echo "[INSTALL] Installing PyTorch 2.x with CUDA 12.1..."
conda run -n "${ENV_NAME}" pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

# ── 5. Install PhysicsNeMo and Earth2Studio from source/NVIDIA ────────────────
echo "[INSTALL] Installing PhysicsNeMo + Earth2Studio..."
conda run -n "${ENV_NAME}" pip install \
  "physicsnemo[all]" \
  "earth2studio[all]"

# ── 6. Configure CDS API (ERA5) ────────────────────────────────────────────────
if [ -f "$HOME/.cdsapirc" ]; then
  echo "[SKIP] ~/.cdsapirc already exists."
else
  echo "[NOTICE] ~/.cdsapirc not found."
  echo "  To download ERA5 data, create ~/.cdsapirc with:"
  echo "    url: https://cds.climate.copernicus.eu/api"
  echo "    key: <your-personal-access-token>"
  echo "  Get your key at: https://cds.climate.copernicus.eu"
fi

# ── 7. Set up .env from template ──────────────────────────────────────────────
if [ -f ".env" ]; then
  echo "[SKIP] .env already exists."
else
  cp .env.example .env
  echo "[CREATED] .env from template. Fill in your API keys before running scripts."
fi

# ── 8. Create checkpoint stage flag ───────────────────────────────────────────
mkdir -p checkpoints
touch checkpoints/stage0_complete.flag

# ── 9. Verification ───────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
conda run -n "${ENV_NAME}" python -c "
import torch
print(f'  PyTorch version : {torch.__version__}')
print(f'  CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU             : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM            : {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
"

conda run -n "${ENV_NAME}" python -c "
import earth2studio; print(f'  earth2studio    : OK ({earth2studio.__version__})')
" 2>/dev/null || echo "  earth2studio    : install may need manual check"

conda run -n "${ENV_NAME}" python -c "
import physicsnemo; print(f'  physicsnemo     : OK')
" 2>/dev/null || echo "  physicsnemo     : install may need manual check"

echo ""
echo "=== Setup complete ==="
echo "Activate with:  conda activate ${ENV_NAME}"
echo "Next step:      python scripts/download_era5.py"
