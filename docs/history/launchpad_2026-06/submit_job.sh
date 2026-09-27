#!/bin/bash
# submit_job.sh — CorrDiff Singapore fine-tuning job submission for NVIDIA LaunchPad
#
# Fill in the LAUNCHPAD_* variables once access is granted, then run this script
# from the LaunchPad terminal to start the fine-tuning job.
#
# Usage:
#   bash scripts/launchpad/submit_job.sh

set -euo pipefail

# ── LaunchPad environment (fill in after access is granted) ─────────────────
LAUNCHPAD_PROJECT_DIR="${LAUNCHPAD_PROJECT_DIR:-/workspace/sg-weather}"
NGC_API_KEY="${NGC_API_KEY:?Set NGC_API_KEY in your environment}"
LAUNCHPAD_CONTAINER="${LAUNCHPAD_CONTAINER:-nvcr.io/nvidia/physicsnemo/physicsnemo:24.09}"
NUM_GPUS="${NUM_GPUS:-4}"
# ────────────────────────────────────────────────────────────────────────────

echo "=== Singapore CorrDiff Fine-tuning Job ==="
echo "  Project dir : $LAUNCHPAD_PROJECT_DIR"
echo "  Container   : $LAUNCHPAD_CONTAINER"
echo "  GPUs        : $NUM_GPUS"
echo ""

# Sync project files to LaunchPad workspace
echo "[1/4] Syncing project files..."
rsync -av --exclude='.git' --exclude='data/raw/radar/' --exclude='checkpoints/nowcaster/' \
    ./ "$LAUNCHPAD_PROJECT_DIR/"

# Pull pre-trained CorrDiff weights from NGC
echo "[2/4] Pulling CorrDiff pre-trained weights from NGC..."
mkdir -p "$LAUNCHPAD_PROJECT_DIR/checkpoints/corrdiff_pretrained"
ngc registry model download-version \
    "nvidia/modulus/corrdiff_inference_package:1" \
    --dest "$LAUNCHPAD_PROJECT_DIR/checkpoints/corrdiff_pretrained" \
    || echo "  WARNING: NGC pull failed — check NGC_API_KEY and model availability"

# Upload Singapore ERA5 + radar data (assumes data/ is already synced above,
# but radar.zarr can be large — consider NGC dataset upload for large archives)
echo "[3/4] Verifying data availability..."
python "$LAUNCHPAD_PROJECT_DIR/scripts/validate_era5.py" \
    && echo "  ERA5: OK" || echo "  ERA5: MISSING — upload data/raw/era5/ to LaunchPad"
ls "$LAUNCHPAD_PROJECT_DIR/data/processed/radar.zarr" > /dev/null 2>&1 \
    && echo "  radar.zarr: OK" || echo "  radar.zarr: MISSING — upload data/processed/radar.zarr"

# Launch fine-tuning container
echo "[4/4] Launching fine-tuning job..."
docker run --rm --gpus all \
    -v "$LAUNCHPAD_PROJECT_DIR:/workspace" \
    -e NGC_API_KEY="$NGC_API_KEY" \
    -w /workspace \
    "$LAUNCHPAD_CONTAINER" \
    torchrun --nproc_per_node="$NUM_GPUS" \
        scripts/launchpad/finetune_corrdiff_sg.py \
        --era5-zarr data/raw/era5/singapore_2022.zarr data/raw/era5/singapore_2023.zarr \
        --radar-zarr data/processed/radar.zarr \
        --pretrained-ckpt checkpoints/corrdiff_pretrained/ \
        --output-dir checkpoints/corrdiff_singapore/ \
        --batch-size 8 \
        --max-steps 50000 \
        --lr 5e-5

echo ""
echo "Job submitted. Monitor with: docker logs <container_id>"
echo "Checkpoint output: checkpoints/corrdiff_singapore/"
