#!/bin/bash
# Direct run script for Humanoid-v4 DREAMING (spiking world model)
# Runs indefinitely (10M step budget). Saves checkpoints every 50k steps.
# Auto-resumes from the latest checkpoint if one exists.
# Usage: nohup bash run_humanoid_dreaming_2M.sh > logs/humanoid_v4_dreaming.log 2>&1 &

set -e
echo "=========================================="
echo "  DREAMING RUN — Humanoid-v4 (open-ended)"
echo "  Started: $(date)"
echo "=========================================="

cd "$(dirname "$0")"
mkdir -p logs checkpoints runs

# Ensure ~/.local/bin is on PATH
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies (--break-system-packages needed on Ubuntu 24.04 / PEP 668)
pip3 install --user --break-system-packages -e . -r requirements.txt 2>&1 | tail -5

# Weights & Biases
export WANDB_API_KEY="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"
export WANDB_ENTITY="rohit-deepa"
export WANDB_PROJECT="Humanoid-v4"

# GPU check
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
            print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Run dreaming (world model ON) — auto-resumes from latest checkpoint
python3 scripts/train.py \
    --config configs/humanoid.yaml \
    --seed 2 \
    --resume \
    --wandb \
    --wandb-project Humanoid-v4

echo "Finished: $(date)"
