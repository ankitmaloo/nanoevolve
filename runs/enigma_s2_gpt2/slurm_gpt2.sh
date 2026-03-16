#!/bin/bash
#SBATCH --job-name=enigma-gpt2
#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/sharefs/user54/nanoe/runs/enigma_s2_gpt2/slurm_%j.out
#SBATCH --error=/mnt/sharefs/user54/nanoe/runs/enigma_s2_gpt2/slurm_%j.err

set -euo pipefail

NANOE_DIR="$HOME/nanoe"
RUN_DIR="$NANOE_DIR/runs/enigma_s2_gpt2"

source "$NANOE_DIR/nanochat/.venv/bin/activate"
export PYTHONPATH="$NANOE_DIR:$NANOE_DIR/adamopt:${PYTHONPATH:-}"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

echo "=== ENIGMA Stage 2 — GPT-2 Scale (depth=12, 100 steps) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python: $(python --version)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo "Specs: $(ls $RUN_DIR/specs/*.json | wc -l) files"
echo "========================================================"

python -m enigma.run_stage2_remote \
    --specs-dir "$RUN_DIR/specs" \
    --results-dir "$RUN_DIR/results" \
    --steps 100 \
    --eval-every 20 \
    --depth 12 \
    --seed 42

echo ""
echo "=== Done. Results ==="
ls -la "$RUN_DIR/results/"
