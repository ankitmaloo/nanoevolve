#!/bin/bash
#SBATCH --job-name=enigma-s4ext-20k
#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/sharefs/user54/nanoe/runs/enigma_s4ext_20k_h60/slurm_%A_%a.out
#SBATCH --error=/mnt/sharefs/user54/nanoe/runs/enigma_s4ext_20k_h60/slurm_%A_%a.err

set -euo pipefail

NANOE_DIR="$HOME/nanoe"
RUN_DIR="$NANOE_DIR/runs/enigma_s4ext_20k_h60"

source "$NANOE_DIR/nanochat/.venv/bin/activate"
export PYTHONPATH="$NANOE_DIR:$NANOE_DIR/adamopt:${PYTHONPATH:-}"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1

MUTATIONS=(
    "none"
    "H60_solo"
)

MUTATION="${MUTATIONS[$SLURM_ARRAY_TASK_ID]}"
export ENIGMA_MUTATION="$MUTATION"

STEPS=20000
EVAL_EVERY=250
DEPTH=12
RESULTS_DIR="$RUN_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "=== ENIGMA Stage 4 Extension — 20k Follow-up: $MUTATION ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Steps: $STEPS, Depth: $DEPTH"

cd "$NANOE_DIR"

python -u enigma/run_stage5.py \
    --mutation "$MUTATION" \
    --output "$RESULTS_DIR/${MUTATION}_20k_real.json" \
    --steps "$STEPS" \
    --depth "$DEPTH" \
    --eval-every "$EVAL_EVERY"
