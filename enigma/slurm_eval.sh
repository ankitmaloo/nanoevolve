#!/bin/bash
#SBATCH --job-name=enigma-eval
#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/sharefs/user54/nanoe/runs/enigma_run/slurm_%j.out
#SBATCH --error=/mnt/sharefs/user54/nanoe/runs/enigma_run/slurm_%j.err

set -euo pipefail

NANOE_DIR="$HOME/nanoe"
RUN_DIR="$NANOE_DIR/runs/enigma_run"
SPECS_DIR="$RUN_DIR/specs"
RESULTS_DIR="$RUN_DIR/results"

mkdir -p "$RESULTS_DIR"

# Activate the nanochat venv (has torch+cuda)
source "$NANOE_DIR/nanochat/.venv/bin/activate"

# adamopt is not a package — add it to PYTHONPATH
export PYTHONPATH="$NANOE_DIR:$NANOE_DIR/adamopt:${PYTHONPATH:-}"
# Use default cache dir for nanochat data (not /data/nanochat)
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

echo "=== ENIGMA Real NanoChat Evaluation ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Python: $(python --version)"
echo "Torch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo "Specs: $(ls $SPECS_DIR/*.json | wc -l) files"
echo "======================================="

# Run the remote eval script
python "$RUN_DIR/remote_eval.py" \
    --specs-dir "$SPECS_DIR" \
    --results-dir "$RESULTS_DIR" \
    --steps 20 \
    --eval-every 10 \
    --depth 4 \
    --seed 42

echo ""
echo "=== Done. Results in $RESULTS_DIR ==="
ls -la "$RESULTS_DIR"
