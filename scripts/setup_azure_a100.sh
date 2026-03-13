#!/usr/bin/env bash
# =============================================================================
# NanoEvolve — Azure A100 setup script
#
# Run this on a fresh Azure A100 VM after cloning the repo.
# Usage:
#   chmod +x scripts/setup_azure_a100.sh
#   ./scripts/setup_azure_a100.sh
#
# Expects: Ubuntu 22.04+, NVIDIA A100, CUDA drivers pre-installed (Azure NC/ND VMs)
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${REPO_DIR}/.venv"

echo "============================================"
echo "NanoEvolve Azure A100 Setup"
echo "============================================"
echo "Repo:  ${REPO_DIR}"
echo "Venv:  ${VENV_DIR}"
echo ""

# -------------------------------------------------------------------
# 0. System checks
# -------------------------------------------------------------------
echo "[0/7] System checks..."

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    echo "  On Azure NC/ND VMs, drivers are usually pre-installed."
    echo "  If not: sudo apt install -y nvidia-driver-535"
    exit 1
fi

echo "  GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Check CUDA version
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "  Driver version: ${CUDA_VERSION}"

# -------------------------------------------------------------------
# 1. System dependencies
# -------------------------------------------------------------------
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.12 python3.12-venv python3.12-dev \
    git curl wget htop tmux \
    build-essential 2>/dev/null || {
    # Fallback: if python3.12 not available, try python3.11 or python3.10
    echo "  python3.12 not available, trying python3.11..."
    sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev 2>/dev/null || {
        echo "  python3.11 not available, trying python3.10..."
        sudo apt-get install -y -qq python3.10 python3.10-venv python3.10-dev
    }
}

# Find the best available python
PYTHON=""
for ver in python3.12 python3.11 python3.10 python3; do
    if command -v "$ver" &>/dev/null; then
        PYTHON="$ver"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: No Python 3.10+ found"
    exit 1
fi

echo "  Using: $PYTHON ($($PYTHON --version))"

# -------------------------------------------------------------------
# 2. Create virtualenv
# -------------------------------------------------------------------
echo "[2/7] Creating virtualenv at ${VENV_DIR}..."
if [ -d "$VENV_DIR" ]; then
    echo "  Venv already exists, reusing."
else
    $PYTHON -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel -q

# -------------------------------------------------------------------
# 3. Install PyTorch with CUDA 12.8
# -------------------------------------------------------------------
echo "[3/7] Installing PyTorch 2.9.1 + CUDA 12.8..."
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 -q

# -------------------------------------------------------------------
# 4. Install nanochat
# -------------------------------------------------------------------
echo "[4/7] Installing nanochat (editable)..."
cd "$REPO_DIR"
pip install -e nanochat/ -q

# -------------------------------------------------------------------
# 5. Install adamopt
# -------------------------------------------------------------------
echo "[5/7] Installing adamopt (editable)..."
pip install -e adamopt/ -q

# -------------------------------------------------------------------
# 6. Download training data (first 10 shards for smoke test)
# -------------------------------------------------------------------
echo "[6/7] Downloading training data (10 shards for smoke test)..."
cd "$REPO_DIR/nanochat"
python -m nanochat.dataset -n 10 || {
    echo "  WARNING: Dataset download failed. You can retry manually:"
    echo "    cd nanochat && python -m nanochat.dataset -n 10"
}
cd "$REPO_DIR"

# -------------------------------------------------------------------
# 7. Verification
# -------------------------------------------------------------------
echo "[7/7] Running verification..."
echo ""

# 7a. Check torch + CUDA
echo "--- PyTorch + CUDA ---"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    # Quick matmul test
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    c = a @ b
    print(f'  Matmul test: OK ({c.shape})')
else:
    print('  WARNING: CUDA not available! Check driver/torch compatibility.')
"
echo ""

# 7b. Check adamopt tests
echo "--- AdamOpt tests ---"
cd "$REPO_DIR"
python -m pytest adamopt/tests -q 2>&1 | tail -5
echo ""

# 7c. NanoChat smoke test (tiny model, ~20 steps, single GPU)
echo "--- NanoChat smoke test (tiny model, 20 steps) ---"
cd "$REPO_DIR/nanochat"
python -m scripts.base_train \
    --run=dummy \
    --depth=4 \
    --max-seq-len=512 \
    --device-batch-size=2 \
    --total-batch-size=512 \
    --eval-tokens=512 \
    --core-metric-every=-1 \
    --num-iterations=20 2>&1 | tail -20
echo ""

echo "============================================"
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To run the full test suite:"
echo "  python -m pytest adamopt/tests -q"
echo ""
echo "To run a NanoChat training smoke test:"
echo "  cd nanochat && python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=2 --total-batch-size=512 --eval-tokens=512 --core-metric-every=-1 --num-iterations=20"
echo ""
echo "To download more data shards (170 recommended for real runs):"
echo "  cd nanochat && python -m nanochat.dataset -n 170"
echo "============================================"
