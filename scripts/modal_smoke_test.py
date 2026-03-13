# /// script
# requires-python = ">=3.10"
# dependencies = ["modal>=1.0.0"]
# ///
"""
NanoEvolve — Modal GPU smoke test.

Usage:
    uv run scripts/modal_smoke_test.py
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "curl")
    # Install uv
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    # Copy nanochat pyproject.toml + lockfile, then uv sync with GPU extra
    .add_local_file(str(REPO_ROOT / "nanochat" / "pyproject.toml"), remote_path="/root/nanoevolve/nanochat/pyproject.toml", copy=True)
    .add_local_file(str(REPO_ROOT / "nanochat" / "uv.lock"), remote_path="/root/nanoevolve/nanochat/uv.lock", copy=True)
    .run_commands(
        "cd /root/nanoevolve/nanochat && /root/.local/bin/uv sync --extra gpu --frozen --no-install-project"
    )
    # pytest for adamopt tests
    .run_commands("/root/.local/bin/uv pip install pytest>=8.0.0 --python /root/nanoevolve/nanochat/.venv/bin/python")
    # Now add the actual source code (changes often, so last for cache)
    .add_local_dir(str(REPO_ROOT / "adamopt"), remote_path="/root/nanoevolve/adamopt", copy=True)
    .add_local_dir(str(REPO_ROOT / "nanochat" / "nanochat"), remote_path="/root/nanoevolve/nanochat/nanochat", copy=True)
    .add_local_dir(str(REPO_ROOT / "nanochat" / "scripts"), remote_path="/root/nanoevolve/nanochat/scripts", copy=True)
)

app = modal.App("nanoevolve", image=image)
data_vol = modal.Volume.from_name("nanoevolve-data", create_if_missing=True)

REMOTE_REPO = "/root/nanoevolve"
REMOTE_NANOCHAT = "/root/nanoevolve/nanochat"
REMOTE_DATA = "/data/nanochat"
PYTHON = "/root/nanoevolve/nanochat/.venv/bin/python"


@app.function(
    gpu="A100",
    volumes={"/data": data_vol},
    timeout=900,
)
def smoke_test():
    import os
    import subprocess

    os.chdir(REMOTE_REPO)
    os.environ["NANOCHAT_BASE_DIR"] = REMOTE_DATA
    # adamopt imports via CWD, nanochat via its venv
    os.environ["PYTHONPATH"] = REMOTE_REPO

    failed = False

    # 1/4: PyTorch + CUDA
    print("=" * 60)
    print("1/4  PyTorch + CUDA")
    print("=" * 60)
    result = subprocess.run(
        [PYTHON, "-c", """
import torch
print(f"  PyTorch:        {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version:   {torch.version.cuda}")
    print(f"  GPU:            {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU memory:     {mem_gb:.1f} GB")
    a = torch.randn(2048, 2048, device='cuda')
    c = a @ a.T
    print(f"  Matmul test:    PASS ({c.shape})")
else:
    raise RuntimeError("CUDA not available")
"""],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"  FAIL: {result.stderr.strip()[-300:]}")
        failed = True
    print()

    # 2/4: AdamOpt tests
    print("=" * 60)
    print("2/4  AdamOpt tests (18 expected)")
    print("=" * 60)
    result = subprocess.run(
        [PYTHON, "-m", "pytest", "adamopt/tests", "-q"],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()[-500:]}")
        failed = True
    print()

    # 3/4: Download data + train tokenizer
    print("=" * 60)
    print("3/4  Dataset download (3 shards) + tokenizer training")
    print("=" * 60)
    os.chdir(REMOTE_NANOCHAT)

    print("  Downloading 3 data shards...")
    result = subprocess.run(
        [PYTHON, "-m", "nanochat.dataset", "-n", "3"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"  Dataset download FAILED:")
        print(f"  STDOUT: {result.stdout.strip()[-300:]}")
        print(f"  STDERR: {result.stderr.strip()[-300:]}")
        failed = True
    else:
        print("  Dataset download: PASS")

    tok_path = Path(REMOTE_DATA) / "tokenizer" / "tokenizer.pkl"
    if tok_path.exists():
        print("  Tokenizer already trained (cached on volume), skipping.")
    else:
        print("  Training tokenizer (first run only, ~1-2 min)...")
        result = subprocess.run(
            [PYTHON, "-m", "scripts.tok_train"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"  Tokenizer training FAILED:")
            print(f"  STDOUT: {result.stdout.strip()[-300:]}")
            print(f"  STDERR: {result.stderr.strip()[-300:]}")
            failed = True
        else:
            print("  Tokenizer training: PASS")
    print()

    # 4/4: NanoChat training (20 steps)
    print("=" * 60)
    print("4/4  NanoChat training (tiny model, 20 steps, A100)")
    print("=" * 60)
    result = subprocess.run(
        [
            PYTHON, "-m", "scripts.base_train",
            "--run=dummy",
            "--depth=4",
            "--max-seq-len=512",
            "--device-batch-size=2",
            "--total-batch-size=1024",
            "--eval-tokens=512",
            "--core-metric-every=-1",
            "--num-iterations=20",
        ],
        capture_output=True, text=True, timeout=300,
    )
    lines = (result.stdout or "").strip().splitlines()
    for line in lines[-25:]:
        print(f"  {line}")
    if result.returncode != 0:
        print(f"  STDERR: {(result.stderr or '').strip()[-500:]}")
        failed = True
    print()

    data_vol.commit()

    print("=" * 60)
    if failed:
        print("SMOKE TEST FAILED")
    else:
        print("ALL CHECKS PASSED — A100 is ready for NanoEvolve.")
    print("=" * 60)


if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            smoke_test.remote()
