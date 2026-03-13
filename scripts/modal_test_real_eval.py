# /// script
# requires-python = ">=3.10"
# dependencies = ["modal>=1.0.0"]
# ///
"""
NanoEvolve — Modal A100 test for real_eval.py.

Thin wrapper: all test logic lives in adamopt.optim_search.test_real_eval_gpu.
Replace this file with any other container runner (Azure, bare metal, etc.)
and just call run_all_tests().

Usage:
    uv run scripts/modal_test_real_eval.py
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    # Install deps from nanochat lockfile (cached layer)
    .add_local_file(str(REPO_ROOT / "nanochat" / "pyproject.toml"), remote_path="/root/nanoevolve/nanochat/pyproject.toml", copy=True)
    .add_local_file(str(REPO_ROOT / "nanochat" / "uv.lock"), remote_path="/root/nanoevolve/nanochat/uv.lock", copy=True)
    .run_commands(
        "cd /root/nanoevolve/nanochat && /root/.local/bin/uv sync --extra gpu --frozen --no-install-project"
    )
    # Source code (changes often, so last for cache)
    .add_local_dir(str(REPO_ROOT / "adamopt"), remote_path="/root/nanoevolve/adamopt", copy=True)
    .add_local_dir(str(REPO_ROOT / "nanochat" / "nanochat"), remote_path="/root/nanoevolve/nanochat/nanochat", copy=True)
    .add_local_dir(str(REPO_ROOT / "nanochat" / "scripts"), remote_path="/root/nanoevolve/nanochat/scripts", copy=True)
)

app = modal.App("nanoevolve-real-eval-test", image=image)
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
def test_real_eval():
    import os
    import subprocess
    import sys

    os.chdir(REMOTE_REPO)
    os.environ["NANOCHAT_BASE_DIR"] = REMOTE_DATA
    os.environ["PYTHONPATH"] = REMOTE_REPO

    # ---- Prerequisites: data + tokenizer (cached on volume) ----
    os.chdir(REMOTE_NANOCHAT)
    tok_path = Path(REMOTE_DATA) / "tokenizer" / "tokenizer.pkl"
    if not tok_path.exists():
        print("Downloading data shards...")
        subprocess.run([PYTHON, "-m", "nanochat.dataset", "-n", "3"],
                       capture_output=True, text=True, timeout=300, check=True)
        print("Training tokenizer...")
        subprocess.run([PYTHON, "-m", "scripts.tok_train"],
                       capture_output=True, text=True, timeout=300, check=True)
    else:
        print("Tokenizer cached on volume, skipping setup.")

    data_vol.commit()
    os.chdir(REMOTE_REPO)

    # ---- Add paths so adamopt + nanochat are importable in-process ----
    # REMOTE_REPO for adamopt.*
    # REMOTE_NANOCHAT for nanochat.* (source package)
    # venv_site for torch, etc.
    venv_site = "/root/nanoevolve/nanochat/.venv/lib/python3.12/site-packages"
    for p in [REMOTE_REPO, REMOTE_NANOCHAT, venv_site]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # ---- Run the container-agnostic test suite ----
    from adamopt.optim_search.test_real_eval_gpu import run_all_tests

    passed, report = run_all_tests(nanochat_base_dir=REMOTE_DATA)
    print(report)

    if not passed:
        raise RuntimeError("Real eval tests failed")


if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            test_real_eval.remote()
