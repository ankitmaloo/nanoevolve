from __future__ import annotations

import shlex
import sys
from pathlib import Path


def local_nanochat_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "nanochat"
    if not (root / "nanochat" / "optim.py").exists():
        raise FileNotFoundError(f"Real nanochat clone not found at {root}")
    return root


def render_print_command(diff_output: str) -> str:
    python_bin = Path(sys.executable).resolve()
    code = f"print({diff_output!r})"
    return f"{shlex.quote(str(python_bin))} -c {shlex.quote(code)}"
