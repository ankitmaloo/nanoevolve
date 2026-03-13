# /// script
# requires-python = ">=3.10"
# dependencies = ["modal>=1.0.0"]
# ///
"""
NanoEvolve — Validate a single candidate optimizer spec on real NanoChat.

Pre-flight check before a mutation enters the real evaluation queue:
1. Parse & validate the spec locally (syntax, constraint checks)
2. Run 20 steps of real NanoChat training on A100
3. Run baseline for comparison
4. Report: PASS/FAIL + telemetry

Usage:
    # Validate a spec from a JSON file:
    uv run scripts/modal_validate_spec.py --spec path/to/candidate.json

    # Validate from stdin (pipe from LLM output):
    echo '{"name": "my_mutation", ...}' | uv run scripts/modal_validate_spec.py --spec -

    # Validate one of the built-in variants:
    uv run scripts/modal_validate_spec.py --builtin stateful_annealing
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:$PATH"})
    .add_local_file(str(REPO_ROOT / "nanochat" / "pyproject.toml"), remote_path="/root/nanoevolve/nanochat/pyproject.toml", copy=True)
    .add_local_file(str(REPO_ROOT / "nanochat" / "uv.lock"), remote_path="/root/nanoevolve/nanochat/uv.lock", copy=True)
    .run_commands(
        "cd /root/nanoevolve/nanochat && /root/.local/bin/uv sync --extra gpu --frozen --no-install-project"
    )
    .add_local_dir(str(REPO_ROOT / "adamopt"), remote_path="/root/nanoevolve/adamopt", copy=True)
    .add_local_dir(str(REPO_ROOT / "nanochat" / "nanochat"), remote_path="/root/nanoevolve/nanochat/nanochat", copy=True)
    .add_local_dir(str(REPO_ROOT / "nanochat" / "scripts"), remote_path="/root/nanoevolve/nanochat/scripts", copy=True)
)

app = modal.App("nanoevolve-validate", image=image)
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
def validate_on_gpu(spec_dict: dict, baseline_dict: dict, steps: int, seed: int) -> dict:
    """Run candidate + baseline on GPU, return both outcomes as plain dicts."""
    import os
    import subprocess
    import sys as _sys

    os.chdir(REMOTE_REPO)
    os.environ["NANOCHAT_BASE_DIR"] = REMOTE_DATA
    os.environ["PYTHONPATH"] = REMOTE_REPO

    # Ensure data is ready
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

    venv_site = "/root/nanoevolve/nanochat/.venv/lib/python3.12/site-packages"
    for p in [REMOTE_REPO, REMOTE_NANOCHAT, venv_site]:
        if p not in _sys.path:
            _sys.path.insert(0, p)

    from adamopt.optim_search.real_eval import RealEvalConfig, evaluate_real_nanochat
    from adamopt.optim_search.spec import MatrixOptimizerSpec

    config = RealEvalConfig(
        steps=steps,
        eval_every=max(1, steps // 2),
        depth=4,
        max_seq_len=512,
        device_batch_size=2,
        total_batch_size=1024,
        device="cuda",
        nanochat_base_dir=REMOTE_DATA,
        seed=seed,
    )

    # Run candidate
    print(f"\n{'='*60}")
    print(f"Running CANDIDATE: {spec_dict.get('name', '?')}")
    print(f"{'='*60}")
    candidate_spec = MatrixOptimizerSpec.from_dict(spec_dict)
    candidate_outcome = evaluate_real_nanochat(candidate_spec, config, candidate_id="candidate")

    # Run baseline
    print(f"\n{'='*60}")
    print(f"Running BASELINE: {baseline_dict.get('name', '?')}")
    print(f"{'='*60}")
    baseline_spec = MatrixOptimizerSpec.from_dict(baseline_dict)
    baseline_outcome = evaluate_real_nanochat(baseline_spec, config, candidate_id="baseline")

    return {
        "candidate": asdict(candidate_outcome),
        "baseline": asdict(baseline_outcome),
    }


# ---------------------------------------------------------------------------
# Local-only helpers (no torch needed)
# ---------------------------------------------------------------------------

def _load_spec_module():
    """Import spec.py without triggering torch-dependent __init__.py."""
    import importlib.util
    import types

    for pkg_name in ("adamopt", "adamopt.optim_search"):
        if pkg_name not in sys.modules:
            sys.modules[pkg_name] = types.ModuleType(pkg_name)

    def _import(name: str, path: str):
        ms = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(ms)
        mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        sys.modules[name] = mod
        ms.loader.exec_module(mod)
        return mod

    return _import(
        "adamopt.optim_search.spec",
        str(REPO_ROOT / "adamopt" / "optim_search" / "spec.py"),
    )


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate a single candidate optimizer spec on real NanoChat training"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec", type=str, help="Path to candidate spec JSON (use '-' for stdin)")
    group.add_argument("--builtin", type=str, choices=["baseline", "trust_ratio", "stateful_annealing"],
                       help="Use a built-in spec variant")
    parser.add_argument("--steps", type=int, default=20, help="Training steps (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Path to save results JSON")
    return parser.parse_args()


def _local_validate(spec_mod, spec_dict: dict) -> tuple[bool, str]:
    """Phase 1: Local syntax & constraint validation. No GPU needed."""
    try:
        spec = spec_mod.MatrixOptimizerSpec.from_dict(spec_dict)
        spec.validate()
        return True, f"OK: spec '{spec.name}' parsed and validated"
    except Exception as e:
        return False, f"FAIL: {type(e).__name__}: {e}"


def _print_telemetry_summary(label: str, outcome: dict):
    """Print a concise telemetry summary for one outcome."""
    m = outcome.get("metrics")
    if not outcome.get("valid"):
        print(f"  {label}: INVALID — {outcome.get('failure_type', '?')}")
        return

    if m is None:
        print(f"  {label}: valid but no metrics")
        return

    print(f"  {label}:")
    print(f"    final_val_bpb:    {m['final_validation_bpb']:.4f}")
    print(f"    best_val_bpb:     {m['best_validation_bpb']:.4f}")
    print(f"    mean_step_ms:     {m['mean_step_time_ms']:.1f}")
    print(f"    tokens/sec:       {m['tokens_per_sec']:.0f}")
    print(f"    nan/inf:          {m['nan_failures']}/{m['inf_failures']}")
    print(f"    grad_spikes:      {m['grad_norm_spikes']}")
    print(f"    stability:        {m['stability_penalty']}")
    print(f"    memory_bytes:     {m['memory_overhead_bytes']}")

    telem = outcome.get("telemetry", [])
    if telem:
        losses = [s["loss"] for s in telem]
        gates = [s["gate_value"] for s in telem]
        print(f"    loss:             {losses[0]:.4f} → {losses[-1]:.4f}")
        print(f"    gate range:       [{min(gates):.4f}, {max(gates):.4f}]")


def _judge_validity(candidate: dict, baseline: dict) -> tuple[bool, list[str]]:
    """Phase 3: Judge if the candidate is valid enough to enter the queue."""
    notes = []

    # Must be valid (no crashes)
    if not candidate.get("valid"):
        return False, [f"CRASHED: {candidate.get('failure_type', '?')}"]

    cm = candidate.get("metrics")
    bm = baseline.get("metrics")
    if cm is None:
        return False, ["No metrics produced"]
    if bm is None:
        notes.append("WARNING: baseline produced no metrics, cannot compare")
        return True, notes

    # Must not have NaN/Inf
    if cm["nan_failures"] > 0 or cm["inf_failures"] > 0:
        return False, [f"NUMERICAL FAILURE: {cm['nan_failures']} NaN, {cm['inf_failures']} Inf"]

    # Loss must be finite and decreasing
    telem = candidate.get("telemetry", [])
    if telem:
        first_loss = telem[0]["loss"]
        last_loss = telem[-1]["loss"]
        if last_loss >= first_loss:
            notes.append(f"WARNING: loss not decreasing ({first_loss:.4f} → {last_loss:.4f})")

    # Must not be catastrophically worse than baseline (>50% worse BPB)
    delta_bpb = bm["final_validation_bpb"] - cm["final_validation_bpb"]
    if delta_bpb < -2.0:
        return False, [f"CATASTROPHICALLY WORSE: Δ BPB = {delta_bpb:+.4f}"]

    # Must not be >3x slower
    speed_ratio = cm["mean_step_time_ms"] / max(bm["mean_step_time_ms"], 1e-8)
    if speed_ratio > 3.0:
        return False, [f"TOO SLOW: {speed_ratio:.1f}x baseline step time"]

    # Informational notes
    if delta_bpb > 0:
        notes.append(f"BETTER than baseline by {delta_bpb:.4f} BPB")
    else:
        notes.append(f"Worse than baseline by {abs(delta_bpb):.4f} BPB (within tolerance)")

    if speed_ratio > 1.5:
        notes.append(f"Slower than baseline: {speed_ratio:.1f}x")

    return True, notes


def main():
    args = _parse_args()
    spec_mod = _load_spec_module()

    # ---- Load candidate spec ----
    if args.builtin:
        builtin_map = {
            "baseline": spec_mod.MatrixOptimizerSpec.baseline_nanochat,
            "trust_ratio": spec_mod.MatrixOptimizerSpec.trust_ratio_variant,
            "stateful_annealing": spec_mod.MatrixOptimizerSpec.stateful_annealing_variant,
        }
        spec = builtin_map[args.builtin]()
        spec_dict = spec.to_dict()
        print(f"Using built-in spec: {args.builtin}")
    elif args.spec == "-":
        spec_dict = json.loads(sys.stdin.read())
    else:
        spec_dict = json.loads(Path(args.spec).read_text())

    baseline_spec = spec_mod.MatrixOptimizerSpec.baseline_nanochat()
    baseline_dict = baseline_spec.to_dict()

    # ---- Phase 1: Local validation ----
    print()
    print("=" * 60)
    print("PHASE 1: Local Validation (syntax + constraints)")
    print("=" * 60)

    ok, msg = _local_validate(spec_mod, spec_dict)
    print(f"  {msg}")

    if not ok:
        print()
        print("VERDICT: ❌ REJECTED — spec failed local validation")
        return 1

    spec_name = spec_dict.get("name", "?")

    # Check if candidate IS the baseline
    is_baseline = (spec_name == baseline_spec.name)
    if is_baseline:
        print("  NOTE: candidate is the baseline itself (smoke test mode)")

    # ---- Phase 2: GPU validation ----
    print()
    print("=" * 60)
    print(f"PHASE 2: GPU Validation ({args.steps} steps on A100)")
    print("=" * 60)
    print(f"  Candidate: {spec_name}")
    print(f"  Baseline:  {baseline_spec.name}")
    print(f"  Steps:     {args.steps}")
    print(f"  Seed:      {args.seed}")
    print()

    t0 = time.time()
    with modal.enable_output():
        with app.run():
            result = validate_on_gpu.remote(spec_dict, baseline_dict, args.steps, args.seed)
    wall_time = time.time() - t0

    candidate_result = result["candidate"]
    baseline_result = result["baseline"]

    # ---- Phase 3: Telemetry report ----
    print()
    print("=" * 60)
    print("PHASE 3: Telemetry Report")
    print("=" * 60)

    _print_telemetry_summary("Candidate", candidate_result)
    print()
    _print_telemetry_summary("Baseline", baseline_result)

    # ---- Phase 4: Verdict ----
    valid, notes = _judge_validity(candidate_result, baseline_result)

    cm = candidate_result.get("metrics")
    bm = baseline_result.get("metrics")

    print()
    print("=" * 60)
    if valid:
        print("VERDICT: ✅ VALID — candidate may enter the evaluation queue")
    else:
        print("VERDICT: ❌ REJECTED — candidate failed validation")
    print("=" * 60)

    for note in notes:
        print(f"  {note}")

    if cm and bm:
        delta = bm["final_validation_bpb"] - cm["final_validation_bpb"]
        speed = cm["mean_step_time_ms"] / max(bm["mean_step_time_ms"], 1e-8)
        print()
        print(f"  Δ BPB:         {delta:+.4f} ({'better' if delta > 0 else 'worse'})")
        print(f"  Speed ratio:   {speed:.2f}x baseline")

    print(f"  Wall time:     {wall_time:.0f}s")

    # ---- Save results ----
    if args.out:
        output = {
            "spec": spec_dict,
            "valid": valid,
            "notes": notes,
            "candidate": candidate_result,
            "baseline": baseline_result,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\n  Results saved to {out_path}")

    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
