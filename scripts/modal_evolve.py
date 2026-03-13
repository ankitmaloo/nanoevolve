# /// script
# requires-python = ">=3.10"
# dependencies = ["modal>=1.0.0"]
# ///
"""
NanoEvolve — Evolve optimizer specs on real NanoChat training via Modal A100.

End-to-end workflow:
1. Start with baseline (or a parent spec JSON)
2. Apply N random DSL mutations
3. Run each mutation + baseline on A100 for 20 steps
4. Score, rank, and report telemetry

Usage:
    # Run 4 mutations against baseline, 20 steps each:
    uv run scripts/modal_evolve.py

    # More mutations, more steps:
    uv run scripts/modal_evolve.py --mutations 8 --steps 40

    # From a parent spec JSON:
    uv run scripts/modal_evolve.py --parent-spec path/to/spec.json
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
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

app = modal.App("nanoevolve-evolve", image=image)
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
def evaluate_spec_on_gpu(spec_dict: dict, steps: int, seed: int, candidate_id: str) -> dict:
    """Evaluate a single MatrixOptimizerSpec on real NanoChat training.

    Takes and returns plain dicts (Modal serialization).
    """
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

    # Add paths so adamopt + nanochat are importable in-process
    venv_site = "/root/nanoevolve/nanochat/.venv/lib/python3.12/site-packages"
    for p in [REMOTE_REPO, REMOTE_NANOCHAT, venv_site]:
        if p not in _sys.path:
            _sys.path.insert(0, p)

    from adamopt.optim_search.real_eval import RealEvalConfig, evaluate_real_nanochat
    from adamopt.optim_search.spec import MatrixOptimizerSpec

    spec = MatrixOptimizerSpec.from_dict(spec_dict)
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

    outcome = evaluate_real_nanochat(spec, config, candidate_id=candidate_id)

    # Serialize to plain dict for Modal transport
    return asdict(outcome)


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="NanoEvolve: evolve optimizer specs on real NanoChat training")
    parser.add_argument("--mutations", type=int, default=4, help="Number of mutations to try")
    parser.add_argument("--steps", type=int, default=20, help="Training steps per eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--parent-spec", type=str, default=None, help="Path to parent spec JSON (default: baseline)")
    parser.add_argument("--out", type=str, default=None, help="Path to write results JSON")
    return parser.parse_args()


def main():
    args = _parse_args()

    # Import spec and mutations directly, bypassing adamopt/optim_search/__init__.py
    # which eagerly imports torch-dependent modules (eval_candidate, candidate_optimizer).
    # spec.py and mutations.py only depend on stdlib — no torch needed locally.
    import importlib.util

    def _import_module_from_file(name: str, path: str):
        mod_spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(mod_spec)
        # Set __package__ so relative imports (from .spec import ...) resolve
        mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
        sys.modules[name] = mod
        mod_spec.loader.exec_module(mod)
        return mod

    # Register package stubs so relative imports (from .spec import ...) resolve
    import types
    for pkg_name in ("adamopt", "adamopt.optim_search"):
        if pkg_name not in sys.modules:
            sys.modules[pkg_name] = types.ModuleType(pkg_name)

    spec_mod = _import_module_from_file(
        "adamopt.optim_search.spec",
        str(REPO_ROOT / "adamopt" / "optim_search" / "spec.py"),
    )
    mutations_mod = _import_module_from_file(
        "adamopt.optim_search.mutations",
        str(REPO_ROOT / "adamopt" / "optim_search" / "mutations.py"),
    )
    MatrixOptimizerSpec = spec_mod.MatrixOptimizerSpec
    mutate_spec = mutations_mod.mutate_spec

    # Load or create parent spec
    if args.parent_spec:
        parent = MatrixOptimizerSpec.from_dict(json.loads(Path(args.parent_spec).read_text()))
    else:
        parent = MatrixOptimizerSpec.baseline_nanochat()

    # Generate mutations
    rng = random.Random(args.seed)
    candidates: list[tuple[str, dict, dict]] = []  # (candidate_id, spec_dict, lineage)

    # Always include baseline
    baseline_id = "baseline"
    baseline_dict = parent.to_dict()

    for i in range(args.mutations):
        mutated, lineage = mutate_spec(parent, rng)
        candidate_id = f"mut_{i:03d}_{lineage.get('mutation', 'unknown')}"
        candidates.append((candidate_id, mutated.to_dict(), lineage))

    print(f"=" * 70)
    print(f"NanoEvolve — Real NanoChat GPU Evaluation")
    print(f"=" * 70)
    print(f"  Parent spec:    {parent.name}")
    print(f"  Mutations:      {len(candidates)}")
    print(f"  Steps/eval:     {args.steps}")
    print(f"  Seed:           {args.seed}")
    print(f"  Total GPU runs: {len(candidates) + 1} (baseline + {len(candidates)} mutations)")
    print()

    for cid, _, lineage in candidates:
        print(f"  {cid}: {lineage.get('mutation', '?')}")
    print()

    # Run all evaluations on Modal in parallel
    with modal.enable_output():
        with app.run():
            # Launch baseline + all mutations in parallel
            print(f"Launching {len(candidates) + 1} GPU evaluations...")

            handles = {}
            handles[baseline_id] = evaluate_spec_on_gpu.spawn(
                baseline_dict, args.steps, args.seed, baseline_id
            )
            for cid, spec_dict, _ in candidates:
                handles[cid] = evaluate_spec_on_gpu.spawn(
                    spec_dict, args.steps, args.seed, cid
                )

            # Collect results
            results = {}
            for cid, handle in handles.items():
                print(f"  Waiting for {cid}...")
                results[cid] = handle.get()

    # Score and report
    baseline_result = results[baseline_id]
    baseline_metrics = baseline_result.get("metrics")

    print()
    print(f"=" * 70)
    print(f"RESULTS")
    print(f"=" * 70)

    if baseline_metrics:
        print(f"\n  Baseline: {parent.name}")
        print(f"    final_val_bpb:    {baseline_metrics['final_validation_bpb']:.4f}")
        print(f"    best_val_bpb:     {baseline_metrics['best_validation_bpb']:.4f}")
        print(f"    mean_step_ms:     {baseline_metrics['mean_step_time_ms']:.1f}")
        print(f"    tokens/sec:       {baseline_metrics['tokens_per_sec']:.0f}")
        print(f"    stability:        {baseline_metrics['stability_penalty']}")
        print(f"    memory_bytes:     {baseline_metrics['memory_overhead_bytes']}")
    else:
        print(f"\n  Baseline FAILED: {baseline_result.get('failure_type')}")

    print()
    print(f"  {'Candidate':<50} {'Val BPB':>10} {'Δ BPB':>10} {'Step ms':>10} {'Status':>10}")
    print(f"  {'-'*50} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    ranked = []
    for cid, spec_dict, lineage in candidates:
        r = results[cid]
        m = r.get("metrics")
        if m and baseline_metrics:
            delta = baseline_metrics["final_validation_bpb"] - m["final_validation_bpb"]
            ranked.append((cid, m, delta, lineage, r))
            status = "✅ BETTER" if delta > 0 else "❌ WORSE"
            print(f"  {cid:<50} {m['final_validation_bpb']:>10.4f} {delta:>+10.4f} {m['mean_step_time_ms']:>10.1f} {status:>10}")
        elif not r.get("valid"):
            print(f"  {cid:<50} {'—':>10} {'—':>10} {'—':>10} {'💀 DEAD':>10}")
        else:
            print(f"  {cid:<50} {'—':>10} {'—':>10} {'—':>10} {'⚠️ NO METRICS':>10}")

    # Sort by improvement
    ranked.sort(key=lambda x: x[2], reverse=True)

    if ranked:
        print()
        print(f"=" * 70)
        print(f"RANKING (best improvement first)")
        print(f"=" * 70)
        for rank, (cid, m, delta, lineage, r) in enumerate(ranked, 1):
            winner = "🏆" if delta > 0 else "  "
            print(f"  {winner} #{rank}: {cid}")
            print(f"       mutation:      {lineage.get('mutation', '?')}")
            print(f"       val_bpb:       {m['final_validation_bpb']:.4f} (Δ {delta:+.4f})")
            print(f"       step_time:     {m['mean_step_time_ms']:.1f} ms")
            print(f"       tokens/sec:    {m['tokens_per_sec']:.0f}")
            print(f"       nan/inf:       {m['nan_failures']}/{m['inf_failures']}")
            print(f"       grad_spikes:   {m['grad_norm_spikes']}")

            # Telemetry summary
            telem = r.get("telemetry", [])
            if telem:
                gates = [s["gate_value"] for s in telem]
                losses = [s["loss"] for s in telem]
                print(f"       loss range:    {losses[0]:.4f} → {losses[-1]:.4f}")
                print(f"       gate range:    [{min(gates):.4f}, {max(gates):.4f}]")
            print()

    # Save results if requested
    if args.out:
        output = {
            "parent_spec": baseline_dict,
            "baseline": baseline_result,
            "candidates": {
                cid: {
                    "spec": spec_dict,
                    "lineage": lineage,
                    "outcome": results[cid],
                }
                for cid, spec_dict, lineage in candidates
            },
            "ranking": [
                {
                    "candidate_id": cid,
                    "delta_bpb": delta,
                    "mutation": lineage.get("mutation"),
                }
                for cid, _, delta, lineage, _ in ranked
            ],
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
