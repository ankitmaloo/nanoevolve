#!/usr/bin/env python3
"""Enigma × AdamOpt: Full doctrine loop with real artifacts and GPU evaluation.

Creates a proper run directory with:
  runs/enigma_YYYYMMDD_HHMMSS/
    config.json              — search config
    code_context.md          — dynamic context
    memory/search_memory.json — full search state
    specs/
      baseline.json          — baseline MatrixOptimizerSpec
      P1_H01_post_momentum.json
      P2_H02_trust_ratio.json
      P3_H04_stateful_gate.json
      P4_H05_ns3.json
      P5_H08_compound.json
    evaluations/
      baseline_toy.json
      P1_toy.json ... P5_toy.json
      baseline_real.json      (after GPU run)
      P1_real.json ... P5_real.json
    loop_001/
      portfolio.json
      results.json
      postmortem.json

Usage:
    # Stage 1: Toy backend (local, no GPU needed)
    python enigma/run_adamopt.py --stage toy

    # Stage 2: Real NanoChat on GPU via SSH
    python enigma/run_adamopt.py --stage real --host user54@35.84.33.219

    # Stage 3: Score and rank results
    python enigma/run_adamopt.py --stage score --run-dir runs/enigma_XXXXX
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch

# adamopt imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adamopt"))

from optim_search.spec import (
    ClipConfig,
    MatrixOptimizerSpec,
    TrustRatioConfig,
)
from optim_search.mutations import mutate_spec
from optim_search.eval_candidate import ToyNanoChatBackend
from optim_search.config import EvaluationConfig, SearchConfig as AdamSearchConfig
from optim_search.score import composite_score, analyze_win_hierarchy

# enigma imports
from enigma.memory import SearchMemory
from enigma.types import (
    AttackSurface,
    ExperimentRole,
    GapEntry,
    Hypothesis,
    HypothesisStatus,
    LoopRecord,
    MutationNeighborhood,
    PortfolioSlot,
    SurfaceStatus,
)


# ── The 5 concrete mutations ───────────────────────────────────

PORTFOLIO_SPEC = [
    {
        "slot_id": "P1",
        "role": "scout",
        "hypothesis_id": "H01",
        "title": "Post-orthogonal momentum",
        "family": "momentum",
        "target_region": "momentum_pipeline",
        "operator_family": "structural",
    },
    {
        "slot_id": "P2",
        "role": "scout",
        "hypothesis_id": "H02",
        "title": "Layerwise trust ratio",
        "family": "trust_ratio",
        "target_region": "trust_ratio_system",
        "operator_family": "policy",
    },
    {
        "slot_id": "P3",
        "role": "exploit",
        "hypothesis_id": "H04",
        "title": "Stateful gate (phase-aware)",
        "family": "stateful_control",
        "target_region": "stateful_gate_system",
        "operator_family": "state",
    },
    {
        "slot_id": "P4",
        "role": "exploit",
        "hypothesis_id": "H05",
        "title": "Fewer Polar Express steps (ns=3)",
        "family": "orthogonalization",
        "target_region": "orthogonalization_depth",
        "operator_family": "schedule",
    },
    {
        "slot_id": "P5",
        "role": "wildcard",
        "hypothesis_id": "H08",
        "title": "Trust ratio + update clipping",
        "family": "compound",
        "target_region": "trust_ratio+clipping",
        "operator_family": "policy",
    },
]


def build_mutation(slot: dict, baseline: MatrixOptimizerSpec) -> tuple[MatrixOptimizerSpec, dict]:
    """Build a concrete MatrixOptimizerSpec for each portfolio slot."""
    hid = slot["hypothesis_id"]

    if hid == "H01":
        spec = replace(baseline, name="enigma_H01_post_momentum",
                       momentum_placement="post_orthogonal",
                       metadata={"enigma_slot": slot["slot_id"], "enigma_hypothesis": hid})
        return spec, {"mutation": "toggle_momentum_placement", "value": "post_orthogonal"}

    elif hid == "H02":
        spec = replace(baseline, name="enigma_H02_trust_ratio",
                       trust_ratio=TrustRatioConfig(mode="layerwise", clamp_min=0.5, clamp_max=1.5),
                       metadata={"enigma_slot": slot["slot_id"], "enigma_hypothesis": hid})
        return spec, {"mutation": "enable_trust_ratio", "mode": "layerwise", "clamp": [0.5, 1.5]}

    elif hid == "H04":
        spec = MatrixOptimizerSpec.stateful_annealing_variant()
        spec = replace(spec, name="enigma_H04_stateful_gate",
                       metadata={"enigma_slot": slot["slot_id"], "enigma_hypothesis": hid})
        return spec, {"mutation": "enable_stateful_control", "variant": "stateful_annealing"}

    elif hid == "H05":
        spec = replace(baseline, name="enigma_H05_ns3", ns_steps=3,
                       metadata={"enigma_slot": slot["slot_id"], "enigma_hypothesis": hid})
        return spec, {"mutation": "adjust_ns_steps", "ns_steps": 3, "baseline_ns_steps": 5}

    elif hid == "H08":
        spec = replace(baseline, name="enigma_H08_compound",
                       trust_ratio=TrustRatioConfig(mode="layerwise", clamp_min=0.5, clamp_max=1.5),
                       clip=ClipConfig(mode="update_rms", threshold=1.0),
                       metadata={"enigma_slot": slot["slot_id"], "enigma_hypothesis": hid})
        return spec, {"mutation": "compound_trust_ratio_clipping"}

    raise ValueError(f"Unknown hypothesis: {hid}")


# ── Stage 1: Toy Backend ───────────────────────────────────────

def run_toy_stage(run_dir: Path) -> dict:
    """Run all 5 mutations through the toy backend with full artifact logging."""
    print(f"\n{'='*70}")
    print(f"  ENIGMA × ADAMOPT — Toy Backend Evaluation")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*70}\n")

    # Create directory structure
    specs_dir = run_dir / "specs"
    evals_dir = run_dir / "evaluations"
    loop_dir = run_dir / "loop_001"
    memory_dir = run_dir / "memory"
    for d in [specs_dir, evals_dir, loop_dir, memory_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Baseline
    baseline = MatrixOptimizerSpec.baseline_nanochat()
    (specs_dir / "baseline.json").write_text(baseline.to_json())

    eval_config = EvaluationConfig(seed=42, steps=24, eval_every=6)
    backend = ToyNanoChatBackend(eval_config)

    print("Evaluating baseline...")
    t0 = time.perf_counter()
    baseline_outcome = backend.evaluate(baseline, seed=42, candidate_id="baseline")
    t_baseline = time.perf_counter() - t0
    assert baseline_outcome.valid, f"Baseline failed: {baseline_outcome.failure_type}"

    baseline_result = {
        "candidate_id": "baseline",
        "spec_name": baseline.name,
        "valid": baseline_outcome.valid,
        "final_validation_bpb": baseline_outcome.metrics.final_validation_bpb,
        "best_validation_bpb": baseline_outcome.metrics.best_validation_bpb,
        "tokens_per_sec": baseline_outcome.metrics.tokens_per_sec,
        "mean_step_time_ms": baseline_outcome.metrics.mean_step_time_ms,
        "stability_penalty": baseline_outcome.metrics.stability_penalty,
        "memory_bytes": baseline_outcome.metrics.memory_overhead_bytes,
        "eval_time_s": t_baseline,
        "curve": [asdict(p) for p in baseline_outcome.curve],
    }
    (evals_dir / "baseline_toy.json").write_text(json.dumps(baseline_result, indent=2))
    print(f"  Baseline: val_bpb={baseline_outcome.metrics.final_validation_bpb:.4f} "
          f"({t_baseline:.1f}s)")

    # Populate search memory
    memory = SearchMemory(memory_dir)
    memory.current_loop = 1
    _populate_memory(memory, baseline_outcome)

    # Run 5 mutations
    print(f"\nEvaluating 5 mutations...")
    results = []

    for slot_spec in PORTFOLIO_SPEC:
        slot_id = slot_spec["slot_id"]
        hid = slot_spec["hypothesis_id"]

        # Build mutation
        spec, lineage = build_mutation(slot_spec, baseline)
        spec.validate()

        # Round-trip check
        rt = MatrixOptimizerSpec.from_dict(spec.to_dict())
        assert rt.stable_id() == spec.stable_id(), f"{slot_id} round-trip failed"

        # Save spec
        spec_file = specs_dir / f"{slot_id}_{hid}_{spec.name}.json"
        spec_file.write_text(spec.to_json())

        # Evaluate
        t0 = time.perf_counter()
        outcome = backend.evaluate(spec, seed=42, candidate_id=f"enigma_{slot_id}")
        t_eval = time.perf_counter() - t0

        # Score
        score = composite_score(outcome, baseline_outcome)
        win = analyze_win_hierarchy(outcome, baseline_outcome)

        # Compute deltas
        delta_bpb = baseline_outcome.metrics.final_validation_bpb - outcome.metrics.final_validation_bpb
        speed_ratio = baseline_outcome.metrics.mean_step_time_ms / max(outcome.metrics.mean_step_time_ms, 1e-8)

        result = {
            "slot": slot_id,
            "hypothesis": hid,
            "role": slot_spec["role"],
            "title": slot_spec["title"],
            "spec_name": spec.name,
            "spec_file": str(spec_file.relative_to(run_dir)),
            "valid": outcome.valid,
            "final_validation_bpb": outcome.metrics.final_validation_bpb,
            "best_validation_bpb": outcome.metrics.best_validation_bpb,
            "delta_bpb": delta_bpb,
            "composite_score": score,
            "winner": win.winner,
            "hierarchy_level": win.hierarchy_level,
            "dominant_axes": win.dominant_axes,
            "speed_ratio": speed_ratio,
            "tokens_per_sec": outcome.metrics.tokens_per_sec,
            "mean_step_time_ms": outcome.metrics.mean_step_time_ms,
            "stability_penalty": outcome.metrics.stability_penalty,
            "memory_bytes": outcome.metrics.memory_overhead_bytes,
            "grad_norm_spikes": outcome.metrics.grad_norm_spikes,
            "lineage": lineage,
            "eval_time_s": t_eval,
            "curve": [asdict(p) for p in outcome.curve],
            "notes": win.notes,
        }
        results.append(result)

        # Save individual eval
        (evals_dir / f"{slot_id}_toy.json").write_text(json.dumps(result, indent=2))

        marker = "WIN" if win.winner else ("+" if delta_bpb > 0 else "-")
        print(f"  [{marker}] {slot_id} ({slot_spec['role']:>8s}) {spec.name:40s} "
              f"bpb={outcome.metrics.final_validation_bpb:.4f} "
              f"Δ={delta_bpb:+.4f} score={score:+.1f} "
              f"speed={speed_ratio:.2f}x")

    # Save combined results
    (loop_dir / "portfolio.json").write_text(json.dumps(
        [asdict(s) for s in memory.portfolio], indent=2, default=str,
    ))
    (loop_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Rank
    ranked = sorted(results, key=lambda r: r["composite_score"], reverse=True)
    print(f"\n{'='*70}")
    print(f"  RANKING (by composite score)")
    print(f"{'='*70}")
    for i, r in enumerate(ranked, 1):
        print(f"  #{i} {r['slot']} ({r['hypothesis']}) score={r['composite_score']:+.1f} "
              f"Δbpb={r['delta_bpb']:+.4f} {r['title']}")

    # Postmortem
    best = ranked[0]
    worst = ranked[-1]
    postmortem = {
        "loop": 1,
        "baseline_bpb": baseline_outcome.metrics.final_validation_bpb,
        "best": {"slot": best["slot"], "hypothesis": best["hypothesis"],
                 "score": best["composite_score"], "delta_bpb": best["delta_bpb"]},
        "worst": {"slot": worst["slot"], "hypothesis": worst["hypothesis"],
                  "score": worst["composite_score"], "delta_bpb": worst["delta_bpb"]},
        "all_valid": all(r["valid"] for r in results),
        "ranking": [{"slot": r["slot"], "score": r["composite_score"]} for r in ranked],
        "next_step": "Run on real NanoChat GPU with --stage real --host <server>",
    }
    (loop_dir / "postmortem.json").write_text(json.dumps(postmortem, indent=2))

    # Update memory with results
    _update_memory_with_results(memory, results, baseline_outcome)
    memory.save()

    # Save config
    (run_dir / "config.json").write_text(json.dumps({
        "stage": "toy",
        "eval_config": asdict(eval_config),
        "timestamp": datetime.now().isoformat(),
        "portfolio_size": len(PORTFOLIO_SPEC),
        "baseline_spec": "nanochat_muon_baseline",
    }, indent=2))

    print(f"\nArtifacts saved to: {run_dir}")
    print(f"  specs/       — {len(list(specs_dir.iterdir()))} spec JSON files")
    print(f"  evaluations/ — {len(list(evals_dir.iterdir()))} evaluation results")
    print(f"  loop_001/    — portfolio, results, postmortem")
    print(f"  memory/      — search_memory.json")

    return postmortem


# ── Stage 2: Real NanoChat on GPU ──────────────────────────────

def run_real_stage(run_dir: Path, host: str) -> None:
    """Ship specs to GPU server, run real NanoChat training, collect results."""
    import subprocess

    specs_dir = run_dir / "specs"
    evals_dir = run_dir / "evaluations"

    spec_files = sorted(specs_dir.glob("*.json"))
    if not spec_files:
        print(f"No spec files in {specs_dir}. Run --stage toy first.")
        return

    print(f"\n{'='*70}")
    print(f"  ENIGMA × ADAMOPT — Real NanoChat GPU Evaluation")
    print(f"  Host: {host}")
    print(f"  Specs: {len(spec_files)} files")
    print(f"{'='*70}\n")

    # Build the remote evaluation script
    eval_script = _build_remote_eval_script()
    eval_script_path = run_dir / "remote_eval.py"
    eval_script_path.write_text(eval_script)

    # Push specs and eval script to server
    remote_dir = f"/tmp/enigma_run_{run_dir.name}"
    print(f"Pushing to {host}:{remote_dir} ...")

    subprocess.run(["ssh", host, f"mkdir -p {remote_dir}/specs {remote_dir}/results"],
                   check=True, capture_output=True)

    # Copy spec files
    for sf in spec_files:
        subprocess.run(["scp", str(sf), f"{host}:{remote_dir}/specs/"],
                       check=True, capture_output=True)

    # Copy eval script
    subprocess.run(["scp", str(eval_script_path), f"{host}:{remote_dir}/"],
                   check=True, capture_output=True)

    # Copy the adamopt and nanochat source needed for real eval
    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(["scp", "-r", str(repo_root / "adamopt"), f"{host}:{remote_dir}/"],
                   check=True, capture_output=True)
    subprocess.run(["scp", "-r", str(repo_root / "nanochat"), f"{host}:{remote_dir}/"],
                   check=True, capture_output=True)

    print(f"Running real evaluations on {host} ...")

    # Run the eval script remotely
    result = subprocess.run(
        ["ssh", host,
         f"cd {remote_dir} && "
         f"PYTHONPATH={remote_dir}:{remote_dir}/adamopt:{remote_dir}/nanochat "
         f"python {remote_dir}/remote_eval.py "
         f"--specs-dir {remote_dir}/specs "
         f"--results-dir {remote_dir}/results "
         f"--steps 20 --eval-every 10 --depth 4"],
        capture_output=True, text=True, timeout=1800,
    )

    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr[:500]}")

    # Pull results back
    print(f"\nPulling results from {host} ...")
    for spec_file in spec_files:
        stem = spec_file.stem
        result_file = f"{remote_dir}/results/{stem}_real.json"
        local_file = evals_dir / f"{stem}_real.json"
        pull = subprocess.run(
            ["scp", f"{host}:{result_file}", str(local_file)],
            capture_output=True, text=True,
        )
        if pull.returncode == 0:
            print(f"  Got: {local_file.name}")
        else:
            print(f"  Missing: {stem}_real.json")

    print(f"\nResults saved to: {evals_dir}")


def _build_remote_eval_script() -> str:
    """Build a self-contained Python script that runs on the GPU server."""
    return '''#!/usr/bin/env python3
"""Remote evaluator — runs each spec through real NanoChat training."""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    from optim_search.spec import MatrixOptimizerSpec
    from optim_search.real_eval import RealEvalConfig, evaluate_real_nanochat

    config = RealEvalConfig(
        seed=args.seed,
        steps=args.steps,
        eval_every=args.eval_every,
        depth=args.depth,
        max_seq_len=512,
        device_batch_size=2,
        total_batch_size=1024,
        device="cuda",
    )

    spec_files = sorted(args.specs_dir.glob("*.json"))
    print(f"Found {len(spec_files)} spec files")

    for spec_file in spec_files:
        print(f"\\nEvaluating: {spec_file.name}")
        spec_dict = json.loads(spec_file.read_text())
        spec = MatrixOptimizerSpec.from_dict(spec_dict)

        t0 = time.perf_counter()
        try:
            outcome = evaluate_real_nanochat(spec, config, candidate_id=spec_file.stem)
            elapsed = time.perf_counter() - t0

            result = {
                "candidate_id": spec_file.stem,
                "spec_name": spec.name,
                "valid": outcome.valid,
                "failure_type": outcome.failure_type,
                "eval_time_s": elapsed,
            }
            if outcome.valid and outcome.metrics:
                result.update({
                    "final_validation_bpb": outcome.metrics.final_validation_bpb,
                    "best_validation_bpb": outcome.metrics.best_validation_bpb,
                    "tokens_per_sec": outcome.metrics.tokens_per_sec,
                    "mean_step_time_ms": outcome.metrics.mean_step_time_ms,
                    "stability_penalty": outcome.metrics.stability_penalty,
                    "memory_bytes": outcome.metrics.memory_overhead_bytes,
                    "grad_norm_spikes": outcome.metrics.grad_norm_spikes,
                    "nan_failures": outcome.metrics.nan_failures,
                    "inf_failures": outcome.metrics.inf_failures,
                })
                result["curve"] = [asdict(p) for p in outcome.curve]
                result["telemetry"] = [asdict(t) for t in outcome.telemetry]

            out_file = args.results_dir / f"{spec_file.stem}_real.json"
            out_file.write_text(json.dumps(result, indent=2))
            bpb = result.get("final_validation_bpb", "N/A")
            print(f"  => valid={outcome.valid} bpb={bpb} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            result = {
                "candidate_id": spec_file.stem,
                "valid": False,
                "error": str(e),
                "eval_time_s": elapsed,
            }
            out_file = args.results_dir / f"{spec_file.stem}_real.json"
            out_file.write_text(json.dumps(result, indent=2))
            print(f"  => FAILED: {e}")

if __name__ == "__main__":
    main()
'''


# ── Memory population helpers ──────────────────────────────────

def _populate_memory(memory: SearchMemory, baseline_outcome) -> None:
    """Populate search memory with doctrine state."""
    # Variables
    memory.variables = {
        "task_snapshot": {
            "target": "NanoChat MuonAdamW optimizer",
            "primary_metric": "final_validation_bpb",
            "baseline_bpb": baseline_outcome.metrics.final_validation_bpb,
        },
    }

    # Surfaces
    for s in [
        AttackSurface(id="S01", region="momentum_pipeline", bottleneck="momentum order",
                      change_class="structural", mechanism="pre/post toggle",
                      leverage=3, plausibility=4, observability=4, implementation_cost=1, overlap_risk=1),
        AttackSurface(id="S02", region="trust_ratio_system", bottleneck="layer-wise mismatch",
                      change_class="policy", mechanism="layerwise trust ratio",
                      leverage=4, plausibility=4, observability=4, implementation_cost=2, overlap_risk=2),
        AttackSurface(id="S03", region="gradient_clipping", bottleneck="gradient spikes",
                      change_class="policy", mechanism="update RMS / global norm clip",
                      leverage=3, plausibility=5, observability=5, implementation_cost=1, overlap_risk=1),
        AttackSurface(id="S04", region="stateful_gate_system", bottleneck="static optimizer",
                      change_class="state", mechanism="sensors → gate → actuators",
                      leverage=5, plausibility=3, observability=3, implementation_cost=4, overlap_risk=2),
        AttackSurface(id="S05", region="orthogonalization_depth", bottleneck="ns_steps tradeoff",
                      change_class="schedule", mechanism="adjust Polar Express iterations",
                      leverage=4, plausibility=4, observability=5, implementation_cost=1, overlap_risk=1),
    ]:
        memory.surfaces[s.id] = s

    # Portfolio
    memory.portfolio = [
        PortfolioSlot(id=s["slot_id"], role=ExperimentRole(s["role"]),
                      hypothesis_id=s["hypothesis_id"], family=s["family"],
                      target_region=s["target_region"], operator_family=s["operator_family"],
                      why_selected=s["title"], expected_signal=f"BPB improvement from {s['title']}",
                      acceptance_test="val BPB improves ≥0.005",
                      kill_condition="no improvement or stability regression",
                      overlap_check="unique")
        for s in PORTFOLIO_SPEC
    ]

    # Register neighborhoods
    for s in PORTFOLIO_SPEC:
        n_id = memory.next_neighborhood_id()
        memory.mutation_ledger[n_id] = MutationNeighborhood(
            id=n_id, loop=1, hypothesis_id=s["hypothesis_id"],
            target_region=s["target_region"], operator_family=s["operator_family"],
            bottleneck_attacked=s["family"], benchmark_slice="all",
            parent_candidate="baseline",
        )


def _update_memory_with_results(memory: SearchMemory, results: list, baseline_outcome) -> None:
    """Record loop postmortem in memory."""
    best = max(results, key=lambda r: r["composite_score"])
    worst = min(results, key=lambda r: r["composite_score"])

    memory.loop_log.append(LoopRecord(
        loop=1,
        date=datetime.now().isoformat(),
        target="NanoChat MuonAdamW",
        parent_baseline="baseline",
        baseline_metrics={"final_validation_bpb": baseline_outcome.metrics.final_validation_bpb},
        active_portfolio=[s["slot_id"] for s in PORTFOLIO_SPEC],
        overlap_conflicts=[],
        best_candidate=best["slot"],
        worst_candidate=worst["slot"],
        score_movement=f"best Δ={best['delta_bpb']:+.4f}",
        holdout_movement="pending real eval",
        stability_movement="all stable on toy",
        what_improved=f"{best['slot']}: {best['title']}",
        what_regressed=worst["slot"] if worst["composite_score"] < 0 else "nothing",
        strongest_evidence=f"{best['slot']} scored {best['composite_score']:.1f}",
        likely_causal_explanation=best["title"],
        hypotheses_promoted=[r["hypothesis"] for r in results if r["composite_score"] > 0],
        hypotheses_killed=[],
        child_hypotheses=[],
        negative_knowledge_added=[],
        neighborhoods_retired=[],
        neighborhoods_reopened=[],
        next_loop_focus="Validate top mutations on real NanoChat GPU",
    ))


# ── CLI ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Enigma × AdamOpt runner")
    parser.add_argument("--stage", choices=["toy", "real", "score"], default="toy")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--host", type=str, default=None, help="SSH host for real eval")
    args = parser.parse_args()

    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_root = Path(__file__).resolve().parent.parent
        args.run_dir = repo_root / "runs" / f"enigma_{timestamp}"

    args.run_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "toy":
        run_toy_stage(args.run_dir)
    elif args.stage == "real":
        if not args.host:
            print("--host required for real stage. e.g., --host user54@35.84.33.219")
            sys.exit(1)
        run_real_stage(args.run_dir, args.host)
    elif args.stage == "score":
        print(f"Scoring results in {args.run_dir} ...")
        # Read real results and score them
        evals_dir = args.run_dir / "evaluations"
        for f in sorted(evals_dir.glob("*_real.json")):
            data = json.loads(f.read_text())
            bpb = data.get("final_validation_bpb", "N/A")
            valid = data.get("valid", False)
            print(f"  {f.stem}: valid={valid} bpb={bpb}")


if __name__ == "__main__":
    main()
