#!/usr/bin/env python3
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
        print(f"\nEvaluating: {spec_file.name}")
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
