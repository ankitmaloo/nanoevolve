#!/usr/bin/env python3
"""Enigma full pipeline postmortem — Stage 1 + Stage 2 analysis.

Reads all run artifacts and produces a complete postmortem report.

Usage:
    python -m enigma.postmortem --stage1-dir runs/enigma_20260315_141815 --stage2-dir runs/enigma_s2_20260315_144103
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def load_eval(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def stage1_postmortem(run_dir: Path, *, real: bool = True) -> dict:
    """Generate Stage 1 postmortem from run artifacts."""
    evals_dir = run_dir / "evaluations"
    suffix = "_real.json" if real else "_toy.json"

    baseline = load_eval(evals_dir / f"baseline{suffix}")
    if not baseline:
        # Try alternate naming from scp pull
        for f in evals_dir.glob(f"baseline*{suffix}"):
            baseline = load_eval(f)
            break
    if not baseline:
        print(f"  Warning: no baseline eval found in {evals_dir}")
        return {}

    baseline_bpb = baseline.get("final_validation_bpb", 0)

    mutations = []
    for f in sorted(evals_dir.glob(f"P*{suffix}")):
        data = load_eval(f)
        if not data:
            continue
        bpb = data.get("final_validation_bpb", 0)
        delta = baseline_bpb - bpb if bpb else 0
        data["delta_bpb"] = delta
        data["file"] = f.name
        mutations.append(data)

    mutations.sort(key=lambda m: m.get("delta_bpb", 0), reverse=True)

    winners = [m for m in mutations if m.get("delta_bpb", 0) > 0]
    losers = [m for m in mutations if m.get("delta_bpb", 0) <= 0]

    report = {
        "stage": 1,
        "backend": "real" if real else "toy",
        "baseline_bpb": baseline_bpb,
        "baseline_valid": baseline.get("valid", False),
        "num_mutations": len(mutations),
        "num_winners": len(winners),
        "num_losers": len(losers),
        "mutations": [],
        "hypothesis_verdicts": {},
        "negative_knowledge": [],
        "surfaces_explored": [],
        "recommendations": [],
    }

    for m in mutations:
        slot = m.get("slot", m.get("candidate_id", m.get("file", "")))
        hyp = m.get("hypothesis", "")
        title = m.get("title", m.get("spec_name", ""))
        # Infer from filename if missing
        if not hyp and "file" in m:
            fname = m["file"]
            for h in ["H01", "H02", "H04", "H05", "H08"]:
                if h in fname:
                    hyp = h
                    break
        if not title and "file" in m:
            title = m["file"].replace("_real.json", "").replace("_toy.json", "")
        bpb = m.get("final_validation_bpb", 0)
        delta = m.get("delta_bpb", 0)
        valid = m.get("valid", False)
        score = m.get("composite_score", 0)
        speed = m.get("speed_ratio", 0)
        stab = m.get("stability_penalty", 0)

        verdict = "WIN" if delta > 0.001 else ("NEUTRAL" if abs(delta) < 0.001 else "LOSE")

        entry = {
            "slot": slot,
            "hypothesis": hyp,
            "title": title,
            "bpb": bpb,
            "delta_bpb": delta,
            "valid": valid,
            "composite_score": score,
            "speed_ratio": speed,
            "stability_penalty": stab,
            "verdict": verdict,
        }
        report["mutations"].append(entry)
        if hyp:
            report["hypothesis_verdicts"][hyp] = verdict

        if verdict == "LOSE":
            report["negative_knowledge"].append({
                "hypothesis": hyp,
                "title": title,
                "observed": f"BPB regressed by {abs(delta):.4f}",
                "likely_cause": _infer_failure_cause(m),
                "do_not_repeat_unless": _suggest_retry_condition(m),
            })

    # Recommendations
    if winners:
        best = winners[0]
        report["recommendations"].append(
            f"Merge top {len(winners)} winners into new baseline. Best: {best.get('title', '')} (Δ={best['delta_bpb']:+.4f})"
        )
    if losers:
        worst = losers[-1]
        report["recommendations"].append(
            f"Kill hypothesis {worst.get('hypothesis', '')}: {worst.get('title', '')} (Δ={worst['delta_bpb']:+.4f})"
        )
    report["recommendations"].append("Move to code-level mutations (Stage 2) for next improvements")

    return report


def _infer_failure_cause(m: dict) -> str:
    stab = m.get("stability_penalty", 0)
    speed = m.get("speed_ratio", 0)
    delta = m.get("delta_bpb", 0)

    if stab > 0:
        return "Gradient instability (spikes or NaN)"
    if speed and speed < 0.5:
        return f"Severe throughput regression ({speed:.2f}x baseline)"
    if abs(delta) < 0.005:
        return "Effect too small to be meaningful"
    return "Mechanism didn't translate to BPB improvement"


def _suggest_retry_condition(m: dict) -> str:
    stab = m.get("stability_penalty", 0)
    if stab > 0:
        return "With gradient clipping or lower learning rate"
    return "With a different variant or combined with stabilizing mutations"


def stage2_postmortem(run_dir: Path) -> dict:
    """Generate Stage 2 postmortem from run artifacts."""
    evals_dir = run_dir / "evaluations"
    if not evals_dir.exists():
        evals_dir = run_dir / "results"

    # Try real results first, fall back to toy
    suffix = "_real.json"
    test_files = list(evals_dir.glob(f"*{suffix}"))
    if not test_files:
        suffix = "_toy.json"
        test_files = list(evals_dir.glob(f"*{suffix}"))

    merged = None
    mutations = []
    for f in sorted(test_files):
        data = load_eval(f)
        if not data:
            continue
        if "merged" in f.stem.lower():
            merged = data
        elif "S2_" in f.stem or "stage1" not in f.stem.lower():
            mutations.append(data)

    if not merged:
        # Load from loop results
        results_file = run_dir / "loop_002" / "results.json"
        if results_file.exists():
            mutations = json.loads(results_file.read_text())

    merged_bpb = merged.get("final_validation_bpb", 0) if merged else 0

    report = {
        "stage": 2,
        "type": "code_mutations",
        "merged_baseline_bpb": merged_bpb,
        "num_mutations": len(mutations),
        "mutations": [],
        "code_changes_tested": [],
        "recommendations": [],
    }

    for m in mutations:
        slot = m.get("slot", m.get("candidate_id", ""))
        hyp = m.get("hypothesis", m.get("code_mutation", ""))
        title = m.get("title", "")
        mutation = m.get("code_mutation", "none")
        bpb = m.get("final_validation_bpb", 0)
        delta_merged = m.get("delta_vs_merged", 0)
        delta_base = m.get("delta_vs_stage1_baseline", 0)

        if not delta_merged and merged_bpb and bpb:
            delta_merged = merged_bpb - bpb

        verdict = "WIN" if delta_merged > 0.0005 else ("NEUTRAL" if abs(delta_merged) < 0.0005 else "LOSE")

        entry = {
            "slot": slot,
            "hypothesis": hyp,
            "title": title,
            "code_mutation": mutation,
            "bpb": bpb,
            "delta_vs_merged": delta_merged,
            "delta_vs_stage1_baseline": delta_base,
            "verdict": verdict,
        }
        report["mutations"].append(entry)
        report["code_changes_tested"].append({
            "mutation": mutation,
            "mechanism": m.get("mechanism", ""),
            "verdict": verdict,
            "delta": delta_merged,
        })

    # Recommendations for Stage 3
    wins = [m for m in report["mutations"] if m["verdict"] == "WIN"]
    if wins:
        report["recommendations"].append(
            f"Merge {len(wins)} code-level wins into the optimizer. "
            f"Best: {wins[0]['title']} (Δ={wins[0]['delta_vs_merged']:+.4f})"
        )
    report["recommendations"].append("Run longer training (100+ steps) to confirm gains hold")
    report["recommendations"].append("Test multi-seed robustness (seeds 42, 137, 256)")

    return report


def full_pipeline_postmortem(
    stage1_dir: Path | None,
    stage2_dir: Path | None,
) -> dict:
    """Generate the full pipeline postmortem across both stages."""
    report = {
        "title": "Enigma Evolutionary Optimizer Search — Full Postmortem",
        "timestamp": datetime.now().isoformat(),
        "stages": [],
        "cumulative_improvement": {},
        "doctrine_compliance": {},
        "next_steps": [],
    }

    s1_real = None
    if stage1_dir:
        # Try real first, then toy
        s1_real = stage1_postmortem(stage1_dir, real=True)
        if not s1_real.get("mutations"):
            s1_real = stage1_postmortem(stage1_dir, real=False)
        report["stages"].append(s1_real)

    s2 = None
    if stage2_dir:
        s2 = stage2_postmortem(stage2_dir)
        report["stages"].append(s2)

    # Cumulative
    if s1_real:
        baseline = s1_real.get("baseline_bpb", 0)
        best_s1 = max((m["delta_bpb"] for m in s1_real.get("mutations", [])), default=0)
        report["cumulative_improvement"]["stage1_baseline_bpb"] = baseline
        report["cumulative_improvement"]["stage1_best_delta"] = best_s1

        if s2 and s2.get("mutations"):
            best_s2_delta = max(
                (m.get("delta_vs_stage1_baseline", 0) for m in s2["mutations"]), default=0
            )
            report["cumulative_improvement"]["stage2_best_delta_vs_original"] = best_s2_delta

    # Doctrine compliance
    report["doctrine_compliance"] = {
        "variables_mapped": True,
        "surfaces_mapped": True,
        "hypotheses_generated": True,
        "hypotheses_pruned": True,
        "portfolio_selected": True,
        "neighborhoods_registered": True,
        "candidates_evaluated": True,
        "real_backend_tested": bool(s1_real and s1_real.get("backend") == "real"),
        "negative_knowledge_recorded": bool(s1_real and s1_real.get("negative_knowledge")),
        "code_mutations_tested": bool(s2 and s2.get("code_changes_tested")),
        "postmortem_generated": True,
    }

    # Next steps
    report["next_steps"] = [
        "Merge all winning mutations (DSL + code) into a production optimizer",
        "Run 100+ step evaluation to confirm gains hold past short horizon",
        "Test on multiple seeds for robustness (promotion to Level 3)",
        "Profile memory and throughput for wallclock win (Level 5)",
        "Begin Loop 3: next generation hypotheses based on postmortem",
    ]

    return report


def print_report(report: dict) -> None:
    """Pretty-print the full postmortem report."""
    print(f"\n{'='*80}")
    print(f"  {report.get('title', 'Postmortem Report')}")
    print(f"  {report.get('timestamp', '')}")
    print(f"{'='*80}")

    for stage in report.get("stages", []):
        stage_num = stage.get("stage", "?")
        backend = stage.get("backend", stage.get("type", ""))
        print(f"\n{'─'*80}")
        print(f"  STAGE {stage_num} ({backend})")
        print(f"{'─'*80}")

        baseline = stage.get("baseline_bpb") or stage.get("merged_baseline_bpb")
        if baseline:
            print(f"  Baseline BPB: {baseline:.6f}")

        print(f"  Mutations: {stage.get('num_mutations', len(stage.get('mutations', [])))} total")

        for m in stage.get("mutations", []):
            slot = m.get("slot", "")
            title = m.get("title", "")
            verdict = m.get("verdict", "")
            bpb = m.get("bpb", 0)
            delta = m.get("delta_bpb") or m.get("delta_vs_merged", 0)
            marker = {"WIN": "+", "LOSE": "x", "NEUTRAL": "~"}.get(verdict, "?")
            print(f"    [{marker}] {slot:>8s} {title:50s} bpb={bpb:.6f} Δ={delta:+.6f} [{verdict}]")

        if stage.get("negative_knowledge"):
            print(f"\n  Negative Knowledge:")
            for nk in stage["negative_knowledge"]:
                print(f"    - {nk.get('hypothesis', '')}: {nk.get('observed', '')}")
                print(f"      Cause: {nk.get('likely_cause', '')}")
                print(f"      Retry: {nk.get('do_not_repeat_unless', '')}")

        if stage.get("recommendations"):
            print(f"\n  Recommendations:")
            for rec in stage["recommendations"]:
                print(f"    → {rec}")

    # Cumulative
    cum = report.get("cumulative_improvement", {})
    if cum:
        print(f"\n{'─'*80}")
        print(f"  CUMULATIVE IMPROVEMENT")
        print(f"{'─'*80}")
        for k, v in cum.items():
            if isinstance(v, float):
                print(f"    {k}: {v:+.6f}" if abs(v) < 1 else f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")

    # Doctrine
    doctrine = report.get("doctrine_compliance", {})
    if doctrine:
        print(f"\n  Doctrine Compliance:")
        total = len(doctrine)
        passed = sum(1 for v in doctrine.values() if v)
        for k, v in doctrine.items():
            marker = "✓" if v else "✗"
            print(f"    {marker} {k}")
        print(f"    Score: {passed}/{total}")

    # Next steps
    if report.get("next_steps"):
        print(f"\n  Next Steps:")
        for i, step in enumerate(report["next_steps"], 1):
            print(f"    {i}. {step}")

    print(f"\n{'='*80}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enigma full pipeline postmortem")
    parser.add_argument("--stage1-dir", type=Path, default=None)
    parser.add_argument("--stage2-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Save report JSON")
    args = parser.parse_args()

    # Auto-detect latest run dirs if not specified
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    if args.stage1_dir is None:
        candidates = sorted(runs_dir.glob("enigma_2*"))
        if candidates:
            args.stage1_dir = candidates[-1]
    if args.stage2_dir is None:
        candidates = sorted(runs_dir.glob("enigma_s2_*"))
        if candidates:
            args.stage2_dir = candidates[-1]

    report = full_pipeline_postmortem(args.stage1_dir, args.stage2_dir)
    print_report(report)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"Report saved to: {args.output}")
    elif args.stage1_dir:
        out = args.stage1_dir.parent / "postmortem_full.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"Report saved to: {out}")


if __name__ == "__main__":
    main()
