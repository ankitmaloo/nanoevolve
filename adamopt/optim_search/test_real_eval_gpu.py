"""
GPU integration tests for real_eval.py.

No container dependency — runs anywhere with CUDA + nanochat installed.
Call run_all_tests() from any GPU environment (Modal, Azure, bare metal).

Returns (passed: bool, report: str).
"""
from __future__ import annotations

import traceback

from .real_eval import RealEvalConfig, evaluate_real_nanochat
from .spec import MatrixOptimizerSpec


def run_all_tests(
    *,
    steps: int = 20,
    depth: int = 4,
    max_seq_len: int = 512,
    device_batch_size: int = 2,
    total_batch_size: int = 1024,
    nanochat_base_dir: str = "/data/nanochat",
    device: str = "cuda",
) -> tuple[bool, str]:
    """Run all real-eval GPU tests. Returns (passed, report)."""
    lines: list[str] = []
    failed = False

    config = RealEvalConfig(
        steps=steps,
        eval_every=max(1, steps // 2),
        depth=depth,
        max_seq_len=max_seq_len,
        device_batch_size=device_batch_size,
        total_batch_size=total_batch_size,
        device=device,
        nanochat_base_dir=nanochat_base_dir,
    )

    # ---- Test 1: Baseline spec produces valid outcome ----
    lines.append("=" * 60)
    lines.append(f"TEST 1: evaluate_real_nanochat ({steps} steps, baseline spec)")
    lines.append("=" * 60)

    outcome = None
    try:
        spec = MatrixOptimizerSpec.baseline_nanochat()
        outcome = evaluate_real_nanochat(spec, config, candidate_id="test_baseline_001")
    except Exception as exc:
        lines.append(f"  FAILED: {type(exc).__name__}: {exc}")
        lines.append(traceback.format_exc())
        failed = True

    if outcome is not None:
        lines.append(f"  valid:                  {outcome.valid}")
        lines.append(f"  failure_type:           {outcome.failure_type}")
        lines.append(f"  curve points:           {len(outcome.curve)}")
        lines.append(f"  telemetry snapshots:    {len(outcome.telemetry)}")

        if not outcome.valid:
            lines.append(f"  FAILED: outcome not valid — {outcome.failure_type}")
            failed = True

        if outcome.metrics is None:
            lines.append("  FAILED: no metrics")
            failed = True
        else:
            m = outcome.metrics
            lines.append(f"  final_validation_bpb:   {m.final_validation_bpb:.4f}")
            lines.append(f"  best_validation_bpb:    {m.best_validation_bpb:.4f}")
            lines.append(f"  train_curve_auc:        {m.train_curve_auc:.4f}")
            lines.append(f"  mean_step_time_ms:      {m.mean_step_time_ms:.1f}")
            lines.append(f"  tokens_per_sec:         {m.tokens_per_sec:.0f}")
            lines.append(f"  nan_failures:           {m.nan_failures}")
            lines.append(f"  inf_failures:           {m.inf_failures}")
            lines.append(f"  grad_norm_spikes:       {m.grad_norm_spikes}")
            lines.append(f"  mean_update_param_ratio:{m.mean_update_param_ratio:.6f}")
            lines.append(f"  memory_overhead_bytes:  {m.memory_overhead_bytes}")
            lines.append(f"  stability_penalty:      {m.stability_penalty}")

            if m.final_validation_bpb <= 0:
                lines.append("  FAILED: final_validation_bpb <= 0")
                failed = True
            if m.mean_step_time_ms <= 0:
                lines.append("  FAILED: mean_step_time_ms <= 0")
                failed = True
            if m.memory_overhead_bytes <= 0:
                lines.append("  FAILED: memory_overhead_bytes <= 0")
                failed = True

    lines.append("")

    # ---- Test 2: Telemetry data at every step ----
    lines.append("=" * 60)
    lines.append("TEST 2: Telemetry data at every step")
    lines.append("=" * 60)

    if outcome is None or not outcome.valid:
        lines.append("  SKIPPED: outcome invalid, cannot check telemetry")
        failed = True
    else:
        telem = outcome.telemetry
        lines.append(f"  Total telemetry snapshots: {len(telem)}")

        if len(telem) != config.steps:
            lines.append(f"  FAILED: expected {config.steps} snapshots, got {len(telem)}")
            failed = True
        else:
            lines.append(f"  Count matches steps:      PASS")

        bad_snapshots = []
        for snap in telem:
            issues = []
            if snap.loss <= 0:
                issues.append("loss <= 0")
            if snap.grad_norm < 0:
                issues.append("grad_norm < 0")
            if not (0.0 <= snap.gate_value <= 1.0):
                issues.append(f"gate_value={snap.gate_value} out of [0,1]")
            if snap.update_multiplier <= 0:
                issues.append(f"update_multiplier={snap.update_multiplier} <= 0")
            if issues:
                bad_snapshots.append((snap.step, issues))

        if bad_snapshots:
            lines.append(f"  FAILED: {len(bad_snapshots)} bad snapshots:")
            for step_num, issues in bad_snapshots[:5]:
                lines.append(f"    step {step_num}: {', '.join(issues)}")
            failed = True
        else:
            lines.append(f"  All snapshots valid:      PASS")

        fracs = [s.step_fraction for s in telem]
        if fracs[-1] <= fracs[0]:
            lines.append(f"  FAILED: step_fraction not increasing ({fracs[0]:.4f} -> {fracs[-1]:.4f})")
            failed = True
        else:
            lines.append(f"  step_fraction increasing: PASS ({fracs[0]:.4f} -> {fracs[-1]:.4f})")

        if telem[-1].loss_ema <= 0:
            lines.append(f"  FAILED: loss_ema not populated at final step")
            failed = True
        else:
            lines.append(f"  loss_ema populated:       PASS ({telem[-1].loss_ema:.4f})")

        # Print first and last snapshot
        lines.append("")
        s = telem[0]
        lines.append("  First snapshot:")
        lines.append(f"    step={s.step} loss={s.loss:.4f} grad_norm={s.grad_norm:.4f} "
                     f"gate={s.gate_value:.4f} update_mult={s.update_multiplier:.4f}")
        lines.append(f"    loss_ema={s.loss_ema:.4f} grad_norm_ema={s.grad_norm_ema:.4f} "
                     f"step_frac={s.step_fraction:.4f}")

        s = telem[-1]
        lines.append("  Last snapshot:")
        lines.append(f"    step={s.step} loss={s.loss:.4f} grad_norm={s.grad_norm:.4f} "
                     f"gate={s.gate_value:.4f} update_mult={s.update_multiplier:.4f}")
        lines.append(f"    loss_ema={s.loss_ema:.4f} grad_norm_ema={s.grad_norm_ema:.4f} "
                     f"step_frac={s.step_fraction:.4f}")

    lines.append("")

    # ---- Test 3: Stateful spec (gate should vary) ----
    lines.append("=" * 60)
    lines.append(f"TEST 3: evaluate_real_nanochat ({steps} steps, stateful spec)")
    lines.append("=" * 60)

    try:
        spec_stateful = MatrixOptimizerSpec.stateful_annealing_variant()
        outcome_stateful = evaluate_real_nanochat(spec_stateful, config, candidate_id="test_stateful_001")
    except Exception as exc:
        lines.append(f"  FAILED: {type(exc).__name__}: {exc}")
        lines.append(traceback.format_exc())
        failed = True
        outcome_stateful = None

    if outcome_stateful is not None and outcome_stateful.valid:
        lines.append(f"  valid:                  {outcome_stateful.valid}")
        telem_s = outcome_stateful.telemetry
        gates = [s.gate_value for s in telem_s]
        lines.append(f"  gate range:             [{min(gates):.4f}, {max(gates):.4f}]")

        if outcome_stateful.metrics:
            ms = outcome_stateful.metrics
            lines.append(f"  final_validation_bpb:   {ms.final_validation_bpb:.4f}")
            lines.append(f"  mean_step_time_ms:      {ms.mean_step_time_ms:.1f}")

        if len(telem_s) != config.steps:
            lines.append(f"  FAILED: expected {config.steps} telemetry snapshots, got {len(telem_s)}")
            failed = True
        else:
            lines.append(f"  Telemetry count:        PASS")
    elif outcome_stateful is not None:
        lines.append(f"  FAILED: outcome not valid — {outcome_stateful.failure_type}")
        failed = True

    lines.append("")

    # ---- Summary ----
    lines.append("=" * 60)
    if failed:
        lines.append("REAL EVAL TESTS: FAILED")
    else:
        lines.append("REAL EVAL TESTS: ALL PASSED")
    lines.append("=" * 60)

    report = "\n".join(lines)
    return (not failed, report)
