from __future__ import annotations

import math

from .config import SearchConfig
from .types import CandidateRecord, EvaluationOutcome, WinAssessment, WinAxisResult


def _first_step_at_or_below(outcome: EvaluationOutcome, target_bpb: float) -> int | None:
    for point in outcome.curve:
        if point.val_bpb is not None and point.val_bpb <= target_bpb:
            return point.step
    return None


def _time_to_target_ratio(outcome: EvaluationOutcome, baseline: EvaluationOutcome) -> float | None:
    if outcome.metrics is None or baseline.metrics is None:
        return None
    target_bpb = baseline.metrics.final_validation_bpb
    baseline_step = _first_step_at_or_below(baseline, target_bpb)
    candidate_step = _first_step_at_or_below(outcome, target_bpb)
    if baseline_step is None or candidate_step is None:
        return None
    baseline_time = baseline_step * baseline.metrics.mean_step_time_ms
    candidate_time = candidate_step * outcome.metrics.mean_step_time_ms
    if candidate_time <= 0:
        return None
    return baseline_time / candidate_time


def analyze_win_hierarchy(
    outcome: EvaluationOutcome,
    baseline: EvaluationOutcome,
    config: SearchConfig | None = None,
    *,
    seed_win_rate: float | None = None,
    allow_multi_seed_hierarchy: bool = False,
) -> WinAssessment:
    config = config or SearchConfig()
    if not outcome.valid or outcome.metrics is None or baseline.metrics is None:
        return WinAssessment(
            primary_metric="time_to_target_validation_bpb",
            winner=False,
            hierarchy_level=0,
            quality=WinAxisResult(name="quality", status="loss", detail="invalid_or_missing_metrics"),
            wallclock=WinAxisResult(name="wallclock", status="loss", detail="invalid_or_missing_metrics"),
            sample_efficiency=WinAxisResult(name="sample_efficiency", status="loss", detail="invalid_or_missing_metrics"),
            stability=WinAxisResult(name="stability", status="loss", detail="invalid_or_missing_metrics"),
            throughput=WinAxisResult(name="throughput", status="loss", detail="invalid_or_missing_metrics"),
            seed_robustness=WinAxisResult(name="seed_robustness", status="inconclusive", detail="not_evaluated"),
            scaling=WinAxisResult(name="scaling", status="inconclusive", detail="not_evaluated"),
            tuning=WinAxisResult(name="tuning", status="inconclusive", detail="not_evaluated"),
            notes=["candidate_invalid"],
        )

    improvement_bpb = baseline.metrics.final_validation_bpb - outcome.metrics.final_validation_bpb
    speed_ratio = baseline.metrics.mean_step_time_ms / max(outcome.metrics.mean_step_time_ms, 1e-8)
    tokens_per_sec_ratio = outcome.metrics.tokens_per_sec / max(baseline.metrics.tokens_per_sec, 1e-8)
    memory_ratio = outcome.metrics.memory_overhead_bytes / max(float(baseline.metrics.memory_overhead_bytes), 1.0)
    stability_penalty_delta = outcome.metrics.stability_penalty - baseline.metrics.stability_penalty
    grad_spike_delta = outcome.metrics.grad_norm_spikes - baseline.metrics.grad_norm_spikes
    time_to_target_ratio = _time_to_target_ratio(outcome, baseline)

    quality_status = "win" if improvement_bpb >= config.improvement_threshold_bpb else ("tie" if improvement_bpb >= 0 else "loss")
    wallclock_status = "win" if time_to_target_ratio is not None and time_to_target_ratio >= config.min_time_to_target_ratio else "loss"
    throughput_status = "win" if tokens_per_sec_ratio >= config.min_tokens_per_sec_ratio else "loss"
    stability_ok = (
        speed_ratio >= 1.0 / config.max_slowdown_ratio
        and memory_ratio <= config.max_memory_ratio
        and grad_spike_delta <= config.max_grad_spike_delta
        and stability_penalty_delta <= config.max_stability_penalty_delta
        and outcome.metrics.nan_failures == 0
        and outcome.metrics.inf_failures == 0
    )
    stability_status = "win" if stability_ok else "loss"

    if seed_win_rate is None:
        seed_axis = WinAxisResult(name="seed_robustness", status="inconclusive", detail="not_evaluated")
    else:
        seed_axis = WinAxisResult(
            name="seed_robustness",
            status="win" if seed_win_rate >= config.min_seed_win_rate else "loss",
            value=seed_win_rate,
            threshold=config.min_seed_win_rate,
            detail="fraction_of_promotion_seeds_meeting_win_constraints",
        )

    dominant_axes: list[str] = []
    notes: list[str] = []
    if quality_status == "win":
        dominant_axes.append("sample_efficiency")
    else:
        notes.append("sample_efficiency_below_threshold")
    if wallclock_status == "win":
        dominant_axes.append("wallclock")
    elif time_to_target_ratio is None:
        notes.append("time_to_target_not_reached")
    else:
        notes.append("wallclock_not_improved")
    if stability_status == "win":
        dominant_axes.append("stability")
    else:
        if memory_ratio > config.max_memory_ratio:
            notes.append("memory_regression")
        if speed_ratio < 1.0 / config.max_slowdown_ratio:
            notes.append("slowdown_regression")
        if grad_spike_delta > config.max_grad_spike_delta:
            notes.append("grad_spike_regression")
        if stability_penalty_delta > config.max_stability_penalty_delta:
            notes.append("stability_penalty_regression")
        if outcome.metrics.nan_failures or outcome.metrics.inf_failures:
            notes.append("non_finite_failure")
    if throughput_status == "win":
        dominant_axes.append("throughput_adjusted_efficiency")
    else:
        notes.append("throughput_regression")

    hierarchy_level = 0
    if quality_status == "win":
        hierarchy_level = max(hierarchy_level, 2)
    if allow_multi_seed_hierarchy and seed_axis.status == "win":
        hierarchy_level = max(hierarchy_level, 3)
    if allow_multi_seed_hierarchy and wallclock_status == "win":
        hierarchy_level = max(hierarchy_level, 5)
    scaling_axis = WinAxisResult(name="scaling", status="inconclusive", detail="not_evaluated")
    tuning_axis = WinAxisResult(name="tuning", status="inconclusive", detail="not_evaluated")
    if quality_status == "win" and stability_status == "win" and throughput_status == "win":
        notes.append("candidate_meets_short_horizon_constraints")

    winner = (
        quality_status == "win"
        and stability_status == "win"
        and throughput_status == "win"
        and (seed_axis.status != "loss")
    )
    if allow_multi_seed_hierarchy:
        winner = winner and seed_axis.status == "win"

    return WinAssessment(
        primary_metric="time_to_target_validation_bpb",
        winner=winner,
        hierarchy_level=hierarchy_level,
        dominant_axes=dominant_axes,
        quality=WinAxisResult(
            name="quality",
            status=quality_status,
            value=improvement_bpb,
            threshold=config.improvement_threshold_bpb,
            detail="baseline_final_validation_bpb - candidate_final_validation_bpb",
        ),
        wallclock=WinAxisResult(
            name="wallclock",
            status=wallclock_status,
            value=time_to_target_ratio,
            threshold=config.min_time_to_target_ratio,
            detail="baseline_time_to_target / candidate_time_to_target",
        ),
        sample_efficiency=WinAxisResult(
            name="sample_efficiency",
            status=quality_status,
            value=improvement_bpb,
            threshold=config.improvement_threshold_bpb,
            detail="quality_at_fixed_compute",
        ),
        stability=WinAxisResult(
            name="stability",
            status=stability_status,
            value=stability_penalty_delta,
            threshold=config.max_stability_penalty_delta,
            detail="candidate_stability_penalty - baseline_stability_penalty",
        ),
        throughput=WinAxisResult(
            name="throughput",
            status=throughput_status,
            value=tokens_per_sec_ratio,
            threshold=config.min_tokens_per_sec_ratio,
            detail="candidate_tokens_per_sec / baseline_tokens_per_sec",
        ),
        seed_robustness=seed_axis,
        scaling=scaling_axis,
        tuning=tuning_axis,
        notes=notes,
    )


def composite_score(outcome: EvaluationOutcome, baseline: EvaluationOutcome, config: SearchConfig | None = None) -> float:
    config = config or SearchConfig()
    if not outcome.valid or outcome.metrics is None or baseline.metrics is None:
        return -1e9

    assessment = analyze_win_hierarchy(outcome, baseline, config)
    quality_gain = baseline.metrics.final_validation_bpb - outcome.metrics.final_validation_bpb
    best_quality_gain = baseline.metrics.best_validation_bpb - outcome.metrics.best_validation_bpb
    auc_gain = baseline.metrics.validation_curve_auc - outcome.metrics.validation_curve_auc
    speed_term = math.log(max(baseline.metrics.mean_step_time_ms, 1e-8) / max(outcome.metrics.mean_step_time_ms, 1e-8))
    throughput_term = math.log(max(outcome.metrics.tokens_per_sec, 1e-8) / max(baseline.metrics.tokens_per_sec, 1e-8))
    memory_penalty = max(
        0.0,
        (outcome.metrics.memory_overhead_bytes - baseline.metrics.memory_overhead_bytes)
        / max(float(baseline.metrics.memory_overhead_bytes), 1.0),
    )
    stability_penalty = max(0.0, outcome.metrics.stability_penalty - baseline.metrics.stability_penalty)
    time_to_target_bonus = 0.0
    if assessment.wallclock and assessment.wallclock.value is not None:
        time_to_target_bonus = max(0.0, assessment.wallclock.value - 1.0)

    axis_bonus = 2.5 * len(assessment.dominant_axes)
    return (
        100.0 * quality_gain
        + 20.0 * best_quality_gain
        + 10.0 * auc_gain
        + 12.0 * time_to_target_bonus
        + 5.0 * speed_term
        + 4.0 * throughput_term
        + axis_bonus
        - 6.0 * memory_penalty
        - 5.0 * stability_penalty
    )


def pareto_objectives(outcome: EvaluationOutcome, baseline: EvaluationOutcome) -> tuple[float, float, float]:
    if not outcome.valid or outcome.metrics is None or baseline.metrics is None:
        return (-1e9, -1e9, -1e9)
    quality = baseline.metrics.final_validation_bpb - outcome.metrics.final_validation_bpb
    speed = outcome.metrics.tokens_per_sec / max(baseline.metrics.tokens_per_sec, 1e-8)
    stability = -(outcome.metrics.stability_penalty - baseline.metrics.stability_penalty)
    return (quality, speed, stability)


def dominates(left: CandidateRecord, right: CandidateRecord, baseline: EvaluationOutcome) -> bool:
    lo = pareto_objectives(left.primary_outcome, baseline)
    ro = pareto_objectives(right.primary_outcome, baseline)
    return all(l >= r for l, r in zip(lo, ro)) and any(l > r for l, r in zip(lo, ro))


def pareto_frontier(records: list[CandidateRecord], baseline: EvaluationOutcome) -> list[CandidateRecord]:
    frontier: list[CandidateRecord] = []
    for record in records:
        if any(dominates(other, record, baseline) for other in records if other.id != record.id):
            continue
        frontier.append(record)
    return frontier
