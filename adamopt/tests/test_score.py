from __future__ import annotations

from adamopt.optim_search.config import SearchConfig
from adamopt.optim_search.score import analyze_win_hierarchy, composite_score
from adamopt.optim_search.types import CurvePoint, EvaluationOutcome, TrialMetrics


def _make_outcome(
    *,
    candidate_id: str,
    final_validation_bpb: float,
    best_validation_bpb: float,
    validation_curve_auc: float,
    mean_step_time_ms: float,
    tokens_per_sec: float,
    grad_norm_spikes: int = 0,
    stability_penalty: float = 0.0,
    memory_overhead_bytes: int = 1024,
    val_curve: list[tuple[int, float]] | None = None,
) -> EvaluationOutcome:
    curve = [
        CurvePoint(step=step, train_bpb=value + 0.2, val_bpb=value, tokens_seen=step * 100)
        for step, value in (val_curve or [(6, final_validation_bpb), (12, final_validation_bpb)])
    ]
    metrics = TrialMetrics(
        final_validation_bpb=final_validation_bpb,
        best_validation_bpb=best_validation_bpb,
        train_curve_auc=validation_curve_auc + 0.5,
        validation_curve_auc=validation_curve_auc,
        mean_step_time_ms=mean_step_time_ms,
        tokens_per_sec=tokens_per_sec,
        nan_failures=0,
        inf_failures=0,
        grad_norm_spikes=grad_norm_spikes,
        max_grad_norm=1.0,
        mean_update_param_ratio=0.01,
        max_update_param_ratio=0.02,
        memory_overhead_bytes=memory_overhead_bytes,
        stability_penalty=stability_penalty,
    )
    return EvaluationOutcome(
        candidate_id=candidate_id,
        spec_name=candidate_id,
        seed=7,
        valid=True,
        metrics=metrics,
        curve=curve,
    )


def test_analyze_win_hierarchy_marks_wallclock_and_seed_robust_winner() -> None:
    config = SearchConfig(
        improvement_threshold_bpb=0.005,
        min_time_to_target_ratio=1.05,
        min_tokens_per_sec_ratio=0.95,
        min_seed_win_rate=2.0 / 3.0,
    )
    baseline = _make_outcome(
        candidate_id="baseline",
        final_validation_bpb=1.20,
        best_validation_bpb=1.19,
        validation_curve_auc=5.0,
        mean_step_time_ms=100.0,
        tokens_per_sec=1000.0,
        val_curve=[(6, 1.30), (12, 1.20), (18, 1.20)],
    )
    candidate = _make_outcome(
        candidate_id="candidate",
        final_validation_bpb=1.18,
        best_validation_bpb=1.17,
        validation_curve_auc=4.8,
        mean_step_time_ms=90.0,
        tokens_per_sec=1125.0,
        val_curve=[(6, 1.19), (12, 1.18), (18, 1.18)],
    )

    assessment = analyze_win_hierarchy(
        candidate,
        baseline,
        config,
        seed_win_rate=1.0,
        allow_multi_seed_hierarchy=True,
    )

    assert assessment.winner is True
    assert assessment.hierarchy_level == 5
    assert "sample_efficiency" in assessment.dominant_axes
    assert "wallclock" in assessment.dominant_axes
    assert assessment.seed_robustness is not None
    assert assessment.seed_robustness.status == "win"


def test_analyze_win_hierarchy_rejects_throughput_regression() -> None:
    config = SearchConfig(min_tokens_per_sec_ratio=0.95)
    baseline = _make_outcome(
        candidate_id="baseline",
        final_validation_bpb=1.20,
        best_validation_bpb=1.19,
        validation_curve_auc=5.0,
        mean_step_time_ms=100.0,
        tokens_per_sec=1000.0,
    )
    candidate = _make_outcome(
        candidate_id="candidate",
        final_validation_bpb=1.18,
        best_validation_bpb=1.17,
        validation_curve_auc=4.9,
        mean_step_time_ms=140.0,
        tokens_per_sec=700.0,
    )

    assessment = analyze_win_hierarchy(candidate, baseline, config)

    assert assessment.winner is False
    assert assessment.throughput is not None
    assert assessment.throughput.status == "loss"
    assert "throughput_regression" in assessment.notes


def test_composite_score_prefers_balanced_candidate_over_slow_peacock() -> None:
    config = SearchConfig()
    baseline = _make_outcome(
        candidate_id="baseline",
        final_validation_bpb=1.20,
        best_validation_bpb=1.19,
        validation_curve_auc=5.0,
        mean_step_time_ms=100.0,
        tokens_per_sec=1000.0,
    )
    balanced = _make_outcome(
        candidate_id="balanced",
        final_validation_bpb=1.19,
        best_validation_bpb=1.18,
        validation_curve_auc=4.9,
        mean_step_time_ms=95.0,
        tokens_per_sec=1050.0,
    )
    peacock = _make_outcome(
        candidate_id="peacock",
        final_validation_bpb=1.185,
        best_validation_bpb=1.18,
        validation_curve_auc=4.85,
        mean_step_time_ms=180.0,
        tokens_per_sec=550.0,
        memory_overhead_bytes=4096,
        stability_penalty=3.0,
        grad_norm_spikes=3,
    )

    assert composite_score(balanced, baseline, config) > composite_score(peacock, baseline, config)
