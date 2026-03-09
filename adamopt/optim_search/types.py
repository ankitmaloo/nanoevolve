from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurvePoint:
    step: int
    train_bpb: float | None = None
    val_bpb: float | None = None
    tokens_seen: int = 0


@dataclass
class TrialMetrics:
    final_validation_bpb: float
    best_validation_bpb: float
    train_curve_auc: float
    validation_curve_auc: float
    mean_step_time_ms: float
    tokens_per_sec: float
    nan_failures: int
    inf_failures: int
    grad_norm_spikes: int
    max_grad_norm: float
    mean_update_param_ratio: float
    max_update_param_ratio: float
    memory_overhead_bytes: int
    stability_penalty: float


@dataclass
class EvaluationOutcome:
    candidate_id: str
    spec_name: str
    seed: int
    valid: bool
    metrics: TrialMetrics | None = None
    curve: list[CurvePoint] = field(default_factory=list)
    failure_type: str | None = None
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class WinAxisResult:
    name: str
    status: str
    value: float | None = None
    threshold: float | None = None
    detail: str | None = None


@dataclass
class WinAssessment:
    primary_metric: str
    winner: bool
    hierarchy_level: int
    dominant_axes: list[str] = field(default_factory=list)
    quality: WinAxisResult | None = None
    wallclock: WinAxisResult | None = None
    sample_efficiency: WinAxisResult | None = None
    stability: WinAxisResult | None = None
    throughput: WinAxisResult | None = None
    seed_robustness: WinAxisResult | None = None
    scaling: WinAxisResult | None = None
    tuning: WinAxisResult | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class PromotionResult:
    candidate_id: str
    winner: bool
    mean_final_validation_bpb: float
    mean_step_time_ms: float
    mean_tokens_per_sec: float
    mean_memory_overhead_bytes: float
    mean_stability_penalty: float
    mean_grad_norm_spikes: float
    baseline_mean_final_validation_bpb: float
    baseline_mean_step_time_ms: float
    baseline_mean_tokens_per_sec: float
    baseline_mean_memory_overhead_bytes: float
    baseline_mean_stability_penalty: float
    baseline_mean_grad_norm_spikes: float
    improvement_bpb: float
    speed_ratio: float
    tokens_per_sec_ratio: float
    memory_ratio: float
    time_to_target_ratio: float | None = None
    seed_win_rate: float = 0.0
    win_assessment: WinAssessment | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class CandidateRecord:
    id: str
    generation: int
    parent_id: str | None
    spec: dict[str, Any]
    score: float
    pareto: bool
    promoted: bool
    status: str
    primary_outcome: EvaluationOutcome
    promotion_result: PromotionResult | None = None
    win_assessment: WinAssessment | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    failure_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TournamentSummary:
    run_dir: str
    baseline_candidate_id: str
    best_candidate_id: str
    total_candidates: int
    winners: list[str]
    pareto_frontier: list[str]
