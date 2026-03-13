from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepTelemetry:
    """Per-step optimizer telemetry snapshot.

    Captures the internal state of the gated optimizer at each training step,
    enabling post-hoc analysis of *why* a candidate behaved the way it did.
    """
    step: int
    loss: float
    grad_norm: float
    # Gate and actuator state (stateful control)
    gate_value: float = 1.0
    update_multiplier: float = 1.0
    trust_ratio_mix: float = 1.0
    clip_threshold: float | None = None
    beta2_override: float | None = None
    orthogonal_mix: float = 1.0
    # Raw sensor signals (pre-normalization)
    loss_ema: float = 0.0
    loss_improvement_ema: float = 0.0
    grad_norm_ema: float = 0.0
    update_ratio_ema: float = 0.0
    grad_alignment_ema: float = 0.0
    step_fraction: float = 0.0
    # Update dynamics
    mean_update_param_ratio: float = 0.0
    max_update_param_ratio: float = 0.0
    mean_trust_ratio: float = 1.0


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
    telemetry: list[StepTelemetry] = field(default_factory=list)
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
class MutationOperatorStats:
    """Tracks how effective each mutation operator is across the tournament."""
    operator: str
    times_applied: int = 0
    times_survived: int = 0
    times_promoted: int = 0
    times_won: int = 0
    mean_score_delta: float = 0.0
    best_score_delta: float = 0.0
    best_candidate_id: str | None = None


@dataclass
class GenealogyNode:
    """One node in the lineage tree, for tracing winning paths."""
    candidate_id: str
    parent_id: str | None
    generation: int
    mutation: str
    status: str
    score: float
    children: list[str] = field(default_factory=list)


@dataclass
class TournamentAnalytics:
    """Rich analytics for a completed tournament run."""
    mutation_stats: list[MutationOperatorStats] = field(default_factory=list)
    genealogy: list[GenealogyNode] = field(default_factory=list)
    generation_diversity: list[dict[str, Any]] = field(default_factory=list)
    winning_lineage_paths: list[list[str]] = field(default_factory=list)


@dataclass
class TournamentSummary:
    run_dir: str
    baseline_candidate_id: str
    best_candidate_id: str
    total_candidates: int
    winners: list[str]
    pareto_frontier: list[str]
    analytics: TournamentAnalytics | None = None
