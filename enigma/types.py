"""Core frozen dataclasses for the Enigma evolution engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    SHORTLISTED = "shortlisted"
    ACTIVE = "active"
    TESTED = "tested"
    KILLED = "killed"
    PROMOTED = "promoted"


class ExperimentRole(str, Enum):
    SCOUT = "scout"
    EXPLOIT = "exploit"
    ABLATION = "ablation"
    WILDCARD = "wildcard"


class SurfaceStatus(str, Enum):
    UNEXPLORED = "unexplored"
    SCOUTED = "scouted"
    EXPLOITED = "exploited"
    RETIRED = "retired"


@dataclass(frozen=True)
class StageResult:
    name: str
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class EvaluationResult:
    valid: bool
    aggregate_score: float
    metrics: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)
    stage_results: list[StageResult] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiffProposal:
    raw_diff: str
    model: str
    stage: str  # which doctrine stage produced this
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    id: str
    generation: int
    loop: int  # which doctrine loop
    parent_id: str | None
    source: str
    aggregate_score: float
    metrics: dict[str, float]
    failure_reasons: list[str]
    stage_results: list[StageResult]
    hypothesis_id: str | None = None
    portfolio_slot: str | None = None
    active: bool = True
    lineage: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttackSurface:
    """A place where a code change could plausibly move the objective."""
    id: str
    region: str
    bottleneck: str
    change_class: str  # structural, policy, state, schedule
    mechanism: str
    leverage: int  # 1-5
    plausibility: int  # 1-5
    observability: int  # 1-5
    implementation_cost: int  # 1-5
    overlap_risk: int  # 1-5
    status: SurfaceStatus = SurfaceStatus.UNEXPLORED
    notes: str = ""


@dataclass(frozen=True)
class Hypothesis:
    """A testable claim about how to improve the target program."""
    id: str
    title: str
    status: HypothesisStatus
    family: str
    bottleneck_attacked: str
    mechanism: str
    code_change_class: str
    expected_win: str
    main_risk: str
    evidence_needed: str
    disproof_signal: str
    cheapest_test: str
    applies_when: str = ""
    avoid_when: str = ""
    parent_hypothesis: str | None = None
    novelty_vs_prior: str = ""
    upside: int = 3
    feasibility: int = 3
    distinctness: int = 3
    information_gain: int = 3
    transferability: int = 3
    notes: str = ""


@dataclass(frozen=True)
class PortfolioSlot:
    """One slot in the execution portfolio."""
    id: str  # P1, P2, ...
    role: ExperimentRole
    hypothesis_id: str
    family: str
    target_region: str
    operator_family: str
    why_selected: str
    expected_signal: str
    acceptance_test: str
    kill_condition: str
    overlap_check: str
    result: str = "pending"


@dataclass(frozen=True)
class MutationNeighborhood:
    """Tracks a mutation neighborhood for overlap detection."""
    id: str
    loop: int
    hypothesis_id: str
    target_region: str
    operator_family: str
    bottleneck_attacked: str
    benchmark_slice: str
    parent_candidate: str | None = None
    relation_to_prior: str = ""
    overlap_status: str = "unique"
    why_allowed: str = ""
    outcome: str = "pending"
    retired: bool = False
    reopen_condition: str = ""


@dataclass(frozen=True)
class NegativeKnowledge:
    """A durable lesson from a failed experiment."""
    id: str
    move_family: str
    observed_failure: str
    likely_cause: str
    evidence: str
    confidence: str  # "high", "medium", "low"
    do_not_repeat_unless: str
    revisit_trigger: str
    related_hypotheses: list[str] = field(default_factory=list)
    source_loops: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class GapEntry:
    """An uncovered area of the search space."""
    id: str
    uncovered_area: str
    why_it_matters: str
    current_portfolio_miss: str
    candidate_hypothesis: str
    expected_information_gain: str
    missing_evidence: str
    urgency: str  # "high", "medium", "low"


@dataclass
class LoopRecord:
    """Record of one completed doctrine loop."""
    loop: int
    date: str
    target: str
    parent_baseline: str
    baseline_metrics: dict[str, float]
    active_portfolio: list[str]
    overlap_conflicts: list[str]
    best_candidate: str
    worst_candidate: str
    score_movement: str
    holdout_movement: str
    stability_movement: str
    what_improved: str
    what_regressed: str
    strongest_evidence: str
    likely_causal_explanation: str
    hypotheses_promoted: list[str]
    hypotheses_killed: list[str]
    child_hypotheses: list[str]
    negative_knowledge_added: list[str]
    neighborhoods_retired: list[str]
    neighborhoods_reopened: list[str]
    next_loop_focus: str


@dataclass
class FeedbackBundle:
    best_metrics: dict[str, float] = field(default_factory=dict)
    weak_failure_reasons: list[str] = field(default_factory=list)
    dropped_notes: list[str] = field(default_factory=list)
    negative_knowledge: list[str] = field(default_factory=list)
    active_surfaces: list[str] = field(default_factory=list)
