from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageResult:
    name: str
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass
class EvaluationResult:
    valid: bool
    aggregate_score: float
    metrics: dict[str, float] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)
    stage_results: list[StageResult] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffProposal:
    raw_diff: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    id: str
    generation: int
    parent_id: str | None
    source: str
    aggregate_score: float
    metrics: dict[str, float]
    failure_reasons: list[str]
    stage_results: list[StageResult]
    active: bool = True
    lineage: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackBundle:
    best_metrics: dict[str, float] = field(default_factory=dict)
    weak_failure_reasons: list[str] = field(default_factory=list)
    dropped_notes: list[str] = field(default_factory=list)


@dataclass
class GenerationRecord:
    generation: int
    parent_id: str
    child_id: str
    metric_key: str
    prompt_id: str
    prompt_reward: float
    candidate_valid: bool
    survivor_ids: list[str]
    dropped_ids: list[str]
    prompt_file: str
    diff_file: str
    evaluation_file: str
