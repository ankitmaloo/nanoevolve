from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvaluationConfig:
    seed: int = 7
    steps: int = 24
    eval_every: int = 6
    batch_size: int = 8
    seq_len: int = 24
    vocab_size: int = 64
    train_eval_batches: int = 2
    val_eval_batches: int = 4
    model_dim: int = 32
    hidden_dim: int = 64
    layers: int = 2
    device: str = "cpu"
    grad_spike_factor: float = 4.0
    learning_rate_scale: float = 1.0

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def token_budget(self) -> int:
        return self.steps * self.tokens_per_step


@dataclass
class SearchConfig:
    generations: int = 3
    candidates_per_generation: int = 6
    survivor_top_k: int = 3
    promotion_top_k: int = 2
    improvement_threshold_bpb: float = 0.005
    min_time_to_target_ratio: float = 1.05
    max_slowdown_ratio: float = 1.35
    max_memory_ratio: float = 1.35
    min_tokens_per_sec_ratio: float = 0.95
    max_grad_spike_delta: int = 0
    max_stability_penalty_delta: float = 0.0
    min_seed_win_rate: float = 2.0 / 3.0
    promotion_seeds: tuple[int, ...] = (7, 13, 29)
    seed: int = 7
    run_name: str = "adamopt_search"
    out_dir: str | None = None

    def resolve_run_dir(self, root: Path) -> Path:
        if self.out_dir:
            return Path(self.out_dir).resolve()
        return (root / "runs" / self.run_name).resolve()


@dataclass
class ComparisonConfig:
    baseline_label: str = "baseline"
    candidate_label: str = "candidate"
    notes: list[str] = field(default_factory=list)


@dataclass
class AutonomousSearchConfig:
    candidate_count: int = 4
    provider: str = "codex"
    instruction_template: str = "Mutate NanoChat AdamW path for candidate {candidate_id}."
    scope: str = "adamw_math"
    command_template: str | None = None
    mutation_concurrency: int = 2
    validation_concurrency: int = 2
    deployment_concurrency: int = 2
    poll_concurrency: int = 8
    poll_interval_s: float = 5.0
    validation_disable_torch_compile: bool = True
    validation_python_executable: str | None = None
    run_name: str = "autonomous_search"
    out_dir: str | None = None

    def resolve_run_dir(self, root: Path) -> Path:
        if self.out_dir:
            return Path(self.out_dir).resolve()
        return (root / "runs" / self.run_name).resolve()
