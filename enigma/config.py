"""Task and search configuration."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TaskConfig:
    """Everything Enigma needs to evolve a target."""
    name: str
    description: str
    seed_file: str
    mutable_files: list[str]
    context_files: list[str] = field(default_factory=list)
    eval_command: str | None = None
    eval_mode: str = "command"
    metric_keys: list[str] = field(default_factory=lambda: ["aggregate_score"])
    maximize: bool = True
    primary_metric: str = "aggregate_score"
    eval_cwd: str | None = None
    eval_timeout: int = 120
    eval_env: dict[str, str] = field(default_factory=dict)

    # Enigma-specific: language hint for context generation
    language: str = "auto"
    # Hardware description for context-aware prompting
    hardware: str = ""
    # Known constraints the LLM should respect
    hard_constraints: list[str] = field(default_factory=list)
    # Benchmark slices to evaluate separately
    benchmark_slices: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, task_dir: Path) -> TaskConfig:
        task_file = task_dir / "task.json"
        if not task_file.exists():
            raise FileNotFoundError(
                f"No task.json found in {task_dir}. "
                "Create one to tell Enigma how to evolve this folder."
            )
        raw = json.loads(task_file.read_text())
        return cls(
            name=raw["name"],
            description=raw["description"],
            seed_file=raw["seed_file"],
            mutable_files=raw.get("mutable_files", [raw["seed_file"]]),
            context_files=raw.get("context_files", []),
            eval_command=raw.get("eval_command"),
            eval_mode=raw.get("eval_mode", "command" if raw.get("eval_command") else "command"),
            metric_keys=raw.get("metric_keys", ["aggregate_score"]),
            maximize=raw.get("maximize", True),
            primary_metric=raw.get("primary_metric", "aggregate_score"),
            eval_cwd=raw.get("eval_cwd"),
            eval_timeout=raw.get("eval_timeout", 120),
            eval_env=raw.get("eval_env", {}),
            language=raw.get("language", "auto"),
            hardware=raw.get("hardware", ""),
            hard_constraints=raw.get("hard_constraints", []),
            benchmark_slices=raw.get("benchmark_slices", []),
        )

    def resolve_seed_path(self, task_dir: Path) -> Path:
        return (task_dir / self.seed_file).resolve()

    def resolve_context_paths(self, task_dir: Path) -> list[Path]:
        return [(task_dir / f).resolve() for f in self.context_files]

    def resolve_mutable_paths(self, task_dir: Path) -> list[Path]:
        return [(task_dir / f).resolve() for f in self.mutable_files]


@dataclass(frozen=True)
class SearchConfig:
    """Controls the evolution search parameters."""
    max_loops: int = 20
    candidates_per_slot: int = 1
    portfolio_size: int = 5
    hypotheses_per_loop: int = 10
    survivor_top_k: int = 6
    diversity_slots: int = 2
    seed: int = 42
    parallel_candidates: int = 4
    llm_concurrency: int = 2

    # LLM provider config
    provider: str = "anthropic"  # "anthropic", "openai", "gemini"
    model_name: str = "claude-sonnet-4-20250514"
    fast_model_name: str | None = None  # for fast screening stages
    temperature: float = 0.7
    max_tokens: int = 4096
    request_timeout_s: int = 120

    # Context generation
    auto_context: bool = True  # auto-generate context from code analysis
    max_context_tokens: int = 8000  # budget for dynamic context

    # Evaluation
    eval_retries: int = 1
    eval_timeout_multiplier: float = 1.5  # for promotion runs


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider: str
    model_name: str
    api_key_env: str = ""  # env var name for API key
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    request_timeout_s: int = 120
