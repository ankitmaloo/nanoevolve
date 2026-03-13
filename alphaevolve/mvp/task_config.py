"""
Generic task configuration for alphaevolve.

Any target folder can be evolved by providing a `task.json` file
that tells alphaevolve:
  - which files the LLM can mutate
  - which files provide read-only context
  - how to evaluate a candidate (a shell command that returns JSON)
  - a human-readable description for the LLM prompt

Example task.json:
{
    "name": "optimize_a_star_heuristic",
    "description": "Evolve the priority and tie-break functions for A* routing.",
    "seed_file": "astar_routing_target.py",
    "mutable_files": ["astar_routing_target.py"],
    "context_files": [],
    "eval_command": null,
    "eval_mode": "builtin_astar",
    "metric_keys": ["aggregate_score", "solved_ratio", "path_quality"],
    "maximize": true,
    "primary_metric": "aggregate_score"
}

For external evaluation (e.g. a CUDA kernel):
{
    "name": "optimize_softmax_kernel",
    "description": "Evolve the softmax CUDA kernel for better throughput.",
    "seed_file": "softmax.cu",
    "mutable_files": ["softmax.cu"],
    "context_files": ["softmax.h", "bench.py"],
    "eval_command": "bash eval.sh {candidate_file}",
    "eval_mode": "command",
    "metric_keys": ["throughput_gbps", "correctness"],
    "maximize": true,
    "primary_metric": "throughput_gbps"
}

eval_command must print a JSON object to stdout:
{
    "valid": true,
    "aggregate_score": 124.5,
    "metrics": {"throughput_gbps": 124.5, "correctness": 1.0},
    "failure_reasons": []
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TaskConfig:
    """Parsed task.json — everything alphaevolve needs to evolve a target."""

    name: str
    description: str
    seed_file: str
    mutable_files: list[str]
    context_files: list[str] = field(default_factory=list)
    eval_command: str | None = None
    eval_mode: str = "builtin_astar"  # "builtin_astar" | "builtin_binpack" | "command"
    metric_keys: list[str] = field(default_factory=lambda: ["aggregate_score"])
    maximize: bool = True
    primary_metric: str = "aggregate_score"

    # Optional: working directory for eval_command (relative to task_dir)
    eval_cwd: str | None = None
    # Optional: timeout for eval_command in seconds
    eval_timeout: int = 120
    # Optional: environment variables for eval_command
    eval_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, task_dir: Path) -> "TaskConfig":
        """Load task.json from a directory."""
        task_file = task_dir / "task.json"
        if not task_file.exists():
            raise FileNotFoundError(
                f"No task.json found in {task_dir}. "
                "Create one to tell alphaevolve how to evolve this folder."
            )
        raw = json.loads(task_file.read_text())
        return cls(
            name=raw["name"],
            description=raw["description"],
            seed_file=raw["seed_file"],
            mutable_files=raw.get("mutable_files", [raw["seed_file"]]),
            context_files=raw.get("context_files", []),
            eval_command=raw.get("eval_command"),
            eval_mode=raw.get("eval_mode", "command" if raw.get("eval_command") else "builtin_astar"),
            metric_keys=raw.get("metric_keys", ["aggregate_score"]),
            maximize=raw.get("maximize", True),
            primary_metric=raw.get("primary_metric", "aggregate_score"),
            eval_cwd=raw.get("eval_cwd"),
            eval_timeout=raw.get("eval_timeout", 120),
            eval_env=raw.get("eval_env", {}),
        )

    def resolve_seed_path(self, task_dir: Path) -> Path:
        return (task_dir / self.seed_file).resolve()

    def resolve_context_paths(self, task_dir: Path) -> list[Path]:
        return [(task_dir / f).resolve() for f in self.context_files]

    def resolve_mutable_paths(self, task_dir: Path) -> list[Path]:
        return [(task_dir / f).resolve() for f in self.mutable_files]
