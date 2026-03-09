from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunConfig:
    mode: str = "mock"
    model_name: str = "gemini-3-flash-lite"
    openai_fast_model_name: str = "gpt-5.2-mini"
    openai_slow_every: int = 4
    openai_request_timeout_s: float = 45.0
    openai_max_retries: int = 1
    openai_max_output_tokens: int = 2500
    parallel_candidates: int = 3
    llm_concurrency: int = 3
    generations: int = 10
    inspirations_k: int = 2
    survivor_top_k: int = 4
    diversity_slots: int = 1
    seed: int = 7
    run_name: str | None = None
    seed_program_path: str = "mvp/tasks/astar_routing_target.py"
    mock_diff_path: str = "mvp/mock_diffs/astar_routing_diffs.json"

    def resolve_seed_program_path(self, base_dir: Path) -> Path:
        return (base_dir / self.seed_program_path).resolve()

    def resolve_mock_diff_path(self, base_dir: Path) -> Path:
        return (base_dir / self.mock_diff_path).resolve()
