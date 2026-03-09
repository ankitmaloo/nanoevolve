from __future__ import annotations

import json
from pathlib import Path

from mvp.config import RunConfig
from mvp.controller import EvolutionController


def test_astar_progression_mock_diffs_improve_across_four_generations(tmp_path: Path) -> None:
    repo_alphaevolve = Path(__file__).resolve().parents[1]

    cfg = RunConfig(
        mode="mock",
        generations=4,
        parallel_candidates=1,
        llm_concurrency=1,
        survivor_top_k=1,
        diversity_slots=0,
        run_name="pytest_astar_progression",
        seed_program_path=str((repo_alphaevolve / "mvp/tasks/astar_routing_target.py").resolve()),
        mock_diff_path=str((repo_alphaevolve / "mvp/mock_diffs/astar_progression_diffs.json").resolve()),
    )
    summary = EvolutionController(base_dir=tmp_path, config=cfg).run()
    run_dir = Path(summary["run_dir"])

    scores: list[float] = []
    for generation in range(1, 5):
        evaluation_path = run_dir / "evaluations" / f"gen_{generation:04d}_slot_00.json"
        evaluation = json.loads(evaluation_path.read_text())
        scores.append(float(evaluation["aggregate_score"]))

    assert all(next_score > prev_score for prev_score, next_score in zip(scores, scores[1:]))
