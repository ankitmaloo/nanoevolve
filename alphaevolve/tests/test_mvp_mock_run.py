from __future__ import annotations

import json
from pathlib import Path

from mvp.config import RunConfig
from mvp.controller import EvolutionController


def test_mock_run_end_to_end(tmp_path: Path) -> None:
    repo_alphaevolve = Path(__file__).resolve().parents[1]

    cfg = RunConfig(
        mode="mock",
        generations=4,
        run_name="pytest_mock_run",
        seed_program_path=str((repo_alphaevolve / "mvp/tasks/toy_target.py").resolve()),
        mock_diff_path=str((repo_alphaevolve / "mvp/mock_diffs/sample_diffs.json").resolve()),
    )

    controller = EvolutionController(base_dir=tmp_path, config=cfg)
    summary = controller.run()

    expected_candidates = 1 + (cfg.generations * cfg.parallel_candidates)
    assert summary["total_candidates"] == expected_candidates
    assert summary["best_candidate_id"].startswith("cand_")

    run_dir = Path(summary["run_dir"])
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "best_program.py").exists()
    assert (run_dir / "summary.json").exists()

    events = (run_dir / "events.jsonl").read_text().strip().splitlines()
    assert len(events) == expected_candidates

    loaded_summary = json.loads((run_dir / "summary.json").read_text())
    assert loaded_summary["mode"] == "mock"
