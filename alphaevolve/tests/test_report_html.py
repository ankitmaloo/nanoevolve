from __future__ import annotations

from pathlib import Path

from mvp.config import RunConfig
from mvp.controller import EvolutionController
from mvp.report_html import generate_html_report


def test_generate_html_report_contains_demo_sections(tmp_path: Path) -> None:
    repo_alphaevolve = Path(__file__).resolve().parents[1]

    cfg = RunConfig(
        mode="mock",
        generations=3,
        run_name="pytest_report_run",
        seed_program_path=str((repo_alphaevolve / "mvp/tasks/astar_routing_target.py").resolve()),
        mock_diff_path=str((repo_alphaevolve / "mvp/mock_diffs/astar_routing_diffs.json").resolve()),
    )

    controller = EvolutionController(base_dir=tmp_path, config=cfg)
    summary = controller.run()
    run_dir = Path(summary["run_dir"])

    report_path = generate_html_report(run_dir)
    content = report_path.read_text()

    assert report_path.exists()
    assert "1) Basic Implementation (Baseline)" in content
    assert "2) Evolution Over Generations" in content
    assert "3) Final Implementation (Best Candidate)" in content
