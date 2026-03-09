from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from adamopt.optim_search.autonomous import AutonomousSearchController
from adamopt.optim_search.config import AutonomousSearchConfig
from adamopt.optim_search.deployment import RemoteTarget
from adamopt.tests.helpers_nanochat import local_nanochat_root, render_print_command


def test_autonomous_loop_runs_candidates_to_completion(tmp_path: Path) -> None:
    try:
        nanochat_root = local_nanochat_root()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    diff_output = (
        "<<<<<<< SEARCH\n"
        "    # Weight decay (decoupled, applied before the update)\n"
        "=======\n"
        "    # Weight decay (decoupled, applied before the update, adamopt autonomous test)\n"
        ">>>>>>> REPLACE"
    )
    command_template = render_print_command(diff_output)

    config = AutonomousSearchConfig(
        candidate_count=2,
        provider="codex",
        instruction_template="Mutate candidate {candidate_id}",
        command_template=command_template,
        mutation_concurrency=2,
        validation_concurrency=2,
        deployment_concurrency=2,
        poll_concurrency=4,
        poll_interval_s=0.05,
        out_dir=str(tmp_path / "run"),
    )
    target = RemoteTarget(
        name="local-smoke",
        transport="local",
        remote_base_dir=str(tmp_path / "remote"),
    )
    controller = AutonomousSearchController(
        root_dir=tmp_path,
        run_dir=config.resolve_run_dir(tmp_path),
        nanochat_root=nanochat_root,
        config=config,
        target=target,
        run_command_template=(
            "python3 - <<'PY'\n"
            "import json, os\n"
            "with open(os.environ['ADAMOPT_RESULT_PATH'], 'w', encoding='utf-8') as handle:\n"
            "    json.dump({'score': 1.0, 'candidate_id': os.environ['ADAMOPT_CANDIDATE_ID']}, handle)\n"
            "print('done', os.environ['ADAMOPT_CANDIDATE_ID'])\n"
            "PY"
        ),
    )

    summary = asyncio.run(controller.run())

    assert sorted(summary["succeeded"]) == ["cand_0001", "cand_0002"]
    state = json.loads((config.resolve_run_dir(tmp_path) / "autonomous_state.json").read_text())
    assert all(candidate["status"] == "succeeded" for candidate in state["candidates"])
    assert all(candidate["result"]["score"] == 1.0 for candidate in state["candidates"])
    assert (config.resolve_run_dir(tmp_path) / "events.jsonl").exists()
