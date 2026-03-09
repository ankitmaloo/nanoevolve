from __future__ import annotations

import time
from pathlib import Path

from adamopt.optim_search.deployment import RemoteTarget, deploy_candidate_workspace, fetch_deployment_trace


def test_local_deployment_and_trace(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "cand_0001"
    workspace_dir = candidate_dir / "workspace"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "hello.txt").write_text("hello\n")

    target = RemoteTarget(
        name="local-smoke",
        transport="local",
        remote_base_dir=str(tmp_path / "remote_runs"),
    )
    artifacts = deploy_candidate_workspace(
        candidate_dir=candidate_dir,
        candidate_id="cand_0001",
        target=target,
        run_command="python3 - <<'PY'\nprint('remote hello')\nPY",
    )

    for _ in range(100):
        trace = fetch_deployment_trace(artifacts.deployment_dir, tail_lines=20)
        if trace["status"]["state"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("deployment did not finish in time")

    assert artifacts.manifest_path.exists()
    assert artifacts.launch_script_path.exists()
    assert artifacts.command_script_path.exists()
    assert "remote hello" in artifacts.fetched_log_tail_path.read_text()
    assert trace["status"]["state"] == "succeeded"
