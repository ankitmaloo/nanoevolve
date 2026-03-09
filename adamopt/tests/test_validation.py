from __future__ import annotations

import json
from pathlib import Path

import pytest

from adamopt.optim_search.command_mutator import patch_nanochat_adamw
from adamopt.optim_search.validation import validate_candidate_workspace
from adamopt.tests.helpers_nanochat import local_nanochat_root, render_print_command


def test_validate_candidate_workspace_succeeds_on_real_nanochat_clone(tmp_path: Path) -> None:
    try:
        nanochat_root = local_nanochat_root()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    candidate_dir = tmp_path / "runs" / "cand_0001"
    diff_output = (
        "<<<<<<< SEARCH\n"
        "    # Weight decay (decoupled, applied before the update)\n"
        "=======\n"
        "    # Weight decay (decoupled, applied before the update, adamopt validation smoke)\n"
        ">>>>>>> REPLACE"
    )
    patch_nanochat_adamw(
        nanochat_root=nanochat_root,
        candidate_dir=candidate_dir,
        candidate_id="cand_0001",
        provider="codex",
        instruction="Add a harmless comment to the AdamW path.",
        command_template=render_print_command(diff_output),
        scope="adamw_math",
    )

    artifacts = validate_candidate_workspace(candidate_dir=candidate_dir, scope="adamw_math")

    assert artifacts.ok
    assert artifacts.summary_path.exists()
    assert artifacts.stdout_path.exists()
    assert artifacts.stderr_path.exists()
    assert artifacts.result["optimizer_type"] in {"MuonAdamW", "DistMuonAdamW"}
    assert artifacts.result["param_group_kinds"] == ["adamw", "muon"]
    assert artifacts.result["state_dict_group_count"] >= 2


def test_validate_candidate_workspace_reports_failure_for_syntax_error(tmp_path: Path) -> None:
    try:
        nanochat_root = local_nanochat_root()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    candidate_dir = tmp_path / "runs" / "cand_0002"
    diff_output = (
        "<<<<<<< SEARCH\n"
        "def adamw_step_fused(\n"
        "=======\n"
        "def adamw_step_fused(\n"
        ">>>>>>> REPLACE"
    )
    patch_nanochat_adamw(
        nanochat_root=nanochat_root,
        candidate_dir=candidate_dir,
        candidate_id="cand_0002",
        provider="codex",
        instruction="Create a candidate workspace before corrupting it.",
        command_template=render_print_command(diff_output),
        scope="adamw_math",
    )
    broken_file = candidate_dir / "workspace" / "nanochat" / "optim.py"
    broken_file.write_text("def broken(:\n")

    artifacts = validate_candidate_workspace(candidate_dir=candidate_dir, scope="adamw_math")

    assert not artifacts.ok
    assert "validation subprocess failed" in artifacts.result["error"]
    assert artifacts.stderr_path.read_text()


def test_validate_candidate_workspace_smoke_on_real_nanochat_clone(tmp_path: Path) -> None:
    try:
        nanochat_root = local_nanochat_root()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    candidate_dir = tmp_path / "runs" / "cand_real_0001"
    diff_output = (
        "<<<<<<< SEARCH\n"
        "    # Weight decay (decoupled, applied before the update)\n"
        "=======\n"
        "    # Weight decay (decoupled, applied before the update, adamopt validation smoke)\n"
        ">>>>>>> REPLACE"
    )
    patch_nanochat_adamw(
        nanochat_root=nanochat_root,
        candidate_dir=candidate_dir,
        candidate_id="cand_real_0001",
        provider="codex",
        instruction="Add a harmless comment inside AdamW for validation smoke coverage.",
        command_template=render_print_command(diff_output),
        scope="adamw_math",
    )

    artifacts = validate_candidate_workspace(candidate_dir=candidate_dir, scope="adamw_math")

    assert artifacts.ok
    assert artifacts.result["optimizer_type"] in {"MuonAdamW", "DistMuonAdamW"}
    assert "nanochat.optim" in artifacts.result["imported_modules"]
    assert "nanochat.gpt" in artifacts.result["imported_modules"]
    summary = json.loads(artifacts.summary_path.read_text())
    assert summary["ok"] is True
