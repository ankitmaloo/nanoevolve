from __future__ import annotations

import json
from pathlib import Path

import pytest

from adamopt.optim_search.command_mutator import patch_nanochat_adamw
from adamopt.tests.helpers_nanochat import local_nanochat_root, render_print_command


def test_patch_nanochat_adamw_tracks_prompt_response_and_diff(tmp_path: Path) -> None:
    try:
        nanochat_root = local_nanochat_root()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    candidate_dir = tmp_path / "runs" / "cand_0001"
    diff_output = (
        "<<<<<<< SEARCH\n"
        "    # Weight decay (decoupled, applied before the update)\n"
        "=======\n"
        "    # Weight decay (decoupled, applied before the update, adamopt test patch)\n"
        ">>>>>>> REPLACE"
    )
    command_template = render_print_command(diff_output)

    artifacts = patch_nanochat_adamw(
        nanochat_root=nanochat_root,
        candidate_dir=candidate_dir,
        candidate_id="cand_0001",
        provider="codex",
        instruction="Mutate AdamW",
        command_template=command_template,
    )

    assert artifacts.patch_path.exists()
    assert artifacts.prompt_path.exists()
    assert artifacts.response_path.exists()
    assert "adamopt test patch" in artifacts.target_file.read_text()

    metadata = json.loads(artifacts.metadata_path.read_text())
    assert metadata["provider"] == "codex"
    assert metadata["target_relpath"] == "nanochat/optim.py"


def test_patch_nanochat_routing_scope_targets_gpt_file(tmp_path: Path) -> None:
    try:
        nanochat_root = local_nanochat_root()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    candidate_dir = tmp_path / "runs" / "cand_0002"
    diff_output = (
        "<<<<<<< SEARCH\n"
        "        Factory = DistMuonAdamW if ddp else MuonAdamW\n"
        "=======\n"
        "        Factory = DistMuonAdamW if ddp else MuonAdamW  # adamopt routing test\n"
        ">>>>>>> REPLACE"
    )
    command_template = render_print_command(diff_output)

    artifacts = patch_nanochat_adamw(
        nanochat_root=nanochat_root,
        candidate_dir=candidate_dir,
        candidate_id="cand_0002",
        provider="codex",
        instruction="Mutate routing",
        command_template=command_template,
        scope="optimizer_routing",
    )

    assert artifacts.target_file.name == "gpt.py"
    assert "adamopt routing test" in artifacts.target_file.read_text()
