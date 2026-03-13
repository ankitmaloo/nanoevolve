"""Test that the generic task_config + evaluator system works."""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from mvp.task_config import TaskConfig
from mvp.generic_evaluator import GenericEvaluator


def test_task_config_load():
    """task.json in mvp/tasks/ loads correctly."""
    task_dir = Path(ROOT_DIR) / "mvp" / "tasks"
    config = TaskConfig.load(task_dir)
    assert config.name == "optimize_astar_heuristic"
    assert config.seed_file == "astar_routing_target.py"
    assert config.eval_mode == "builtin_astar"
    assert "aggregate_score" in config.metric_keys
    assert config.maximize is True


def test_generic_evaluator_builtin_astar():
    """GenericEvaluator with builtin_astar evaluates the seed program correctly."""
    task_dir = Path(ROOT_DIR) / "mvp" / "tasks"
    config = TaskConfig.load(task_dir)
    evaluator = GenericEvaluator(config, task_dir)

    seed_source = config.resolve_seed_path(task_dir).read_text()
    result = evaluator.evaluate(seed_source)

    assert result.valid is True
    assert result.aggregate_score > 0
    assert "solved_ratio" in result.metrics or "aggregate_score" in result.metrics


def test_generic_evaluator_command_mode(tmp_path):
    """GenericEvaluator with eval_mode='command' runs a shell command."""
    # Create a simple eval script
    eval_script = tmp_path / "eval.sh"
    eval_script.write_text('#!/bin/bash\necho \'{"valid": true, "aggregate_score": 42.0, "metrics": {"speed": 100}}\'\n')
    eval_script.chmod(0o755)

    # Create a seed file
    seed = tmp_path / "target.py"
    seed.write_text("# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n")

    # Create task.json
    task_json = {
        "name": "test_command",
        "description": "Test command evaluation",
        "seed_file": "target.py",
        "mutable_files": ["target.py"],
        "eval_command": "bash eval.sh {candidate_file}",
        "eval_mode": "command",
        "metric_keys": ["speed"],
        "primary_metric": "speed",
    }
    (tmp_path / "task.json").write_text(json.dumps(task_json))

    config = TaskConfig.load(tmp_path)
    evaluator = GenericEvaluator(config, tmp_path)

    result = evaluator.evaluate(seed.read_text())
    assert result.valid is True
    assert result.aggregate_score == 42.0
    assert result.metrics["speed"] == 100


def test_generic_evaluator_command_failure(tmp_path):
    """GenericEvaluator handles eval_command failure gracefully."""
    eval_script = tmp_path / "eval.sh"
    eval_script.write_text("#!/bin/bash\nexit 1\n")
    eval_script.chmod(0o755)

    seed = tmp_path / "target.py"
    seed.write_text("# EVOLVE-BLOCK-START\nx = 1\n# EVOLVE-BLOCK-END\n")

    task_json = {
        "name": "test_fail",
        "description": "Test failure",
        "seed_file": "target.py",
        "eval_command": "bash eval.sh",
        "eval_mode": "command",
    }
    (tmp_path / "task.json").write_text(json.dumps(task_json))

    config = TaskConfig.load(tmp_path)
    evaluator = GenericEvaluator(config, tmp_path)

    result = evaluator.evaluate(seed.read_text())
    assert result.valid is False
    assert result.aggregate_score == -1.0
