"""
Generic evaluator that runs an external command and reads JSON metrics.

This replaces the hardcoded A*/BinPack evaluator for external targets.
It also wraps the existing builtin evaluators so the controller doesn't
need to know which type is in use.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from mvp.task_config import TaskConfig
from mvp.types import EvaluationResult, StageResult


class GenericEvaluator:
    """Evaluate a candidate program using the method specified in task.json."""

    def __init__(self, task_config: TaskConfig, task_dir: Path) -> None:
        self.task_config = task_config
        self.task_dir = task_dir

        # Lazily load builtin evaluator only if needed
        self._builtin = None
        if task_config.eval_mode.startswith("builtin_"):
            from mvp.evaluator import Evaluator
            self._builtin = Evaluator()

    def evaluate(self, program_source: str) -> EvaluationResult:
        mode = self.task_config.eval_mode

        if mode == "builtin_astar" and self._builtin is not None:
            return self._builtin.evaluate(program_source)

        if mode == "builtin_binpack" and self._builtin is not None:
            return self._builtin.evaluate(program_source)

        if mode == "command":
            return self._evaluate_command(program_source)

        raise ValueError(f"Unknown eval_mode: {mode}")

    def _evaluate_command(self, program_source: str) -> EvaluationResult:
        """Write candidate to temp file, run eval_command, parse JSON output."""
        if not self.task_config.eval_command:
            return EvaluationResult(
                valid=False,
                aggregate_score=-1.0,
                failure_reasons=["eval_command is not set in task.json"],
                stage_results=[StageResult(name="config", passed=False, message="no eval_command")],
            )

        # Write candidate source to a temp file
        suffix = Path(self.task_config.seed_file).suffix or ".py"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, dir=str(self.task_dir), delete=False, prefix="candidate_"
        ) as f:
            f.write(program_source)
            candidate_path = f.name

        try:
            # Build command with {candidate_file} substitution
            cmd = self.task_config.eval_command.replace("{candidate_file}", candidate_path)

            # Determine working directory
            cwd = self.task_dir
            if self.task_config.eval_cwd:
                cwd = (self.task_dir / self.task_config.eval_cwd).resolve()

            # Build environment
            env = os.environ.copy()
            env.update(self.task_config.eval_env)
            env["CANDIDATE_FILE"] = candidate_path
            env["TASK_DIR"] = str(self.task_dir)

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.task_config.eval_timeout,
                cwd=str(cwd),
                env=env,
            )

            if result.returncode != 0:
                return EvaluationResult(
                    valid=False,
                    aggregate_score=-1.0,
                    failure_reasons=[
                        f"eval_command exited with code {result.returncode}",
                        f"stderr: {result.stderr[:500]}",
                    ],
                    stage_results=[
                        StageResult(name="eval_command", passed=False, message=f"exit code {result.returncode}")
                    ],
                )

            # Parse JSON from stdout
            try:
                output = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                return EvaluationResult(
                    valid=False,
                    aggregate_score=-1.0,
                    failure_reasons=[
                        f"eval_command output is not valid JSON: {e}",
                        f"stdout: {result.stdout[:500]}",
                    ],
                    stage_results=[
                        StageResult(name="eval_command", passed=False, message=f"JSON parse error: {e}")
                    ],
                )

            # Extract standard fields
            valid = output.get("valid", True)
            aggregate_score = float(output.get("aggregate_score", output.get("score", -1.0)))
            metrics = output.get("metrics", {})
            failure_reasons = output.get("failure_reasons", [])

            # If primary_metric is in metrics, use it as aggregate_score if not explicit
            if "aggregate_score" not in output and "score" not in output:
                pm = self.task_config.primary_metric
                if pm in metrics:
                    aggregate_score = float(metrics[pm])

            return EvaluationResult(
                valid=valid,
                aggregate_score=aggregate_score,
                metrics=metrics,
                failure_reasons=failure_reasons,
                stage_results=[
                    StageResult(
                        name="eval_command",
                        passed=valid,
                        metrics=metrics,
                        message=f"exit 0, score={aggregate_score:.4f}",
                    )
                ],
                diagnostics={"stdout_tail": result.stdout[-1000:], "stderr_tail": result.stderr[-500:]},
            )

        except subprocess.TimeoutExpired:
            return EvaluationResult(
                valid=False,
                aggregate_score=-1.0,
                failure_reasons=[f"eval_command timed out after {self.task_config.eval_timeout}s"],
                stage_results=[
                    StageResult(name="eval_command", passed=False, message="timeout")
                ],
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(candidate_path)
            except OSError:
                pass
