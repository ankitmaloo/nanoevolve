"""Generic evaluation harness — runs any external command and parses JSON metrics."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from enigma.config import TaskConfig
from enigma.types import EvaluationResult, StageResult


class Evaluator:
    """Evaluate a candidate program using the method specified in task.json."""

    def __init__(self, task_config: TaskConfig, task_dir: Path) -> None:
        self.task_config = task_config
        self.task_dir = task_dir

    def evaluate(
        self,
        program_source: str,
        *,
        timeout_multiplier: float = 1.0,
        extra_env: dict[str, str] | None = None,
    ) -> EvaluationResult:
        if self.task_config.eval_mode != "command":
            return EvaluationResult(
                valid=False,
                aggregate_score=-1.0,
                failure_reasons=[f"Unsupported eval_mode: {self.task_config.eval_mode}"],
                stage_results=[StageResult(
                    name="config", passed=False,
                    message=f"Unknown eval_mode: {self.task_config.eval_mode}",
                )],
            )

        if not self.task_config.eval_command:
            return EvaluationResult(
                valid=False,
                aggregate_score=-1.0,
                failure_reasons=["eval_command is not set in task.json"],
                stage_results=[StageResult(name="config", passed=False, message="no eval_command")],
            )

        return self._evaluate_command(
            program_source,
            timeout_multiplier=timeout_multiplier,
            extra_env=extra_env,
        )

    def _evaluate_command(
        self,
        program_source: str,
        *,
        timeout_multiplier: float = 1.0,
        extra_env: dict[str, str] | None = None,
    ) -> EvaluationResult:
        suffix = Path(self.task_config.seed_file).suffix or ".py"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, dir=str(self.task_dir),
            delete=False, prefix="enigma_cand_",
        ) as f:
            f.write(program_source)
            candidate_path = f.name

        try:
            cmd = self.task_config.eval_command.replace("{candidate_file}", candidate_path)

            cwd = self.task_dir
            if self.task_config.eval_cwd:
                cwd = (self.task_dir / self.task_config.eval_cwd).resolve()

            env = os.environ.copy()
            env.update(self.task_config.eval_env)
            if extra_env:
                env.update(extra_env)
            env["CANDIDATE_FILE"] = candidate_path
            env["TASK_DIR"] = str(self.task_dir)

            timeout = int(self.task_config.eval_timeout * timeout_multiplier)

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
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
                    stage_results=[StageResult(
                        name="eval_command", passed=False,
                        message=f"exit code {result.returncode}",
                    )],
                )

            # Parse JSON from stdout — find the last JSON object/line
            stdout = result.stdout.strip()
            json_str = self._extract_json(stdout)

            try:
                output = json.loads(json_str)
            except json.JSONDecodeError as e:
                return EvaluationResult(
                    valid=False,
                    aggregate_score=-1.0,
                    failure_reasons=[
                        f"eval_command output is not valid JSON: {e}",
                        f"stdout tail: {stdout[-500:]}",
                    ],
                    stage_results=[StageResult(
                        name="eval_command", passed=False,
                        message=f"JSON parse error: {e}",
                    )],
                )

            valid = output.get("valid", True)
            aggregate_score = float(output.get("aggregate_score", output.get("score", -1.0)))
            metrics = output.get("metrics", {})
            failure_reasons = output.get("failure_reasons", [])

            if "aggregate_score" not in output and "score" not in output:
                pm = self.task_config.primary_metric
                if pm in metrics:
                    aggregate_score = float(metrics[pm])

            return EvaluationResult(
                valid=valid,
                aggregate_score=aggregate_score,
                metrics=metrics,
                failure_reasons=failure_reasons,
                stage_results=[StageResult(
                    name="eval_command", passed=valid, metrics=metrics,
                    message=f"exit 0, score={aggregate_score:.4f}",
                )],
                diagnostics={
                    "stdout_tail": result.stdout[-1000:],
                    "stderr_tail": result.stderr[-500:],
                },
            )

        except subprocess.TimeoutExpired:
            return EvaluationResult(
                valid=False,
                aggregate_score=-1.0,
                failure_reasons=[f"eval_command timed out after {timeout}s"],
                stage_results=[StageResult(name="eval_command", passed=False, message="timeout")],
            )
        finally:
            try:
                os.unlink(candidate_path)
            except OSError:
                pass

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain non-JSON lines."""
        # Try the whole thing first
        text = text.strip()
        if text.startswith("{") or text.startswith("["):
            return text

        # Try last line
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line.startswith("{") or line.startswith("["):
                return line

        # Fallback: find last { ... } block
        last_brace = text.rfind("}")
        if last_brace >= 0:
            first_brace = text.rfind("{", 0, last_brace)
            if first_brace >= 0:
                return text[first_brace:last_brace + 1]

        return text
