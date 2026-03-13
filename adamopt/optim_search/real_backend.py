"""
Real NanoChat backend — runs candidates on Modal A100 GPUs.

This backend has the same .evaluate() interface as ToyNanoChatBackend,
so the tournament loop doesn't need to know which one it's using.

Usage:
    from adamopt.optim_search.real_backend import RealNanoChatBackend
    backend = RealNanoChatBackend()
    outcome = backend.evaluate(spec, seed=42, candidate_id="cand_0001")
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from .spec import MatrixOptimizerSpec
from .types import (
    CurvePoint,
    EvaluationOutcome,
    StepTelemetry,
    TrialMetrics,
)


class RealNanoChatBackend:
    """Backend that runs real NanoChat training via Modal A100.

    Two modes:
      mode="modal"  — spawns a Modal GPU function from the local machine (default).
      mode="direct"  — calls evaluate_real_nanochat() directly (only works on GPU).
    """

    def __init__(
        self,
        *,
        steps: int = 20,
        eval_every: int = 10,
        depth: int = 4,
        max_seq_len: int = 512,
        device_batch_size: int = 2,
        total_batch_size: int = 1024,
        device: str = "cuda",
        nanochat_base_dir: str = "/data/nanochat",
        mode: str = "modal",
    ) -> None:
        self.steps = steps
        self.eval_every = eval_every
        self.depth = depth
        self.max_seq_len = max_seq_len
        self.device_batch_size = device_batch_size
        self.total_batch_size = total_batch_size
        self.device = device
        self.nanochat_base_dir = nanochat_base_dir
        self.mode = mode

        # Cache baseline outcome to avoid redundant baseline runs.
        # The baseline is the same for every candidate in a tournament generation,
        # so we only run it once per seed.
        self._baseline_cache: dict[int, EvaluationOutcome] = {}

    def evaluate(
        self, spec: MatrixOptimizerSpec, *, seed: int, candidate_id: str
    ) -> EvaluationOutcome:
        """Evaluate a candidate spec on real NanoChat training.

        Returns the same EvaluationOutcome type as ToyNanoChatBackend.
        """
        if self.mode == "direct":
            return self._evaluate_direct(spec, seed=seed, candidate_id=candidate_id)
        elif self.mode == "modal":
            return self._evaluate_modal(spec, seed=seed, candidate_id=candidate_id)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _evaluate_direct(
        self, spec: MatrixOptimizerSpec, *, seed: int, candidate_id: str
    ) -> EvaluationOutcome:
        """Call evaluate_real_nanochat() directly — only works on GPU."""
        from .real_eval import RealEvalConfig, evaluate_real_nanochat

        config = RealEvalConfig(
            seed=seed,
            steps=self.steps,
            eval_every=self.eval_every,
            depth=self.depth,
            max_seq_len=self.max_seq_len,
            device_batch_size=self.device_batch_size,
            total_batch_size=self.total_batch_size,
            device=self.device,
            nanochat_base_dir=self.nanochat_base_dir,
        )
        return evaluate_real_nanochat(spec, config, candidate_id=candidate_id)

    def _evaluate_modal(
        self, spec: MatrixOptimizerSpec, *, seed: int, candidate_id: str
    ) -> EvaluationOutcome:
        """Spawn a Modal function to run evaluation on A100."""
        import modal

        # Reuse the same Modal app/image as modal_validate_spec.py
        repo_root = Path(__file__).resolve().parents[2]

        image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install("git", "build-essential", "curl")
            .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
            .env({"PATH": "/root/.local/bin:$PATH"})
            .add_local_file(
                str(repo_root / "nanochat" / "pyproject.toml"),
                remote_path="/root/nanoevolve/nanochat/pyproject.toml",
                copy=True,
            )
            .add_local_file(
                str(repo_root / "nanochat" / "uv.lock"),
                remote_path="/root/nanoevolve/nanochat/uv.lock",
                copy=True,
            )
            .run_commands(
                "cd /root/nanoevolve/nanochat && /root/.local/bin/uv sync --extra gpu --frozen --no-install-project"
            )
            .add_local_dir(
                str(repo_root / "adamopt"),
                remote_path="/root/nanoevolve/adamopt",
                copy=True,
            )
            .add_local_dir(
                str(repo_root / "nanochat" / "nanochat"),
                remote_path="/root/nanoevolve/nanochat/nanochat",
                copy=True,
            )
            .add_local_dir(
                str(repo_root / "nanochat" / "scripts"),
                remote_path="/root/nanoevolve/nanochat/scripts",
                copy=True,
            )
        )

        app = modal.App("nanoevolve-tournament", image=image)
        data_vol = modal.Volume.from_name("nanoevolve-data", create_if_missing=True)

        steps = self.steps
        eval_every = self.eval_every
        depth = self.depth
        max_seq_len = self.max_seq_len
        device_batch_size = self.device_batch_size
        total_batch_size = self.total_batch_size

        @app.function(
            gpu="A100",
            volumes={"/data": data_vol},
            timeout=900,
        )
        def run_on_gpu(spec_dict: dict, seed_val: int, cid: str) -> dict:
            import os
            import subprocess
            import sys as _sys

            remote_repo = "/root/nanoevolve"
            remote_nanochat = "/root/nanoevolve/nanochat"
            remote_data = "/data/nanochat"
            python_bin = "/root/nanoevolve/nanochat/.venv/bin/python"

            os.chdir(remote_repo)
            os.environ["NANOCHAT_BASE_DIR"] = remote_data
            os.environ["PYTHONPATH"] = remote_repo

            # Ensure data is ready
            os.chdir(remote_nanochat)
            tok_path = Path(remote_data) / "tokenizer" / "tokenizer.pkl"
            if not tok_path.exists():
                subprocess.run(
                    [python_bin, "-m", "nanochat.dataset", "-n", "3"],
                    capture_output=True, text=True, timeout=300, check=True,
                )
                subprocess.run(
                    [python_bin, "-m", "scripts.tok_train"],
                    capture_output=True, text=True, timeout=300, check=True,
                )
            data_vol.commit()
            os.chdir(remote_repo)

            venv_site = "/root/nanoevolve/nanochat/.venv/lib/python3.12/site-packages"
            for p in [remote_repo, remote_nanochat, venv_site]:
                if p not in _sys.path:
                    _sys.path.insert(0, p)

            from adamopt.optim_search.real_eval import RealEvalConfig, evaluate_real_nanochat
            from adamopt.optim_search.spec import MatrixOptimizerSpec as MS
            from dataclasses import asdict as _asdict

            config = RealEvalConfig(
                seed=seed_val,
                steps=steps,
                eval_every=eval_every,
                depth=depth,
                max_seq_len=max_seq_len,
                device_batch_size=device_batch_size,
                total_batch_size=total_batch_size,
                device="cuda",
                nanochat_base_dir=remote_data,
            )
            spec_obj = MS.from_dict(spec_dict)
            outcome = evaluate_real_nanochat(spec_obj, config, candidate_id=cid)
            return _asdict(outcome)

        spec_dict = spec.to_dict()
        with modal.enable_output():
            with app.run():
                result_dict = run_on_gpu.remote(spec_dict, seed, candidate_id)

        return _outcome_from_dict(result_dict)


def _outcome_from_dict(d: dict) -> EvaluationOutcome:
    """Reconstruct an EvaluationOutcome from a plain dict (Modal serialization)."""
    metrics = None
    if d.get("metrics") is not None:
        metrics = TrialMetrics(**d["metrics"])

    curve = [CurvePoint(**p) for p in d.get("curve", [])]
    telemetry = [StepTelemetry(**t) for t in d.get("telemetry", [])]

    return EvaluationOutcome(
        candidate_id=d["candidate_id"],
        spec_name=d["spec_name"],
        seed=d["seed"],
        valid=d["valid"],
        metrics=metrics,
        curve=curve,
        telemetry=telemetry,
        failure_type=d.get("failure_type"),
        notes=d.get("notes", []),
        diagnostics=d.get("diagnostics", {}),
    )
