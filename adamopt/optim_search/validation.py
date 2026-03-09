from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from .optimizer_targets import get_patch_target


@dataclass
class ValidationArtifacts:
    candidate_dir: Path
    scope: str
    ok: bool
    summary_path: Path
    stdout_path: Path
    stderr_path: Path
    result: dict[str, object]


def _validator_script() -> str:
    return textwrap.dedent(
        """
        import json
        import os
        import py_compile
        import sys
        from pathlib import Path

        scope = sys.argv[1]
        workspace = Path(sys.argv[2]).resolve()
        result_path = Path(sys.argv[3]).resolve()
        disable_compile = sys.argv[4] == "1"

        sys.path.insert(0, str(workspace))

        if disable_compile:
            import torch

            def _identity_compile(fn=None, *args, **kwargs):
                if fn is None:
                    def decorator(inner):
                        return inner
                    return decorator
                return fn

            torch.compile = _identity_compile

        from adamopt.optim_search.optimizer_targets import get_patch_target

        patch_target = get_patch_target(scope)
        compile_files = sorted({patch_target.target_relpath, *patch_target.reference_relpaths})
        compile_results = []
        for relpath in compile_files:
            full = workspace / relpath
            py_compile.compile(str(full), doraise=True)
            compile_results.append(relpath)

        import importlib

        optim_module = importlib.import_module("nanochat.optim")
        gpt_module = importlib.import_module("nanochat.gpt")

        import torch

        GPT = getattr(gpt_module, "GPT")
        GPTConfig = getattr(gpt_module, "GPTConfig")
        config = GPTConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=32,
            window_pattern="L",
        )
        model = GPT(config)
        model.init_weights()
        optimizer = model.setup_optimizer(
            unembedding_lr=0.004,
            embedding_lr=0.02,
            matrix_lr=0.01,
            weight_decay=0.0,
            adam_betas=(0.8, 0.95),
            scalar_lr=0.05,
        )

        x = torch.randint(0, config.vocab_size, (2, 4), dtype=torch.long)
        y = torch.randint(0, config.vocab_size, (2, 4), dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        optimizer.step()
        state_dict = optimizer.state_dict()

        param_groups = [
            {
                "kind": group.get("kind"),
                "lr": group.get("lr"),
                "num_params": len(group.get("params", [])),
            }
            for group in optimizer.param_groups
        ]

        result = {
            "ok": True,
            "scope": scope,
            "compiled_files": compile_results,
            "imported_modules": ["nanochat.optim", "nanochat.gpt"],
            "optimizer_type": type(optimizer).__name__,
            "param_group_kinds": sorted({group["kind"] for group in param_groups}),
            "param_groups": param_groups,
            "loss": float(loss.item()),
            "state_dict_group_count": len(state_dict.get("param_groups", [])),
            "state_dict_state_count": len(state_dict.get("state", {})),
        }
        result_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        """
    )


def validate_candidate_workspace(
    *,
    candidate_dir: Path,
    scope: str,
    python_executable: str | None = None,
    disable_torch_compile: bool = True,
) -> ValidationArtifacts:
    workspace_dir = candidate_dir / "workspace"
    if not workspace_dir.exists():
        raise ValueError(f"Candidate workspace does not exist: {workspace_dir}")

    get_patch_target(scope)
    summary_path = candidate_dir / "validation.json"
    stdout_path = candidate_dir / "validation.stdout.txt"
    stderr_path = candidate_dir / "validation.stderr.txt"
    started = time.perf_counter()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{workspace_dir}:{env.get('PYTHONPATH', '')}".rstrip(":")
    result = subprocess.run(
        [
            python_executable or sys.executable,
            "-c",
            _validator_script(),
            scope,
            str(workspace_dir),
            str(summary_path),
            "1" if disable_torch_compile else "0",
        ],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    stdout_path.write_text(result.stdout or "")
    stderr_path.write_text(result.stderr or "")

    if result.returncode != 0:
        failure = {
            "ok": False,
            "scope": scope,
            "error": f"validation subprocess failed with exit code {result.returncode}",
            "duration_s": time.perf_counter() - started,
        }
        summary_path.write_text(json.dumps(failure, indent=2))
        return ValidationArtifacts(
            candidate_dir=candidate_dir,
            scope=scope,
            ok=False,
            summary_path=summary_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            result=failure,
        )

    payload = json.loads(summary_path.read_text())
    payload["duration_s"] = time.perf_counter() - started
    summary_path.write_text(json.dumps(payload, indent=2))
    return ValidationArtifacts(
        candidate_dir=candidate_dir,
        scope=scope,
        ok=bool(payload.get("ok")),
        summary_path=summary_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        result=payload,
    )
