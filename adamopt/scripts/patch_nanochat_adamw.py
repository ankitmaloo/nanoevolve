from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from adamopt.optim_search.command_mutator import patch_nanochat_adamw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch NanoChat's AdamW code path and track each mutation separately.")
    parser.add_argument("--provider", choices=["codex", "claude"], required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--candidate-id", type=str, required=True)
    parser.add_argument("--nanochat-root", type=str, default="nanochat")
    parser.add_argument("--run-dir", type=str, default="adamopt/runs/code_mutations")
    parser.add_argument("--scope", choices=["adamw_math", "muon_math", "optimizer_routing"], default="adamw_math")
    parser.add_argument("--command-template", type=str, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    nanochat_root = Path(args.nanochat_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    candidate_dir = run_dir / args.candidate_id
    artifacts = patch_nanochat_adamw(
        nanochat_root=nanochat_root,
        candidate_dir=candidate_dir,
        candidate_id=args.candidate_id,
        provider=args.provider,
        instruction=args.instruction,
        command_template=args.command_template,
        scope=args.scope,
    )
    print(
        json.dumps(
            {
                "candidate_id": artifacts.candidate_id,
                "provider": artifacts.provider,
                "workspace_dir": str(artifacts.workspace_dir),
                "target_file": str(artifacts.target_file),
                "patch_path": str(artifacts.patch_path),
                "prompt_path": str(artifacts.prompt_path),
                "response_path": str(artifacts.response_path),
                "metadata_path": str(artifacts.metadata_path),
                "changed_files": artifacts.changed_files,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
