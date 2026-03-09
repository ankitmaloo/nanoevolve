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

from adamopt.optim_search.deployment import RemoteTarget, deploy_candidate_workspace, fetch_deployment_trace


def _parse_env(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        key, sep, raw = value.partition("=")
        if not sep:
            raise ValueError(f"Invalid --env entry: {value}")
        env[key] = raw
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy a patched AdamOpt/NanoChat candidate and fetch traces.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy = subparsers.add_parser("deploy", help="stage and launch a candidate workspace")
    deploy.add_argument("--candidate-dir", required=True, type=str)
    deploy.add_argument("--candidate-id", required=True, type=str)
    deploy.add_argument("--run-command", required=True, type=str)
    deploy.add_argument("--target-name", required=True, type=str)
    deploy.add_argument("--transport", choices=["ssh", "local"], default="ssh")
    deploy.add_argument("--host", type=str, default="")
    deploy.add_argument("--user", type=str, default=None)
    deploy.add_argument("--port", type=int, default=22)
    deploy.add_argument("--identity-file", type=str, default=None)
    deploy.add_argument("--ssh-option", action="append", default=[])
    deploy.add_argument("--remote-base-dir", type=str, default="~/adamopt_remote")
    deploy.add_argument("--deployment-root", type=str, default=None)
    deploy.add_argument("--deployment-label", type=str, default=None)
    deploy.add_argument("--env", action="append", default=[])

    trace = subparsers.add_parser("trace", help="fetch latest status and log tail for a deployment")
    trace.add_argument("--deployment-dir", required=True, type=str)
    trace.add_argument("--tail-lines", type=int, default=200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "deploy":
        target = RemoteTarget(
            name=args.target_name,
            transport=args.transport,
            host=args.host,
            user=args.user,
            port=args.port,
            identity_file=args.identity_file,
            ssh_options=args.ssh_option,
            remote_base_dir=args.remote_base_dir,
        )
        artifacts = deploy_candidate_workspace(
            candidate_dir=Path(args.candidate_dir).resolve(),
            candidate_id=args.candidate_id,
            target=target,
            run_command=args.run_command,
            deployment_root=Path(args.deployment_root).resolve() if args.deployment_root else None,
            deployment_label=args.deployment_label,
            env=_parse_env(args.env),
        )
        print(
            json.dumps(
                {
                    "deployment_id": artifacts.deployment_id,
                    "candidate_id": artifacts.candidate_id,
                    "deployment_dir": str(artifacts.deployment_dir),
                    "manifest_path": str(artifacts.manifest_path),
                    "remote_dir": artifacts.remote_dir,
                    "remote_log_path": artifacts.remote_log_path,
                    "remote_status_path": artifacts.remote_status_path,
                },
                indent=2,
            )
        )
        return 0

    trace = fetch_deployment_trace(Path(args.deployment_dir).resolve(), tail_lines=args.tail_lines)
    print(json.dumps(trace, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

