from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from adamopt.optim_search.autonomous import AutonomousSearchController
from adamopt.optim_search.config import AutonomousSearchConfig, ComparisonConfig, EvaluationConfig, SearchConfig
from adamopt.optim_search.command_mutator import (
    MutatorConfig,
    ProviderConfig,
    ensemble_patch_nanochat,
    patch_nanochat_adamw,
    SUPPORTED_PROVIDERS,
)
from adamopt.optim_search.deployment import RemoteTarget, deploy_candidate_workspace, fetch_deployment_trace
from adamopt.optim_search.eval_candidate import ToyNanoChatBackend, compare_baseline_candidate, write_metrics_json
from adamopt.optim_search.real_backend import RealNanoChatBackend
from adamopt.optim_search.spec import MatrixOptimizerSpec
from adamopt.optim_search.tournament import OptimizerTournament
from adamopt.optim_search.validation import validate_candidate_workspace


def _load_mutator_config(path: str | None) -> MutatorConfig:
    """Load a MutatorConfig from a JSON file, or return the default."""
    if path is None:
        return MutatorConfig()
    raw = json.loads(Path(path).read_text())
    providers = [
        ProviderConfig(
            name=p["name"],
            command_template=p["command_template"],
            enabled=p.get("enabled", True),
        )
        for p in raw.get("providers", [])
    ]
    return MutatorConfig(providers=providers) if providers else MutatorConfig()


def _parse_seeds(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_env(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        key, sep, raw = value.partition("=")
        if not sep:
            raise ValueError(f"Invalid --env entry: {value}")
        env[key] = raw
    return env


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["toy", "real"], default="toy")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--eval-every", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    # Real backend (Modal A100) args
    parser.add_argument("--gpu-steps", type=int, default=20, help="Training steps for real GPU eval")
    parser.add_argument("--gpu-eval-every", type=int, default=10, help="Eval cadence for GPU runs")
    parser.add_argument("--gpu-depth", type=int, default=4, help="Model depth for GPU runs")
    parser.add_argument("--gpu-seq-len", type=int, default=512, help="Sequence length for GPU runs")
    parser.add_argument("--gpu-batch-size", type=int, default=2, help="Device batch size for GPU runs")
    parser.add_argument("--gpu-total-batch", type=int, default=1024, help="Total batch size for GPU runs")


def _build_eval_config(args: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        seed=args.seed,
        steps=args.steps,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        model_dim=args.model_dim,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        device=args.device,
    )


def cmd_compare(args: argparse.Namespace) -> int:
    eval_config = _build_eval_config(args)
    backend = ToyNanoChatBackend(eval_config)
    payload = compare_baseline_candidate(
        backend=backend,
        baseline_spec=MatrixOptimizerSpec.baseline_nanochat(),
        candidate_spec=MatrixOptimizerSpec.trust_ratio_variant(),
        evaluation_config=eval_config,
        comparison_config=ComparisonConfig(notes=["phase_1_baseline_vs_hand_designed_variant"]),
    )
    if args.out:
        write_metrics_json(Path(args.out).resolve(), payload)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_tournament(args: argparse.Namespace) -> int:
    eval_config = _build_eval_config(args)
    search_config = SearchConfig(
        generations=args.generations,
        candidates_per_generation=args.population,
        survivor_top_k=args.survivor_top_k,
        promotion_top_k=args.promotion_top_k,
        improvement_threshold_bpb=args.improvement_threshold_bpb,
        min_time_to_target_ratio=args.min_time_to_target_ratio,
        max_slowdown_ratio=args.max_slowdown_ratio,
        max_memory_ratio=args.max_memory_ratio,
        min_tokens_per_sec_ratio=args.min_tokens_per_sec_ratio,
        max_grad_spike_delta=args.max_grad_spike_delta,
        max_stability_penalty_delta=args.max_stability_penalty_delta,
        min_seed_win_rate=args.min_seed_win_rate,
        promotion_seeds=_parse_seeds(args.promotion_seeds),
        seed=args.seed,
        run_name=args.run_name,
        out_dir=args.out_dir,
    )

    if args.backend == "real":
        backend = RealNanoChatBackend(
            steps=args.gpu_steps,
            eval_every=args.gpu_eval_every,
            depth=args.gpu_depth,
            max_seq_len=args.gpu_seq_len,
            device_batch_size=args.gpu_batch_size,
            total_batch_size=args.gpu_total_batch,
            mode="modal",
        )
    else:
        backend = ToyNanoChatBackend(eval_config)

    tournament = OptimizerTournament(
        root_dir=Path(__file__).resolve().parents[1],
        search_config=search_config,
        evaluation_config=eval_config,
        backend=backend,
    )
    summary = tournament.run()
    print(json.dumps(asdict(summary), indent=2))
    return 0


def cmd_patch_code(args: argparse.Namespace) -> int:
    cfg = _load_mutator_config(getattr(args, "providers_config", None))
    artifacts = patch_nanochat_adamw(
        nanochat_root=Path(args.nanochat_root).resolve(),
        candidate_dir=(Path(args.run_dir).resolve() / args.candidate_id),
        candidate_id=args.candidate_id,
        provider=args.provider,
        instruction=args.instruction,
        command_template=args.command_template,
        scope=args.scope,
        config=cfg,
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


def cmd_deploy_code(args: argparse.Namespace) -> int:
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


def cmd_trace_deployment(args: argparse.Namespace) -> int:
    trace = fetch_deployment_trace(Path(args.deployment_dir).resolve(), tail_lines=args.tail_lines)
    print(json.dumps(trace, indent=2))
    return 0


def cmd_validate_code(args: argparse.Namespace) -> int:
    artifacts = validate_candidate_workspace(
        candidate_dir=Path(args.candidate_dir).resolve(),
        scope=args.scope,
        python_executable=args.python_executable,
        disable_torch_compile=not args.keep_torch_compile,
    )
    print(
        json.dumps(
            {
                "ok": artifacts.ok,
                "scope": artifacts.scope,
                "summary_path": str(artifacts.summary_path),
                "stdout_path": str(artifacts.stdout_path),
                "stderr_path": str(artifacts.stderr_path),
                "result": artifacts.result,
            },
            indent=2,
        )
    )
    return 0


def cmd_ensemble_patch(args: argparse.Namespace) -> int:
    cfg = _load_mutator_config(getattr(args, "providers_config", None))
    providers = [p.strip() for p in args.providers.split(",")] if args.providers else None
    command_templates: dict[str, str] = {}
    for entry in args.command_template_override:
        provider, sep, template = entry.partition("=")
        if not sep:
            raise ValueError(f"Invalid --command-template-override: {entry} (expected provider=template)")
        command_templates[provider] = template

    result = ensemble_patch_nanochat(
        nanochat_root=Path(args.nanochat_root).resolve(),
        base_candidate_dir=Path(args.run_dir).resolve(),
        candidate_id_prefix=args.candidate_id_prefix,
        providers=providers,
        instruction=args.instruction,
        command_templates=command_templates or None,
        scope=args.scope,
        config=cfg,
    )
    output: dict[str, object] = {
        "candidate_id_prefix": result.candidate_id_prefix,
        "providers_succeeded": result.successful_providers,
        "providers_failed": result.failed_providers,
        "errors": result.provider_errors,
        "artifacts": {
            provider: {
                "candidate_id": a.candidate_id,
                "workspace_dir": str(a.workspace_dir),
                "patch_path": str(a.patch_path),
            }
            for provider, a in result.provider_results.items()
            if a is not None
        },
    }
    if result.diversity_review is not None:
        dr = result.diversity_review
        output["diversity_review"] = {
            "covers_solution_space": dr.covers_solution_space,
            "summary": dr.summary,
            "suggested_reprompts": dr.suggested_reprompts,
        }
    print(json.dumps(output, indent=2))
    return 0


def cmd_autonomous_run(args: argparse.Namespace) -> int:
    config = AutonomousSearchConfig(
        candidate_count=args.candidate_count,
        provider=args.provider,
        instruction_template=args.instruction_template,
        scope=args.scope,
        command_template=args.command_template,
        mutation_concurrency=args.mutation_concurrency,
        validation_concurrency=args.validation_concurrency,
        deployment_concurrency=args.deployment_concurrency,
        poll_concurrency=args.poll_concurrency,
        poll_interval_s=args.poll_interval_s,
        validation_disable_torch_compile=not args.keep_torch_compile,
        validation_python_executable=args.validation_python_executable,
        run_name=args.run_name,
        out_dir=args.out_dir,
    )
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
    root_dir = Path(__file__).resolve().parents[1]
    controller = AutonomousSearchController(
        root_dir=root_dir,
        run_dir=config.resolve_run_dir(root_dir),
        nanochat_root=Path(args.nanochat_root).resolve(),
        config=config,
        target=target,
        run_command_template=args.run_command_template,
    )
    summary = asyncio.run(controller.run())
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AdamOpt optimizer-search lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser("compare", help="run one baseline-vs-candidate comparison")
    _add_eval_args(compare_parser)
    compare_parser.add_argument("--out", type=str, default=None)
    compare_parser.set_defaults(func=cmd_compare)

    tournament_parser = subparsers.add_parser("tournament", help="run a small optimizer tournament")
    _add_eval_args(tournament_parser)
    tournament_parser.add_argument("--generations", type=int, default=3)
    tournament_parser.add_argument("--population", type=int, default=6)
    tournament_parser.add_argument("--survivor-top-k", type=int, default=3)
    tournament_parser.add_argument("--promotion-top-k", type=int, default=2)
    tournament_parser.add_argument("--promotion-seeds", type=str, default="7,13,29")
    tournament_parser.add_argument("--improvement-threshold-bpb", type=float, default=0.005)
    tournament_parser.add_argument("--min-time-to-target-ratio", type=float, default=1.05)
    tournament_parser.add_argument("--max-slowdown-ratio", type=float, default=1.35)
    tournament_parser.add_argument("--max-memory-ratio", type=float, default=1.35)
    tournament_parser.add_argument("--min-tokens-per-sec-ratio", type=float, default=0.95)
    tournament_parser.add_argument("--max-grad-spike-delta", type=int, default=0)
    tournament_parser.add_argument("--max-stability-penalty-delta", type=float, default=0.0)
    tournament_parser.add_argument("--min-seed-win-rate", type=float, default=2.0 / 3.0)
    tournament_parser.add_argument("--run-name", type=str, default="adamopt_search")
    tournament_parser.add_argument("--out-dir", type=str, default=None)
    tournament_parser.set_defaults(func=cmd_tournament)

    patch_parser = subparsers.add_parser("patch-code", help="patch NanoChat AdamW code with a CLI mutator")
    patch_parser.add_argument("--provider", choices=list(SUPPORTED_PROVIDERS), required=True)
    patch_parser.add_argument("--instruction", type=str, required=True)
    patch_parser.add_argument("--candidate-id", type=str, required=True)
    patch_parser.add_argument("--nanochat-root", type=str, default="nanochat")
    patch_parser.add_argument("--run-dir", type=str, default="adamopt/runs/code_mutations")
    patch_parser.add_argument("--scope", choices=["adamw_math", "muon_math", "optimizer_routing"], default="adamw_math")
    patch_parser.add_argument("--command-template", type=str, default=None)
    patch_parser.add_argument("--providers-config", type=str, default=None, help="path to providers.json config file")
    patch_parser.set_defaults(func=cmd_patch_code)

    ensemble_parser = subparsers.add_parser("ensemble-patch", help="run the same mutation through multiple providers (codex, claude, copilot)")
    ensemble_parser.add_argument("--providers", type=str, default=None, help="comma-separated list of providers (default: all enabled in config)")
    ensemble_parser.add_argument("--instruction", type=str, required=True)
    ensemble_parser.add_argument("--candidate-id-prefix", type=str, required=True)
    ensemble_parser.add_argument("--nanochat-root", type=str, default="nanochat")
    ensemble_parser.add_argument("--run-dir", type=str, default="adamopt/runs/ensemble_mutations")
    ensemble_parser.add_argument("--scope", choices=["adamw_math", "muon_math", "optimizer_routing"], default="adamw_math")
    ensemble_parser.add_argument("--command-template-override", action="append", default=[], help="provider=template override, e.g. codex='codex -q ...'")
    ensemble_parser.add_argument("--providers-config", type=str, default=None, help="path to providers.json config file")
    ensemble_parser.set_defaults(func=cmd_ensemble_patch)

    deploy_parser = subparsers.add_parser("deploy-code", help="deploy a patched candidate workspace to a remote target")
    deploy_parser.add_argument("--candidate-dir", required=True, type=str)
    deploy_parser.add_argument("--candidate-id", required=True, type=str)
    deploy_parser.add_argument("--run-command", required=True, type=str)
    deploy_parser.add_argument("--target-name", required=True, type=str)
    deploy_parser.add_argument("--transport", choices=["ssh", "local"], default="ssh")
    deploy_parser.add_argument("--host", type=str, default="")
    deploy_parser.add_argument("--user", type=str, default=None)
    deploy_parser.add_argument("--port", type=int, default=22)
    deploy_parser.add_argument("--identity-file", type=str, default=None)
    deploy_parser.add_argument("--ssh-option", action="append", default=[])
    deploy_parser.add_argument("--remote-base-dir", type=str, default="~/adamopt_remote")
    deploy_parser.add_argument("--deployment-root", type=str, default=None)
    deploy_parser.add_argument("--deployment-label", type=str, default=None)
    deploy_parser.add_argument("--env", action="append", default=[])
    deploy_parser.set_defaults(func=cmd_deploy_code)

    trace_parser = subparsers.add_parser("trace-deployment", help="fetch status and log tail for a deployment")
    trace_parser.add_argument("--deployment-dir", required=True, type=str)
    trace_parser.add_argument("--tail-lines", type=int, default=200)
    trace_parser.set_defaults(func=cmd_trace_deployment)

    validate_parser = subparsers.add_parser("validate-code", help="run local preflight validation on a patched candidate workspace")
    validate_parser.add_argument("--candidate-dir", required=True, type=str)
    validate_parser.add_argument("--scope", choices=["adamw_math", "muon_math", "optimizer_routing"], default="adamw_math")
    validate_parser.add_argument("--python-executable", type=str, default=None)
    validate_parser.add_argument("--keep-torch-compile", action="store_true")
    validate_parser.set_defaults(func=cmd_validate_code)

    autonomous_parser = subparsers.add_parser("autonomous-run", help="run the fully autonomous patch/deploy/poll loop")
    autonomous_parser.add_argument("--candidate-count", type=int, default=4)
    autonomous_parser.add_argument("--provider", choices=list(SUPPORTED_PROVIDERS), required=True)
    autonomous_parser.add_argument("--instruction-template", type=str, required=True)
    autonomous_parser.add_argument("--scope", choices=["adamw_math", "muon_math", "optimizer_routing"], default="adamw_math")
    autonomous_parser.add_argument("--command-template", type=str, default=None)
    autonomous_parser.add_argument("--mutation-concurrency", type=int, default=2)
    autonomous_parser.add_argument("--validation-concurrency", type=int, default=2)
    autonomous_parser.add_argument("--deployment-concurrency", type=int, default=2)
    autonomous_parser.add_argument("--poll-concurrency", type=int, default=8)
    autonomous_parser.add_argument("--poll-interval-s", type=float, default=5.0)
    autonomous_parser.add_argument("--validation-python-executable", type=str, default=None)
    autonomous_parser.add_argument("--keep-torch-compile", action="store_true")
    autonomous_parser.add_argument("--run-name", type=str, default="autonomous_search")
    autonomous_parser.add_argument("--out-dir", type=str, default=None)
    autonomous_parser.add_argument("--nanochat-root", type=str, default="nanochat")
    autonomous_parser.add_argument("--run-command-template", type=str, required=True)
    autonomous_parser.add_argument("--target-name", type=str, required=True)
    autonomous_parser.add_argument("--transport", choices=["ssh", "local"], default="ssh")
    autonomous_parser.add_argument("--host", type=str, default="")
    autonomous_parser.add_argument("--user", type=str, default=None)
    autonomous_parser.add_argument("--port", type=int, default=22)
    autonomous_parser.add_argument("--identity-file", type=str, default=None)
    autonomous_parser.add_argument("--ssh-option", action="append", default=[])
    autonomous_parser.add_argument("--remote-base-dir", type=str, default="~/adamopt_remote")
    autonomous_parser.set_defaults(func=cmd_autonomous_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
