#!/usr/bin/env python3
"""Remote evaluator for Stage 2 — runs specs with code mutations on real NanoChat.

This is the GPU-side script submitted via Slurm. It extends the Stage 1
remote_eval.py with Stage2Optimizer support for code-level mutations.

Usage (on GPU node):
    python -m enigma.run_stage2_remote \
        --specs-dir runs/enigma_s2_XXXXX/specs \
        --results-dir runs/enigma_s2_XXXXX/results \
        --steps 20 --eval-every 10 --depth 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adamopt"))

from optim_search.spec import MatrixOptimizerSpec
from optim_search.candidate_optimizer import SpecCandidateOptimizer
from optim_search.types import CurvePoint, EvaluationOutcome, StepTelemetry, TrialMetrics

# Import Stage2Optimizer from run_stage2
from enigma.run_stage2 import Stage2Optimizer


@dataclass
class RealEvalConfig:
    seed: int = 42
    steps: int = 20
    eval_every: int = 10
    depth: int = 4
    max_seq_len: int = 512
    device_batch_size: int = 2
    total_batch_size: int = 1024
    device: str = "cuda"
    grad_spike_factor: float = 4.0

    @property
    def tokens_per_step(self) -> int:
        return self.total_batch_size

    @property
    def token_budget(self) -> int:
        return self.steps * self.tokens_per_step


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss_to_bpb(loss_value: float) -> float:
    return loss_value / math.log(2.0)


def _curve_auc(points: list[CurvePoint], attr: str) -> float:
    filtered = [(p.step, getattr(p, attr)) for p in points if getattr(p, attr) is not None]
    if len(filtered) < 2:
        return filtered[0][1] if filtered else float("inf")
    area = 0.0
    for (x0, y0), (x1, y1) in zip(filtered, filtered[1:]):
        area += (x1 - x0) * (float(y0) + float(y1)) * 0.5
    return area


def _build_optimizer_for_spec(
    model: torch.nn.Module,
    spec: MatrixOptimizerSpec,
    code_mutation: str = "none",
) -> SpecCandidateOptimizer:
    """Build optimizer from real NanoChat model, using Stage2Optimizer for code mutations."""
    model_dim = model.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    all_block_params = list(model.transformer.h.parameters())
    matrix_params = [p for p in all_block_params if p.ndim >= 2]
    block_scalar_params = [p for p in all_block_params if p.ndim < 2]
    embedding_params = list(model.transformer.wte.parameters())
    value_embeds_params = list(model.value_embeds.parameters())
    lm_head_params = list(model.lm_head.parameters())
    resid_params = [model.resid_lambdas]
    x0_params = [model.x0_lambdas]

    adam_betas = (0.8, 0.95)

    param_groups: list[dict] = [
        dict(kind="adamw", group_name="lm_head", params=lm_head_params,
             lr=0.004 * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", group_name="embedding", params=embedding_params,
             lr=0.2 * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", group_name="value_embeds", params=value_embeds_params,
             lr=0.2 * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", group_name="block_non_matrix", params=block_scalar_params,
             lr=0.5 * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", group_name="resid_scalars", params=resid_params,
             lr=0.5 * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", group_name="x0_scalars", params=x0_params,
             lr=0.5, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    ]

    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind="matrix_candidate",
            group_name=f"matrix_{shape}",
            params=group_params,
            lr=0.02,
            weight_decay=0.2,
        ))

    param_groups = [g for g in param_groups if g["params"]]

    if code_mutation and code_mutation != "none":
        optimizer = Stage2Optimizer(param_groups, spec, code_mutation=code_mutation)
    else:
        optimizer = SpecCandidateOptimizer(param_groups, spec)

    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    return optimizer


def evaluate_spec(
    spec: MatrixOptimizerSpec,
    config: RealEvalConfig,
    candidate_id: str,
    code_mutation: str = "none",
) -> EvaluationOutcome:
    """Run real NanoChat training with optional code mutation."""
    os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))

    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    from nanochat.dataloader import (
        tokenizing_distributed_data_loader_bos_bestfit,
        tokenizing_distributed_data_loader_with_state_bos_bestfit,
    )
    from nanochat.loss_eval import evaluate_bpb

    _seed_everything(config.seed)
    device = config.device

    model_config = GPTConfig(n_layer=config.depth, sequence_len=config.max_seq_len)
    model = GPT(model_config).to(device)
    model.train()

    optimizer = _build_optimizer_for_spec(model, spec, code_mutation)

    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, config.device_batch_size, config.max_seq_len, split="train", device=device,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, config.device_batch_size, config.max_seq_len, split="val", device=device,
    )

    tokens_per_fwdbwd = config.device_batch_size * config.max_seq_len
    assert config.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = config.total_batch_size // tokens_per_fwdbwd

    x, y, _ = next(train_loader)

    warmup_steps = max(1, config.steps // 10)
    def get_lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, config.steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    curve: list[CurvePoint] = []
    telemetry: list[StepTelemetry] = []
    step_times_ms: list[float] = []
    update_ratios: list[float] = []
    max_ratio = 0.0
    grad_norm_spikes = 0
    running_grad_norm = 0.0
    nan_failures = 0
    inf_failures = 0
    failure_type: str | None = None

    for step in range(1, config.steps + 1):
        model.train()
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for micro_step in range(grad_accum_steps):
            loss = model(x, y)
            accumulated_loss += loss.detach().item()
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            x, y, _ = next(train_loader)

        train_loss = accumulated_loss / grad_accum_steps

        if not math.isfinite(train_loss):
            failure_type = "non_finite_loss"
            nan_failures += int(math.isnan(train_loss))
            inf_failures += int(math.isinf(train_loss))
            break

        grad_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq += float(param.grad.float().pow(2).sum().item())
        grad_norm = math.sqrt(grad_sq)

        if running_grad_norm > 0 and grad_norm > config.grad_spike_factor * running_grad_norm:
            grad_norm_spikes += 1
        running_grad_norm = grad_norm if running_grad_norm == 0 else 0.95 * running_grad_norm + 0.05 * grad_norm

        lrm = get_lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm

        optimizer.set_step_context(
            loss_value=train_loss,
            step=step,
            total_steps=config.steps,
        )

        optimizer.step()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        step_times_ms.append(elapsed_ms)
        update_ratios.append(optimizer.last_step_stats.mean_update_param_ratio)
        max_ratio = max(max_ratio, optimizer.last_step_stats.max_update_param_ratio)

        telemetry.append(optimizer.snapshot_telemetry(
            step=step,
            loss=train_loss,
            grad_norm=grad_norm,
        ))

        invalid_params = False
        for param in model.parameters():
            if torch.isnan(param).any():
                nan_failures += 1
                invalid_params = True
                break
            if torch.isinf(param).any():
                inf_failures += 1
                invalid_params = True
                break
        if invalid_params:
            failure_type = "non_finite_parameters"
            break

        train_bpb = _loss_to_bpb(train_loss)
        point = CurvePoint(
            step=step,
            train_bpb=train_bpb,
            tokens_seen=step * config.total_batch_size,
        )

        if step % config.eval_every == 0 or step == config.steps:
            model.eval()
            val_loader = build_val_loader()
            eval_steps = max(1, 512 // (config.device_batch_size * config.max_seq_len))
            with torch.no_grad():
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            point.val_bpb = float(val_bpb)
            model.train()

        curve.append(point)

    if not curve:
        return EvaluationOutcome(
            candidate_id=candidate_id,
            spec_name=spec.name,
            seed=config.seed,
            valid=False,
            failure_type=failure_type or "empty_run",
            diagnostics={"spec": spec.to_dict()},
        )

    val_points = [p.val_bpb for p in curve if p.val_bpb is not None]
    final_validation_bpb = float(val_points[-1] if val_points else curve[-1].train_bpb)
    best_validation_bpb = float(min(val_points) if val_points else final_validation_bpb)
    mean_step_time_ms = sum(step_times_ms) / max(1, len(step_times_ms))
    total_time_s = sum(step_times_ms) / 1000.0
    tokens_per_sec = (config.token_budget / total_time_s) if total_time_s > 0 else 0.0
    stability_penalty = float((nan_failures + inf_failures) * 10 + grad_norm_spikes)

    metrics = TrialMetrics(
        final_validation_bpb=final_validation_bpb,
        best_validation_bpb=best_validation_bpb,
        train_curve_auc=_curve_auc(curve, "train_bpb"),
        validation_curve_auc=_curve_auc(curve, "val_bpb"),
        mean_step_time_ms=mean_step_time_ms,
        tokens_per_sec=tokens_per_sec,
        nan_failures=nan_failures,
        inf_failures=inf_failures,
        grad_norm_spikes=grad_norm_spikes,
        max_grad_norm=optimizer.last_step_stats.max_grad_norm,
        mean_update_param_ratio=(sum(update_ratios) / len(update_ratios)) if update_ratios else 0.0,
        max_update_param_ratio=max_ratio,
        memory_overhead_bytes=optimizer.estimated_state_bytes,
        stability_penalty=stability_penalty,
    )

    return EvaluationOutcome(
        candidate_id=candidate_id,
        spec_name=spec.name,
        seed=config.seed,
        valid=failure_type is None,
        metrics=metrics,
        curve=curve,
        telemetry=telemetry,
        failure_type=failure_type,
        diagnostics={
            "spec": spec.to_dict(),
            "token_budget": config.token_budget,
            "depth": config.depth,
            "code_mutation": code_mutation,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Enigma Stage 2 remote evaluator")
    parser.add_argument("--specs-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    config = RealEvalConfig(
        seed=args.seed,
        steps=args.steps,
        eval_every=args.eval_every,
        depth=args.depth,
    )

    spec_files = sorted(args.specs_dir.glob("*.json"))
    print(f"Found {len(spec_files)} spec files")

    for spec_file in spec_files:
        print(f"\nEvaluating: {spec_file.name}")
        spec_dict = json.loads(spec_file.read_text())
        spec = MatrixOptimizerSpec.from_dict(spec_dict)

        # Extract code_mutation from spec metadata
        code_mutation = spec.metadata.get("code_mutation", "none") if spec.metadata else "none"
        if code_mutation == "none" or code_mutation is None:
            code_mutation = "none"
        print(f"  code_mutation: {code_mutation}")

        t0 = time.perf_counter()
        try:
            outcome = evaluate_spec(spec, config, candidate_id=spec_file.stem,
                                    code_mutation=code_mutation)
            elapsed = time.perf_counter() - t0

            result = {
                "candidate_id": spec_file.stem,
                "spec_name": spec.name,
                "code_mutation": code_mutation,
                "valid": outcome.valid,
                "failure_type": outcome.failure_type,
                "eval_time_s": elapsed,
            }
            if outcome.valid and outcome.metrics:
                result.update({
                    "final_validation_bpb": outcome.metrics.final_validation_bpb,
                    "best_validation_bpb": outcome.metrics.best_validation_bpb,
                    "tokens_per_sec": outcome.metrics.tokens_per_sec,
                    "mean_step_time_ms": outcome.metrics.mean_step_time_ms,
                    "stability_penalty": outcome.metrics.stability_penalty,
                    "memory_bytes": outcome.metrics.memory_overhead_bytes,
                    "grad_norm_spikes": outcome.metrics.grad_norm_spikes,
                    "nan_failures": outcome.metrics.nan_failures,
                    "inf_failures": outcome.metrics.inf_failures,
                })
                result["curve"] = [asdict(p) for p in outcome.curve]
                result["telemetry"] = [asdict(t) for t in outcome.telemetry]

            out_file = args.results_dir / f"{spec_file.stem}_real.json"
            out_file.write_text(json.dumps(result, indent=2))
            bpb = result.get("final_validation_bpb", "N/A")
            print(f"  => valid={outcome.valid} bpb={bpb} mutation={code_mutation} ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            result = {
                "candidate_id": spec_file.stem,
                "code_mutation": code_mutation,
                "valid": False,
                "error": str(e),
                "eval_time_s": elapsed,
            }
            out_file = args.results_dir / f"{spec_file.stem}_real.json"
            out_file.write_text(json.dumps(result, indent=2))
            print(f"  => FAILED: {e}")


if __name__ == "__main__":
    main()
