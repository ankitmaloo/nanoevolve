"""
Real NanoChat evaluator — runs a candidate spec on actual GPT training.

This module provides the bridge between the optimizer DSL and real NanoChat
training. It takes a MatrixOptimizerSpec, builds a real GPT model, replaces
the Muon optimizer groups with SpecCandidateOptimizer matrix groups, and
runs a short-horizon training loop collecting the same EvaluationOutcome
metrics as the toy backend.

The key insight: NanoChat's setup_optimizer() produces param groups with
kind='muon' (matrix) and kind='adamw' (everything else). We keep the AdamW
groups untouched and replace kind='muon' with kind='matrix_candidate', then
hand the whole thing to SpecCandidateOptimizer.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .candidate_optimizer import SpecCandidateOptimizer
from .spec import MatrixOptimizerSpec
from .types import CurvePoint, EvaluationOutcome, StepTelemetry, TrialMetrics


@dataclass
class RealEvalConfig:
    """Configuration for a real NanoChat evaluation run."""
    seed: int = 42
    steps: int = 20
    eval_every: int = 10
    depth: int = 12
    max_seq_len: int = 1024
    device_batch_size: int = 4
    total_batch_size: int = 4096
    device: str = "cuda"
    grad_spike_factor: float = 4.0
    nanochat_base_dir: str = "/data/nanochat"

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


def _build_candidate_optimizer_from_nanochat_model(
    model: torch.nn.Module,
    spec: MatrixOptimizerSpec,
) -> SpecCandidateOptimizer:
    """Build a SpecCandidateOptimizer using the real NanoChat model's parameter structure.

    This mirrors GPT.setup_optimizer() but replaces kind='muon' with
    kind='matrix_candidate' so the SpecCandidateOptimizer handles matrix params.
    AdamW groups (embeddings, scalars) stay on our AdamW path.
    """
    model_dim = model.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    # Same grouping as GPT.setup_optimizer — but only 2D+ params are matrix
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

    # Matrix params grouped by shape — same as NanoChat but kind='matrix_candidate'
    for shape in sorted({p.shape for p in matrix_params}):
        group_params = [p for p in matrix_params if p.shape == shape]
        param_groups.append(dict(
            kind="matrix_candidate",
            group_name=f"matrix_{shape}",
            params=group_params,
            lr=0.02,
            weight_decay=0.2,
        ))

    # Filter empty groups
    param_groups = [g for g in param_groups if g["params"]]

    optimizer = SpecCandidateOptimizer(param_groups, spec)

    # Store initial_lr for schedule support
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    return optimizer


def evaluate_real_nanochat(
    spec: MatrixOptimizerSpec,
    config: RealEvalConfig,
    candidate_id: str,
) -> EvaluationOutcome:
    """Run a real NanoChat training loop with the candidate optimizer spec.

    This function must be called inside a Modal container (or any environment
    where nanochat is importable and CUDA is available).
    """
    import os
    os.environ.setdefault("NANOCHAT_BASE_DIR", config.nanochat_base_dir)

    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    from nanochat.dataloader import (
        tokenizing_distributed_data_loader_bos_bestfit,
        tokenizing_distributed_data_loader_with_state_bos_bestfit,
    )
    from nanochat.loss_eval import evaluate_bpb

    _seed_everything(config.seed)
    device = config.device

    # Build model
    model_config = GPTConfig(n_layer=config.depth, sequence_len=config.max_seq_len)
    model = GPT(model_config).to(device)
    model.train()

    # Build candidate optimizer (matrix params use the evolved spec)
    optimizer = _build_candidate_optimizer_from_nanochat_model(model, spec)

    # Tokenizer and data
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, config.device_batch_size, config.max_seq_len, split="train", device=device,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, config.device_batch_size, config.max_seq_len, split="val", device=device,
    )

    # Grad accumulation
    tokens_per_fwdbwd = config.device_batch_size * config.max_seq_len
    assert config.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = config.total_batch_size // tokens_per_fwdbwd

    # Prefetch first batch
    x, y, _ = next(train_loader)

    # LR schedule: warmup 10%, cosine decay over the rest
    warmup_steps = max(1, config.steps // 10)
    def get_lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, config.steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # Training loop
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

        # Forward + backward with grad accumulation
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

        # Compute grad norm
        grad_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq += float(param.grad.float().pow(2).sum().item())
        grad_norm = math.sqrt(grad_sq)

        if running_grad_norm > 0 and grad_norm > config.grad_spike_factor * running_grad_norm:
            grad_norm_spikes += 1
        running_grad_norm = grad_norm if running_grad_norm == 0 else 0.95 * running_grad_norm + 0.05 * grad_norm

        # LR schedule
        lrm = get_lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm

        # Feed context to stateful optimizer
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

        # Check for NaN/Inf in params
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

        # Record training curve point
        train_bpb = _loss_to_bpb(train_loss)
        point = CurvePoint(
            step=step,
            train_bpb=train_bpb,
            tokens_seen=step * config.total_batch_size,
        )

        # Validation eval
        if step % config.eval_every == 0 or step == config.steps:
            model.eval()
            val_loader = build_val_loader()
            eval_steps = max(1, 512 // (config.device_batch_size * config.max_seq_len))
            with torch.no_grad():
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            point.val_bpb = float(val_bpb)
            model.train()

        curve.append(point)

    # Build outcome
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
            "steps": config.steps,
        },
    )
