#!/usr/bin/env python3
"""Enigma Stage 3: Doctrine-driven code-level mutations (deep hypothesis generation).

Portfolio (8 hypotheses across 5 pipeline stages):
  Slot 1 (trust)    H39: Per-neuron trust (col-wise wide, row-wise tall)
  Slot 2 (2nd mom)  H42: Second moment bias correction for early steps
  Slot 3 (WD)       H59: Soft cautious WD with sigmoid alignment
  Slot 4 (momentum) H46: Nesterov blend decoupled from momentum beta
  Slot 5 (momentum) H41: Momentum warmup 0.85 -> 0.95 over 200 steps
  Slot 6 (ortho)    H44: Post-orthogonalization scale preservation
  Slot 7 (WD)       H54: Cautious WD on raw gradient alignment
  Slot 8 (trust)    H36: Asymmetric trust — soft floor, hard ceiling

Base: H02 trust ratio (the Stage 2 winner) = baseline_nanochat + layerwise trust [0.5, 1.5]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adamopt"))

from optim_search.spec import (
    MatrixOptimizerSpec,
    TrustRatioConfig,
)
from optim_search.candidate_optimizer import (
    POLAR_EXPRESS_COEFFS,
    OptimizerStepStats,
    SpecCandidateOptimizer,
    build_nanochat_param_groups,
)
from optim_search.eval_candidate import ToyNanoChatBackend
from optim_search.config import EvaluationConfig


# ── Base spec: H02 trust ratio winner ────────────────────────

def base_spec() -> MatrixOptimizerSpec:
    return MatrixOptimizerSpec.trust_ratio_variant()


# ── Portfolio ─────────────────────────────────────────────────

STAGE3_PORTFOLIO = [
    {
        "slot_id": "S3_P1", "role": "exploit", "hypothesis_id": "H39",
        "title": "Per-neuron trust (aspect-aware axis)",
        "code_mutation": "H39_perneuron_trust",
        "mechanism": "Trust ratio per output neuron: row-wise norm for tall, col-wise for wide. Shape (N,rows,1) or (N,1,cols).",
    },
    {
        "slot_id": "S3_P2", "role": "exploit", "hypothesis_id": "H42",
        "title": "Second moment bias correction",
        "code_mutation": "H42_bias_correction",
        "mechanism": "Apply 1/(1-beta2^step) correction to second moment buffer for first ~50 steps.",
    },
    {
        "slot_id": "S3_P3", "role": "exploit", "hypothesis_id": "H59",
        "title": "Soft cautious WD with sigmoid",
        "code_mutation": "H59_soft_wd",
        "mechanism": "Replace binary (update*params)>=0 mask with sigmoid soft mask. Temperature=5.0.",
    },
    {
        "slot_id": "S3_P4", "role": "scout", "hypothesis_id": "H46",
        "title": "Nesterov blend decoupled (0.8)",
        "code_mutation": "H46_nesterov_decouple",
        "mechanism": "Nesterov interpolation uses blend=0.8 while EMA keeps momentum=0.95.",
    },
    {
        "slot_id": "S3_P5", "role": "scout", "hypothesis_id": "H41",
        "title": "Momentum warmup 0.85 to 0.95",
        "code_mutation": "H41_momentum_warmup",
        "mechanism": "Momentum beta linearly warms from 0.85 to 0.95 over first 200 steps.",
    },
    {
        "slot_id": "S3_P6", "role": "scout", "hypothesis_id": "H44",
        "title": "Post-ortho scale preservation",
        "code_mutation": "H44_scale_preserve",
        "mechanism": "Renormalize after Polar Express to preserve pre-ortho Frobenius norm per matrix.",
    },
    {
        "slot_id": "S3_P7", "role": "exploit", "hypothesis_id": "H54",
        "title": "Cautious WD on raw gradient alignment",
        "code_mutation": "H54_raw_grad_wd",
        "mechanism": "Use (raw_grads * params) >= 0 for cautious mask instead of (processed_update * params) >= 0.",
    },
    {
        "slot_id": "S3_P8", "role": "wildcard", "hypothesis_id": "H36",
        "title": "Asymmetric trust — soft floor, hard ceiling",
        "code_mutation": "H36_asymmetric_trust",
        "mechanism": "Soft sigmoid floor at 0.5 when trust<1, hard linear clamp at 1.3 when trust>1.",
    },
]


# ── Stage3Optimizer: all code mutations ──────────────────────

class Stage3Optimizer(SpecCandidateOptimizer):
    """Extended optimizer with Stage 3 code-level mutations."""

    def __init__(self, param_groups: list[dict], spec: MatrixOptimizerSpec,
                 code_mutation: str = "none") -> None:
        super().__init__(param_groups, spec)
        self.code_mutation = code_mutation

    # ── H39: Per-neuron trust ratio ──────────────────────────
    def _apply_trust_ratio(self, update: Tensor, params: Tensor,
                           *, trust_mix: float = 1.0) -> tuple[Tensor, Tensor]:
        if self.code_mutation in ("H39_perneuron_trust", "COMPOUND_H41_H39_H46"):
            return self._apply_trust_ratio_perneuron(update, params, trust_mix=trust_mix)
        if self.code_mutation == "H36_asymmetric_trust":
            return self._apply_trust_ratio_asymmetric(update, params, trust_mix=trust_mix)
        return super()._apply_trust_ratio(update, params, trust_mix=trust_mix)

    def _apply_trust_ratio_perneuron(self, update: Tensor, params: Tensor,
                                      *, trust_mix: float = 1.0) -> tuple[Tensor, Tensor]:
        if self.spec.trust_ratio.mode == "none":
            ones = torch.ones(update.size(0), device=update.device, dtype=update.dtype)
            return update, ones
        eps = self.spec.trust_ratio.eps
        # Tall (rows > cols): trust per-column (dim=-2), shape (N, 1, cols)
        # Wide (cols > rows): trust per-row (dim=-1), shape (N, rows, 1)
        if update.size(-2) > update.size(-1):
            param_norm = params.float().norm(dim=-2, keepdim=True).clamp_min(eps)
            update_norm = update.float().norm(dim=-2, keepdim=True).clamp_min(eps)
        else:
            param_norm = params.float().norm(dim=-1, keepdim=True).clamp_min(eps)
            update_norm = update.float().norm(dim=-1, keepdim=True).clamp_min(eps)
        trust = (param_norm / update_norm).clamp(self.spec.trust_ratio.clamp_min,
                                                  self.spec.trust_ratio.clamp_max)
        trust = 1.0 + trust_mix * (trust - 1.0)
        # Per-layer mean for stats reporting
        trust_mean = trust.float().mean(dim=(-2, -1))
        return update * trust.to(dtype=update.dtype), trust_mean

    # ── H36: Asymmetric trust ────────────────────────────────
    def _apply_trust_ratio_asymmetric(self, update: Tensor, params: Tensor,
                                       *, trust_mix: float = 1.0) -> tuple[Tensor, Tensor]:
        if self.spec.trust_ratio.mode == "none":
            ones = torch.ones(update.size(0), device=update.device, dtype=update.dtype)
            return update, ones
        eps = self.spec.trust_ratio.eps
        param_norm = params.float().flatten(1).norm(dim=1).clamp_min(eps)
        update_norm = update.float().flatten(1).norm(dim=1).clamp_min(eps)
        raw_trust = param_norm / update_norm
        # Soft floor via sigmoid when trust < 1 (update too large)
        # Hard ceiling at 1.3 when trust > 1 (update too small)
        trust = torch.where(
            raw_trust < 1.0,
            0.5 + 0.5 * torch.sigmoid(4.0 * (raw_trust - 0.5)),
            raw_trust.clamp_max(1.3),
        )
        trust = 1.0 + trust_mix * (trust - 1.0)
        reshape = trust.view(-1, *([1] * (update.ndim - 1)))
        return update * reshape.to(dtype=update.dtype), trust

    # ── H42: Second moment bias correction ───────────────────
    def _apply_second_moment(self, update: Tensor, second_moment_buffer: Tensor,
                              *, beta2_override: float | None = None) -> Tensor:
        if self.code_mutation == "H42_bias_correction":
            return self._apply_second_moment_biascorr(update, second_moment_buffer,
                                                       beta2_override=beta2_override)
        return super()._apply_second_moment(update, second_moment_buffer,
                                             beta2_override=beta2_override)

    def _apply_second_moment_biascorr(self, update: Tensor, second_moment_buffer: Tensor,
                                       *, beta2_override: float | None = None) -> Tensor:
        if self.spec.second_moment.mode == "none":
            return update
        red_dim = -1 if update.size(-2) >= update.size(-1) else -2
        v_mean = update.float().square().mean(dim=red_dim, keepdim=True)
        beta2 = self.spec.second_moment.beta2 if beta2_override is None else beta2_override
        second_moment_buffer.lerp_(v_mean.to(dtype=second_moment_buffer.dtype), 1 - beta2)
        # Bias correction for early steps
        step = max(1, int(self._step_context.get("step", 1)))
        if step <= 50:
            correction = 1.0 / (1.0 - beta2 ** step)
            corrected = second_moment_buffer * correction
        else:
            corrected = second_moment_buffer
        step_size = corrected.clamp_min(self.spec.second_moment.eps).rsqrt()
        # Norm-preserving rescale (keep it — NK03 proved it's critical)
        red_dim_size = update.size(red_dim)
        v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(self.spec.second_moment.eps)
        final_scale = step_size * (v_norm / v_norm_new)
        return update * final_scale.to(dtype=update.dtype)

    # ── H46: Decoupled Nesterov blend ────────────────────────
    def _apply_momentum(self, update: Tensor, momentum_buffer: Tensor) -> Tensor:
        if self.code_mutation == "H46_nesterov_decouple":
            momentum_buffer.lerp_(update, 1 - self.spec.momentum)
            if self.spec.nesterov:
                return update.lerp(momentum_buffer, 0.8)  # blend != momentum
            return momentum_buffer
        if self.code_mutation == "H41_momentum_warmup":
            step = max(1, int(self._step_context.get("step", 1)))
            warmup_steps = 200
            beta = 0.85 + (self.spec.momentum - 0.85) * min(1.0, step / warmup_steps)
            momentum_buffer.lerp_(update, 1 - beta)
            if self.spec.nesterov:
                return update.lerp(momentum_buffer, beta)
            return momentum_buffer
        if self.code_mutation in ("COMPOUND_H41_H46", "COMPOUND_H41_H39_H46"):
            # H41: momentum warmup 0.85→0.95 over 200 steps
            # H46: decoupled Nesterov blend at 0.8
            step = max(1, int(self._step_context.get("step", 1)))
            warmup_steps = 200
            beta = 0.85 + (self.spec.momentum - 0.85) * min(1.0, step / warmup_steps)
            momentum_buffer.lerp_(update, 1 - beta)
            if self.spec.nesterov:
                return update.lerp(momentum_buffer, 0.8)  # H46 blend, not beta
            return momentum_buffer
        return super()._apply_momentum(update, momentum_buffer)

    # ── H44: Post-ortho scale preservation ───────────────────
    def _orthogonalize(self, update: Tensor, *, orthogonal_mix: float = 1.0) -> Tensor:
        if self.code_mutation == "H44_scale_preserve":
            return self._orthogonalize_scale_preserve(update, orthogonal_mix=orthogonal_mix)
        return super()._orthogonalize(update, orthogonal_mix=orthogonal_mix)

    def _orthogonalize_scale_preserve(self, update: Tensor, *, orthogonal_mix: float = 1.0) -> Tensor:
        if self.spec.orthogonalization == "none":
            return update
        raw_update = update
        # Measure pre-ortho norm per matrix
        pre_norm = update.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        x = update.to(dtype=torch.float32)
        x = x / (x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6) * 1.02)
        for a, b, c in POLAR_EXPRESS_COEFFS[:self.spec.ns_steps]:
            if x.size(-2) > x.size(-1):
                gram = x.mT @ x
                x = a * x + x @ (b * gram + c * (gram @ gram))
            else:
                gram = x @ x.mT
                x = a * x + (b * gram + c * (gram @ gram)) @ x
        orthogonalized = x.to(dtype=update.dtype)
        # Rescale to preserve pre-ortho Frobenius norm
        post_norm = orthogonalized.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        scale = (pre_norm / post_norm).view(-1, *([1] * (update.ndim - 1)))
        orthogonalized = orthogonalized * scale.to(dtype=update.dtype)
        if orthogonal_mix >= 1.0:
            return orthogonalized
        if orthogonal_mix <= 0.0:
            return raw_update
        return orthogonal_mix * orthogonalized + (1.0 - orthogonal_mix) * raw_update

    # ── H59, H54: Weight decay mutations ─────────────────────
    # These require overriding _step_matrix_group since WD is inline
    def _step_matrix_group(self, group: dict) -> OptimizerStepStats:
        if self.code_mutation in ("H59_soft_wd", "H54_raw_grad_wd"):
            return self._step_matrix_group_wd_variant(group)
        return super()._step_matrix_group(group)

    def _step_matrix_group_wd_variant(self, group: dict) -> OptimizerStepStats:
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return OptimizerStepStats()

        anchor = params[0]
        state = self.state[anchor]
        shape = tuple(anchor.shape)
        if state.get("shape") != shape or state.get("count") != len(params):
            state["shape"] = shape
            state["count"] = len(params)
            state["momentum_buffer"] = torch.zeros((len(params), *shape),
                                                    dtype=anchor.dtype, device=anchor.device)
            red_dim = -1 if shape[-2] >= shape[-1] else -2
            state_shape = (len(params), shape[-2], 1) if red_dim == -1 else (len(params), 1, shape[-1])
            state["second_moment_buffer"] = torch.zeros(state_shape,
                                                         dtype=anchor.dtype, device=anchor.device)

        grads = torch.stack([p.grad.detach() for p in params])
        stacked_params = torch.stack([p.detach() for p in params])
        gate = self._gate_value()
        orthogonal_mix = self._actuator_value("orthogonal_mix", gate) if self._stateful_enabled() else 1.0
        trust_mix = self._actuator_value("trust_ratio_mix", gate) if self._stateful_enabled() else 1.0
        clip_threshold = self._actuator_value("clip_threshold", gate) if self._stateful_enabled() else None
        beta2_override = self._actuator_value("beta2", gate) if self._stateful_enabled() else None
        update_multiplier = self.spec.update_multiplier
        if self._stateful_enabled():
            update_multiplier *= self._actuator_value("update_multiplier", gate)

        if self.spec.momentum_placement == "pre_orthogonal":
            update = self._apply_momentum(grads, state["momentum_buffer"])
            update = self._orthogonalize(update, orthogonal_mix=orthogonal_mix)
        else:
            update = self._orthogonalize(grads, orthogonal_mix=orthogonal_mix)
            update = self._apply_momentum(update, state["momentum_buffer"])

        update = self._apply_second_moment(update, state["second_moment_buffer"],
                                            beta2_override=beta2_override)
        update, trust = self._apply_trust_ratio(update, stacked_params, trust_mix=trust_mix)
        update = self._apply_clip(update, threshold_override=clip_threshold)
        update = update * update_multiplier

        lr = float(group["lr"])
        if self.spec.lr_aspect_scale:
            lr *= math.sqrt(max(1.0, stacked_params.size(-2) / stacked_params.size(-1)))

        # Weight decay variant
        if self.spec.decay.mode == "cautious" and self.spec.decay.weight_decay:
            if self.code_mutation == "H59_soft_wd":
                # Soft sigmoid mask instead of binary
                alignment = update * stacked_params
                scale = alignment.abs().mean().clamp_min(1e-8)
                soft_mask = torch.sigmoid(alignment / scale * 5.0)
                stacked_params.sub_(lr * update + lr * self.spec.decay.weight_decay * stacked_params * soft_mask)
            elif self.code_mutation == "H54_raw_grad_wd":
                # Use raw gradient alignment instead of processed update
                mask = (grads * stacked_params) >= 0
                stacked_params.sub_(lr * update + lr * self.spec.decay.weight_decay * stacked_params * mask)
        elif self.spec.decay.mode == "decoupled" and self.spec.decay.weight_decay:
            stacked_params.mul_(1 - lr * self.spec.decay.weight_decay)
            stacked_params.add_(update, alpha=-lr)
        else:
            stacked_params.add_(update, alpha=-lr)

        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

        param_norm = stacked_params.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        update_norm = update.float().flatten(1).norm(dim=1)
        ratios = update_norm / param_norm
        grad_norm = grads.float().flatten(1).norm(dim=1)
        return OptimizerStepStats(
            mean_update_param_ratio=float(ratios.mean().item()),
            max_update_param_ratio=float(ratios.max().item()),
            mean_trust_ratio=float(trust.float().mean().item()),
            max_grad_norm=float(grad_norm.max().item()),
            mean_gate=gate, groups_touched=1,
        )


# ── Build optimizer ──────────────────────────────────────────

def build_slot_optimizer(model, spec: MatrixOptimizerSpec, code_mutation: str = "none"):
    param_groups = build_nanochat_param_groups(model, weight_decay=spec.decay.weight_decay)
    return Stage3Optimizer(param_groups, spec, code_mutation=code_mutation)


# ── Toy screening ────────────────────────────────────────────

def run_toy_stage(run_dir: Path) -> dict:
    print(f"\n{'='*70}")
    print(f"  ENIGMA Stage 3 — Deep Code Mutations (Toy Screening)")
    print(f"  Portfolio: {len(STAGE3_PORTFOLIO)} hypotheses")
    print(f"{'='*70}\n")

    specs_dir = run_dir / "specs"
    evals_dir = run_dir / "evaluations"
    doctrine_dir = run_dir / "doctrine"
    for d in [specs_dir, evals_dir, doctrine_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save doctrine artifacts
    (doctrine_dir / "portfolio.json").write_text(json.dumps(STAGE3_PORTFOLIO, indent=2))

    eval_config = EvaluationConfig(seed=42, steps=24, eval_every=6)
    backend = ToyNanoChatBackend(eval_config)

    # Baseline = H02 trust ratio winner
    spec = base_spec()
    spec_with_meta = replace(spec, name="stage3_baseline_h02",
                              metadata={"stage": 3, "baseline": True, "code_mutation": "none"})
    (specs_dir / "baseline_h02.json").write_text(spec_with_meta.to_json())

    print("Evaluating baseline (H02 trust ratio)...")
    baseline_outcome = backend.evaluate(spec_with_meta, seed=42, candidate_id="baseline_h02")
    assert baseline_outcome.valid
    base_bpb = baseline_outcome.metrics.final_validation_bpb
    print(f"  Baseline BPB: {base_bpb:.4f}")

    results = []
    for slot in STAGE3_PORTFOLIO:
        slot_id = slot["slot_id"]
        code_mutation = slot["code_mutation"]

        slot_spec = replace(spec_with_meta,
                            name=f"enigma_{slot['hypothesis_id']}",
                            metadata={"stage": 3, "slot": slot_id,
                                      "hypothesis": slot["hypothesis_id"],
                                      "code_mutation": code_mutation})
        (specs_dir / f"{slot_id}_{slot['hypothesis_id']}.json").write_text(slot_spec.to_json())

        def opt_factory(model, sp, cm=code_mutation):
            return build_slot_optimizer(model, sp, cm)

        t0 = time.perf_counter()
        outcome = backend.evaluate(slot_spec, seed=42, candidate_id=f"enigma_{slot_id}",
                                   optimizer_factory=opt_factory)
        t_eval = time.perf_counter() - t0

        delta = base_bpb - outcome.metrics.final_validation_bpb
        result = {
            "slot": slot_id, "hypothesis": slot["hypothesis_id"],
            "role": slot["role"], "title": slot["title"],
            "code_mutation": code_mutation,
            "valid": outcome.valid,
            "final_validation_bpb": outcome.metrics.final_validation_bpb,
            "delta_vs_baseline": delta,
            "eval_time_s": t_eval,
        }
        results.append(result)
        (evals_dir / f"{slot_id}_toy.json").write_text(json.dumps(result, indent=2))

        marker = "WIN" if delta > 0.0005 else ("~" if abs(delta) < 0.0005 else " x ")
        print(f"  [{marker}] {slot_id} ({slot['role']:>8s}) {slot['title']:45s} "
              f"bpb={outcome.metrics.final_validation_bpb:.4f} Δ={delta:+.4f}")

    (run_dir / "toy_results.json").write_text(json.dumps(results, indent=2))
    valid = all(r["valid"] for r in results)
    print(f"\nAll valid: {valid}")
    print(f"Artifacts: {run_dir}")
    return {"results": results, "all_valid": valid}


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Enigma Stage 3")
    parser.add_argument("--stage", choices=["toy", "real"], default="toy")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.run_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_root = Path(__file__).resolve().parent.parent
        args.run_dir = repo_root / "runs" / f"enigma_s3_{ts}"
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "toy":
        run_toy_stage(args.run_dir)


if __name__ == "__main__":
    main()
