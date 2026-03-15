#!/usr/bin/env python3
"""Enigma Stage 2: Doctrine-driven code-level optimizer mutations.

Follows the full Enigma doctrine loop:
  Steps 1-2: Variable + Surface mapping (14 surfaces identified)
  Step 3: 15 hypotheses generated (H10-H24)
  Step 4: Pruned to 9 KEEP, 3 KILL, 3 DEMOTE
  Step 5: Gap analysis (all high-leverage BPB surfaces covered)
  Step 6: Portfolio of 7 selected (3 scout, 3 exploit, 1 wildcard)
  Step 7: Neighborhoods registered, no overlaps with Stage 1

Portfolio:
  Slot 1 (scout)    H10: bfloat16 Polar Express dtype (match production)
  Slot 2 (scout)    H20: Widen trust ratio clamp [0.1, 8.0]
  Slot 3 (scout)    H18: Remove norm-preserving rescale from second_moment
  Slot 4 (exploit)  H19: Amplify gate step_fraction annealing
  Slot 5 (exploit)  H11: Reorder trust_ratio before second_moment
  Slot 6 (exploit)  H16: Cautious WD gated by trust threshold
  Slot 7 (wildcard) H23: Disable second_moment entirely

Base: merged P5 (trust+clip) + P3 (stateful gate) = "enigma_stage2_merged"

Usage:
    python -m enigma.run_stage2 --stage toy
    python -m enigma.run_stage2 --stage real --host user54@35.84.33.219
    python -m enigma.run_stage2 --stage score --run-dir runs/enigma_s2_XXXXX
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "adamopt"))

from optim_search.spec import (
    AdaptiveActuatorConfig,
    AdaptiveRange,
    ClipConfig,
    DecayConfig,
    GateCoefficients,
    GateConfig,
    MatrixOptimizerSpec,
    SecondMomentConfig,
    StatefulControlConfig,
    TrustRatioConfig,
)
from optim_search.candidate_optimizer import (
    POLAR_EXPRESS_COEFFS,
    SpecCandidateOptimizer,
    build_nanochat_param_groups,
)
from optim_search.eval_candidate import ToyNanoChatBackend
from optim_search.config import EvaluationConfig
from optim_search.score import composite_score, analyze_win_hierarchy


# ── Merged baseline from Stage 1 winners ─────────────────────

def merged_winner_spec() -> MatrixOptimizerSpec:
    """Merge P5 (trust+clip) + P3 (stateful gate) into a single spec."""
    spec = MatrixOptimizerSpec(
        name="enigma_stage2_merged",
        momentum=0.95,
        nesterov=True,
        momentum_placement="pre_orthogonal",
        orthogonalization="polar_express",
        ns_steps=5,
        trust_ratio=TrustRatioConfig(mode="layerwise", clamp_min=0.5, clamp_max=1.5),
        clip=ClipConfig(mode="update_rms", threshold=1.0),
        decay=DecayConfig(mode="cautious", weight_decay=0.2),
        second_moment=SecondMomentConfig(mode="factored_rms", beta2=0.95, eps=1e-10),
        stateful_control=StatefulControlConfig(
            enabled=True, ema_beta=0.9,
            loss_normalizer=2.0, improvement_normalizer=0.05,
            grad_norm_normalizer=1.0, update_ratio_normalizer=0.05,
            gate=GateConfig(
                coefficients=GateCoefficients(
                    loss_ema=0.4, loss_improvement_ema=-0.8,
                    grad_norm_ema=0.5, update_ratio_ema=0.3,
                    grad_alignment_ema=0.7, step_fraction=-1.0,
                ),
                bias=-0.2, sharpness=1.2,
            ),
            actuators=AdaptiveActuatorConfig(
                update_multiplier=AdaptiveRange(aggressive=1.15, conservative=0.8),
                trust_ratio_mix=AdaptiveRange(aggressive=0.8, conservative=0.25),
                clip_threshold=AdaptiveRange(aggressive=1.5, conservative=0.65),
                beta2=AdaptiveRange(aggressive=0.9, conservative=0.985),
                orthogonal_mix=AdaptiveRange(aggressive=1.0, conservative=0.45),
            ),
        ),
        update_multiplier=1.0,
        lr_aspect_scale=True,
        metadata={"stage": 2, "parent": "P5+P3 merge"},
    )
    spec.validate()
    return spec


# ── Portfolio: 7 doctrine-driven hypotheses ───────────────────

STAGE2_PORTFOLIO = [
    {
        "slot_id": "S2_P1", "role": "scout", "hypothesis_id": "H10",
        "title": "bfloat16 Polar Express dtype (match production)",
        "code_mutation": "H10_bf16_polar",
        "mechanism": "Production Muon casts to bfloat16 for Polar Express; candidate uses float32. Coefficients were tuned for bf16 arithmetic.",
        "spec_mutation": None,  # code change only
    },
    {
        "slot_id": "S2_P2", "role": "scout", "hypothesis_id": "H20",
        "title": "Widen trust ratio clamp [0.1, 8.0]",
        "code_mutation": None,  # spec change only
        "mechanism": "Stage 1 trust won with tight [0.5, 1.5]. Test if wider range amplifies the win.",
        "spec_mutation": {"trust_ratio": {"clamp_min": 0.1, "clamp_max": 8.0}},
    },
    {
        "slot_id": "S2_P3", "role": "scout", "hypothesis_id": "H18",
        "title": "Simplified second moment (no norm-preserving rescale)",
        "code_mutation": "H18_simple_second_moment",
        "mechanism": "Remove norm-preserving rescale from _apply_second_moment. Let trust_ratio handle magnitude.",
        "spec_mutation": None,
    },
    {
        "slot_id": "S2_P4", "role": "exploit", "hypothesis_id": "H19",
        "title": "Amplify gate step_fraction annealing",
        "code_mutation": None,  # spec change only
        "mechanism": "Push step_fraction from -1.0 to -2.0, bias +0.5, sharpness 1.5. Stronger early-aggressive/late-conservative.",
        "spec_mutation": {
            "stateful_control": {
                "gate": {
                    "coefficients": {"step_fraction": -2.0},
                    "bias": 0.5,
                    "sharpness": 1.5,
                }
            }
        },
    },
    {
        "slot_id": "S2_P5", "role": "exploit", "hypothesis_id": "H11",
        "title": "Reorder: trust_ratio before second_moment",
        "code_mutation": "H11_trust_before_second",
        "mechanism": "Trust ratio should see raw orthogonalized norms, not second_moment-rescaled ones.",
        "spec_mutation": None,
    },
    {
        "slot_id": "S2_P6", "role": "exploit", "hypothesis_id": "H16",
        "title": "Cautious WD gated by trust threshold",
        "code_mutation": "H16_trust_gated_wd",
        "mechanism": "Only apply cautious WD where trust >= 0.8, preventing double-penalization of constrained neurons.",
        "spec_mutation": None,
    },
    {
        "slot_id": "S2_P7", "role": "wildcard", "hypothesis_id": "H23",
        "title": "Disable second_moment, rely on trust+clip",
        "code_mutation": None,  # spec change only
        "mechanism": "Tests whether NorMuon variance reduction is essential when trust+clip are present.",
        "spec_mutation": {"second_moment": {"mode": "none"}},
    },
]


# ── Stage2Optimizer: implements code-level mutations ──────────

class Stage2Optimizer(SpecCandidateOptimizer):
    """Extended optimizer supporting code-level mutations from Stage 2 doctrine."""

    def __init__(self, param_groups: list[dict], spec: MatrixOptimizerSpec,
                 code_mutation: str = "none") -> None:
        super().__init__(param_groups, spec)
        self.code_mutation = code_mutation
        # For H16: store pre-trust update direction
        self._pre_trust_update: Tensor | None = None

    # ── H10: bfloat16 Polar Express ──────────────────────────
    def _orthogonalize(self, update: Tensor, *, orthogonal_mix: float = 1.0) -> Tensor:
        if self.code_mutation == "H10_bf16_polar":
            return self._orthogonalize_bf16(update, orthogonal_mix=orthogonal_mix)
        return super()._orthogonalize(update, orthogonal_mix=orthogonal_mix)

    def _orthogonalize_bf16(self, update: Tensor, *, orthogonal_mix: float = 1.0) -> Tensor:
        """Match production: bfloat16 instead of float32 for Polar Express."""
        if self.spec.orthogonalization == "none":
            return update
        raw_update = update
        # KEY CHANGE: bfloat16 instead of float32 (matches optim.py line 115)
        x = update.to(dtype=torch.bfloat16)
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        for a, b, c in POLAR_EXPRESS_COEFFS[:self.spec.ns_steps]:
            if x.size(-2) > x.size(-1):
                gram = x.mT @ x
                x = a * x + x @ (b * gram + c * (gram @ gram))
            else:
                gram = x @ x.mT
                x = a * x + (b * gram + c * (gram @ gram)) @ x
        orthogonalized = x.to(dtype=update.dtype)
        if orthogonal_mix >= 1.0:
            return orthogonalized
        if orthogonal_mix <= 0.0:
            return raw_update
        return orthogonal_mix * orthogonalized + (1.0 - orthogonal_mix) * raw_update

    # ── H18: Simplified second moment ────────────────────────
    def _apply_second_moment(self, update: Tensor, second_moment_buffer: Tensor,
                              *, beta2_override: float | None = None) -> Tensor:
        if self.code_mutation == "H18_simple_second_moment":
            return self._apply_second_moment_simple(update, second_moment_buffer,
                                                     beta2_override=beta2_override)
        return super()._apply_second_moment(update, second_moment_buffer,
                                             beta2_override=beta2_override)

    def _apply_second_moment_simple(self, update: Tensor, second_moment_buffer: Tensor,
                                     *, beta2_override: float | None = None) -> Tensor:
        """Remove norm-preserving rescale — just per-neuron rsqrt scaling."""
        if self.spec.second_moment.mode == "none":
            return update
        red_dim = -1 if update.size(-2) >= update.size(-1) else -2
        v_mean = update.float().square().mean(dim=red_dim, keepdim=True)
        beta2 = self.spec.second_moment.beta2 if beta2_override is None else beta2_override
        second_moment_buffer.lerp_(v_mean.to(dtype=second_moment_buffer.dtype), 1 - beta2)
        # KEY CHANGE: direct rsqrt without norm-preserving correction
        step_size = second_moment_buffer.clamp_min(self.spec.second_moment.eps).rsqrt()
        return update * step_size.to(dtype=update.dtype)

    # ── H11: Reorder trust before second_moment ──────────────
    def _step_matrix_group(self, group: dict) -> "OptimizerStepStats":
        if self.code_mutation == "H11_trust_before_second":
            return self._step_matrix_group_reordered(group)
        elif self.code_mutation == "H16_trust_gated_wd":
            return self._step_matrix_group_trust_gated_wd(group)
        return super()._step_matrix_group(group)

    def _step_matrix_group_reordered(self, group: dict) -> "OptimizerStepStats":
        """Pipeline reorder: trust_ratio BEFORE second_moment."""
        from optim_search.candidate_optimizer import OptimizerStepStats
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return OptimizerStepStats()

        anchor = params[0]
        state = self.state[anchor]
        shape = tuple(anchor.shape)
        if state.get("shape") != shape or state.get("count") != len(params):
            state["shape"] = shape
            state["count"] = len(params)
            state["momentum_buffer"] = torch.zeros((len(params), *shape), dtype=anchor.dtype, device=anchor.device)
            red_dim = -1 if shape[-2] >= shape[-1] else -2
            state_shape = (len(params), shape[-2], 1) if red_dim == -1 else (len(params), 1, shape[-1])
            state["second_moment_buffer"] = torch.zeros(state_shape, dtype=anchor.dtype, device=anchor.device)

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

        # KEY CHANGE: trust_ratio BEFORE second_moment
        update, trust = self._apply_trust_ratio(update, stacked_params, trust_mix=trust_mix)
        update = self._apply_second_moment(update, state["second_moment_buffer"], beta2_override=beta2_override)
        update = self._apply_clip(update, threshold_override=clip_threshold)
        update = update * update_multiplier

        lr = float(group["lr"])
        if self.spec.lr_aspect_scale:
            lr *= math.sqrt(max(1.0, stacked_params.size(-2) / stacked_params.size(-1)))

        if self.spec.decay.mode == "cautious" and self.spec.decay.weight_decay:
            mask = (update * stacked_params) >= 0
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

    # ── H16: Cautious WD gated by trust threshold ────────────
    def _step_matrix_group_trust_gated_wd(self, group: dict) -> "OptimizerStepStats":
        """Only apply cautious WD where trust >= 0.8."""
        from optim_search.candidate_optimizer import OptimizerStepStats
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return OptimizerStepStats()

        anchor = params[0]
        state = self.state[anchor]
        shape = tuple(anchor.shape)
        if state.get("shape") != shape or state.get("count") != len(params):
            state["shape"] = shape
            state["count"] = len(params)
            state["momentum_buffer"] = torch.zeros((len(params), *shape), dtype=anchor.dtype, device=anchor.device)
            red_dim = -1 if shape[-2] >= shape[-1] else -2
            state_shape = (len(params), shape[-2], 1) if red_dim == -1 else (len(params), 1, shape[-1])
            state["second_moment_buffer"] = torch.zeros(state_shape, dtype=anchor.dtype, device=anchor.device)

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

        update = self._apply_second_moment(update, state["second_moment_buffer"], beta2_override=beta2_override)
        update, trust = self._apply_trust_ratio(update, stacked_params, trust_mix=trust_mix)
        update = self._apply_clip(update, threshold_override=clip_threshold)
        update = update * update_multiplier

        lr = float(group["lr"])
        if self.spec.lr_aspect_scale:
            lr *= math.sqrt(max(1.0, stacked_params.size(-2) / stacked_params.size(-1)))

        # KEY CHANGE: only apply cautious WD where trust >= 0.8
        if self.spec.decay.mode == "cautious" and self.spec.decay.weight_decay:
            alignment_mask = (update * stacked_params) >= 0
            trust_reshape = trust.view(-1, *([1] * (update.ndim - 1)))
            trust_mask = trust_reshape >= 0.8  # only decay where trust is high enough
            combined_mask = alignment_mask & trust_mask
            stacked_params.sub_(lr * update + lr * self.spec.decay.weight_decay * stacked_params * combined_mask)
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


# ── Build spec + optimizer for each slot ──────────────────────

def build_slot_spec(slot: dict, base: MatrixOptimizerSpec) -> MatrixOptimizerSpec:
    """Apply spec-level mutations for a portfolio slot."""
    spec = base
    sm = slot.get("spec_mutation")
    if not sm:
        return replace(spec,
                       name=f"enigma_{slot['hypothesis_id']}",
                       metadata={**spec.metadata, "slot": slot["slot_id"], "hypothesis": slot["hypothesis_id"],
                                 "code_mutation": slot.get("code_mutation", "none")})

    # Apply spec mutations
    if "trust_ratio" in sm:
        tr = spec.trust_ratio
        spec = replace(spec, trust_ratio=replace(tr, **sm["trust_ratio"]))
    if "second_moment" in sm:
        sm2 = spec.second_moment
        spec = replace(spec, second_moment=replace(sm2, **sm["second_moment"]))
    if "stateful_control" in sm:
        sc = spec.stateful_control
        sc_mut = sm["stateful_control"]
        if "gate" in sc_mut:
            g = sc.gate
            g_mut = sc_mut["gate"]
            if "coefficients" in g_mut:
                coeffs = replace(g.coefficients, **g_mut["coefficients"])
                g = replace(g, coefficients=coeffs)
            g = replace(g, **{k: v for k, v in g_mut.items() if k != "coefficients"})
            sc = replace(sc, gate=g)
        spec = replace(spec, stateful_control=sc)

    return replace(spec,
                   name=f"enigma_{slot['hypothesis_id']}",
                   metadata={**spec.metadata, "slot": slot["slot_id"], "hypothesis": slot["hypothesis_id"],
                             "code_mutation": slot.get("code_mutation", "none")})


def build_slot_optimizer(model, spec: MatrixOptimizerSpec, code_mutation: str = "none"):
    """Build optimizer for a slot, using Stage2Optimizer for code mutations."""
    param_groups = build_nanochat_param_groups(model, weight_decay=spec.decay.weight_decay)
    if code_mutation and code_mutation != "none":
        return Stage2Optimizer(param_groups, spec, code_mutation=code_mutation)
    return Stage2Optimizer(param_groups, spec, code_mutation="none")


# ── Toy evaluation ────────────────────────────────────────────

def run_toy_stage(run_dir: Path) -> dict:
    """Run Stage 2 doctrine portfolio through toy backend."""
    print(f"\n{'='*70}")
    print(f"  ENIGMA Stage 2 — Doctrine-Driven Code Mutations (Toy)")
    print(f"  Portfolio: {len(STAGE2_PORTFOLIO)} hypotheses")
    print(f"  Run dir: {run_dir}")
    print(f"{'='*70}\n")

    specs_dir = run_dir / "specs"
    evals_dir = run_dir / "evaluations"
    loop_dir = run_dir / "loop_002"
    doctrine_dir = run_dir / "doctrine"
    for d in [specs_dir, evals_dir, loop_dir, doctrine_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Save doctrine artifacts
    _save_doctrine_artifacts(doctrine_dir)

    eval_config = EvaluationConfig(seed=42, steps=24, eval_every=6)
    backend = ToyNanoChatBackend(eval_config)

    # Stage 1 baseline
    stage1_baseline = MatrixOptimizerSpec.baseline_nanochat()
    print("Evaluating Stage 1 baseline...")
    baseline_outcome = backend.evaluate(stage1_baseline, seed=42, candidate_id="stage1_baseline")
    assert baseline_outcome.valid
    (evals_dir / "stage1_baseline_toy.json").write_text(json.dumps({
        "candidate_id": "stage1_baseline", "valid": True,
        "final_validation_bpb": baseline_outcome.metrics.final_validation_bpb,
    }, indent=2))

    # Merged baseline
    merged = merged_winner_spec()
    merged.validate()
    (specs_dir / "merged_baseline.json").write_text(merged.to_json())

    print("Evaluating merged baseline (P5+P3)...")
    merged_outcome = backend.evaluate(merged, seed=42, candidate_id="merged_baseline")
    delta_m = baseline_outcome.metrics.final_validation_bpb - merged_outcome.metrics.final_validation_bpb
    print(f"  Merged: bpb={merged_outcome.metrics.final_validation_bpb:.4f} Δ={delta_m:+.4f}")
    (evals_dir / "merged_baseline_toy.json").write_text(json.dumps({
        "candidate_id": "merged_baseline", "valid": merged_outcome.valid,
        "final_validation_bpb": merged_outcome.metrics.final_validation_bpb,
    }, indent=2))

    # Run portfolio
    print(f"\nEvaluating {len(STAGE2_PORTFOLIO)} doctrine-selected mutations...")
    results = []

    for slot in STAGE2_PORTFOLIO:
        slot_id = slot["slot_id"]
        code_mutation = slot.get("code_mutation") or "none"

        # Build spec
        spec = build_slot_spec(slot, merged)
        spec.validate()
        spec_file = specs_dir / f"{slot_id}_{slot['hypothesis_id']}.json"
        spec_file.write_text(spec.to_json())

        # Build optimizer factory
        def opt_factory(model, sp, cm=code_mutation):
            return build_slot_optimizer(model, sp, cm)

        # Evaluate
        t0 = time.perf_counter()
        outcome = backend.evaluate(spec, seed=42, candidate_id=f"enigma_{slot_id}",
                                   optimizer_factory=opt_factory)
        t_eval = time.perf_counter() - t0

        # Score vs stage 1 baseline
        score = composite_score(outcome, baseline_outcome)
        win = analyze_win_hierarchy(outcome, baseline_outcome)

        delta_vs_s1 = baseline_outcome.metrics.final_validation_bpb - outcome.metrics.final_validation_bpb
        delta_vs_merged = merged_outcome.metrics.final_validation_bpb - outcome.metrics.final_validation_bpb

        result = {
            "slot": slot_id, "hypothesis": slot["hypothesis_id"],
            "role": slot["role"], "title": slot["title"],
            "code_mutation": code_mutation,
            "mechanism": slot["mechanism"],
            "spec_mutation": slot.get("spec_mutation"),
            "valid": outcome.valid,
            "final_validation_bpb": outcome.metrics.final_validation_bpb,
            "delta_vs_stage1_baseline": delta_vs_s1,
            "delta_vs_merged": delta_vs_merged,
            "composite_score": score,
            "winner_vs_baseline": win.winner,
            "speed_ratio": baseline_outcome.metrics.mean_step_time_ms / max(outcome.metrics.mean_step_time_ms, 1e-8),
            "tokens_per_sec": outcome.metrics.tokens_per_sec,
            "stability_penalty": outcome.metrics.stability_penalty,
            "eval_time_s": t_eval,
        }
        results.append(result)
        (evals_dir / f"{slot_id}_toy.json").write_text(json.dumps(result, indent=2))

        marker = "WIN" if delta_vs_merged > 0.0005 else ("~" if abs(delta_vs_merged) < 0.0005 else " x ")
        print(f"  [{marker}] {slot_id} ({slot['role']:>8s}) {slot['title']:50s} "
              f"bpb={outcome.metrics.final_validation_bpb:.4f} "
              f"Δs1={delta_vs_s1:+.4f} Δmerge={delta_vs_merged:+.4f}")

    # Save & rank
    (loop_dir / "results.json").write_text(json.dumps(results, indent=2))
    ranked = sorted(results, key=lambda r: r["composite_score"], reverse=True)

    print(f"\n{'='*70}")
    print(f"  RANKING (by composite score vs Stage 1 baseline)")
    print(f"{'='*70}")
    for i, r in enumerate(ranked, 1):
        print(f"  #{i} {r['slot']:>6s} ({r['hypothesis']}) score={r['composite_score']:+.1f} "
              f"Δs1={r['delta_vs_stage1_baseline']:+.4f} Δmerge={r['delta_vs_merged']:+.4f} "
              f"{r['title']}")

    # Postmortem
    best = ranked[0]
    worst = ranked[-1]
    postmortem = {
        "loop": 2, "stage": "code_mutations", "doctrine_steps_completed": 11,
        "stage1_baseline_bpb": baseline_outcome.metrics.final_validation_bpb,
        "merged_baseline_bpb": merged_outcome.metrics.final_validation_bpb,
        "best": {"slot": best["slot"], "hypothesis": best["hypothesis"],
                 "score": best["composite_score"], "delta_vs_merged": best["delta_vs_merged"],
                 "title": best["title"]},
        "worst": {"slot": worst["slot"], "hypothesis": worst["hypothesis"],
                  "score": worst["composite_score"], "delta_vs_merged": worst["delta_vs_merged"],
                  "title": worst["title"]},
        "all_valid": all(r["valid"] for r in results),
        "ranking": [{"slot": r["slot"], "score": r["composite_score"],
                     "delta_vs_merged": r["delta_vs_merged"]} for r in ranked],
        "negative_knowledge": [
            {"hypothesis": r["hypothesis"], "title": r["title"],
             "delta": r["delta_vs_merged"], "mechanism_disproved": r["mechanism"]}
            for r in results if r["delta_vs_merged"] < -0.001
        ],
        "next_step": "Run winners on real NanoChat GPU with --stage real",
    }
    (loop_dir / "postmortem.json").write_text(json.dumps(postmortem, indent=2))

    (run_dir / "config.json").write_text(json.dumps({
        "stage": 2, "backend": "toy", "doctrine_driven": True,
        "eval_config": asdict(eval_config),
        "timestamp": datetime.now().isoformat(),
        "portfolio_size": len(STAGE2_PORTFOLIO),
        "merged_from": ["P5_compound", "P3_stateful"],
        "hypotheses_generated": 15, "hypotheses_pruned": 6, "hypotheses_kept": 9,
        "portfolio_scouts": 3, "portfolio_exploits": 3, "portfolio_wildcards": 1,
    }, indent=2))

    print(f"\nArtifacts saved to: {run_dir}")
    print(f"  specs/     — {len(list(specs_dir.iterdir()))} spec files")
    print(f"  evals/     — {len(list(evals_dir.iterdir()))} evaluation results")
    print(f"  loop_002/  — portfolio, results, postmortem")
    print(f"  doctrine/  — variables, surfaces, hypotheses, pruning, portfolio")
    return postmortem


def _save_doctrine_artifacts(doctrine_dir: Path) -> None:
    """Save the full doctrine state as JSON for auditability."""
    # Portfolio
    (doctrine_dir / "portfolio.json").write_text(json.dumps(STAGE2_PORTFOLIO, indent=2))

    # Hypotheses generated (summary)
    hypotheses = {
        "generated": 15, "kept": 9, "killed": 3, "demoted": 3,
        "killed_ids": {
            "H15": "Weak mechanism — per-matrix norm already handled by Polar Express",
            "H21": "80% overlap with H14 (momentum warmup)",
            "H24": "Diagnostic, not a BPB hypothesis (reveals ns_steps>5 is no-op)",
        },
        "demoted_ids": {
            "H12": "torch.compile too complex for Stage 2, reserve for Stage 3",
            "H17": "Adaptive per-group clip partially covered by stateful control",
            "H22": "Compound change — run components first, then compound in Stage 3",
        },
    }
    (doctrine_dir / "hypotheses.json").write_text(json.dumps(hypotheses, indent=2))

    # Surfaces (summary)
    surfaces = {
        "total": 14,
        "top_by_leverage_x_plausibility": [
            {"id": "S11", "score": 16, "title": "torch.compile fusion", "covered_by": "DEMOTED (H12)"},
            {"id": "S01", "score": 12, "title": "bfloat16 Polar Express", "covered_by": "H10"},
            {"id": "S02", "score": 12, "title": "Reorder trust before second_moment", "covered_by": "H11"},
            {"id": "S12", "score": 12, "title": "Per-neuron trust ratio", "covered_by": "H13 (kept, not in portfolio)"},
            {"id": "S06", "score": 12, "title": "Momentum warmup", "covered_by": "H14 (kept, not in portfolio)"},
        ],
    }
    (doctrine_dir / "surfaces.json").write_text(json.dumps(surfaces, indent=2))

    # Negative knowledge from Stage 1
    negative_knowledge = [
        {"id": "NK01", "hypothesis": "H01", "observed": "Post-orthogonal momentum catastrophic (Δ=-0.131)",
         "cause": "Orthogonalization extremely sensitive to input quality",
         "do_not_repeat": "Never change momentum_placement"},
        {"id": "NK02", "hypothesis": "H05", "observed": "ns_steps=3 lost (Δ=-0.009)",
         "cause": "Orthogonalization quality matters for BPB",
         "do_not_repeat": "Never reduce ns_steps without compensation"},
    ]
    (doctrine_dir / "negative_knowledge.json").write_text(json.dumps(negative_knowledge, indent=2))


# ── Real NanoChat eval ────────────────────────────────────────

def run_real_stage(run_dir: Path, host: str) -> None:
    """Push specs + Stage2Optimizer code to GPU cluster and submit Slurm job."""
    import subprocess

    specs_dir = run_dir / "specs"
    if not list(specs_dir.glob("*.json")):
        print("No specs found. Run --stage toy first.")
        sys.exit(1)

    remote_run_dir = f"~/nanoe/runs/{run_dir.name}"

    print(f"\n{'='*70}")
    print(f"  ENIGMA Stage 2 — Real NanoChat GPU Evaluation")
    print(f"  Host: {host}")
    print(f"  Specs: {len(list(specs_dir.glob('*.json')))} files")
    print(f"{'='*70}\n")

    # Sync code
    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(["rsync", "-az", "--exclude=__pycache__",
                    str(repo_root / "adamopt/"), f"{host}:~/nanoe/adamopt/"],
                   check=True, capture_output=True)
    subprocess.run(["rsync", "-az", "--exclude=__pycache__",
                    str(repo_root / "enigma/"), f"{host}:~/nanoe/enigma/"],
                   check=True, capture_output=True)

    # Create remote dirs and push specs
    subprocess.run(["ssh", host, f"mkdir -p {remote_run_dir}/specs {remote_run_dir}/results"],
                   check=True, capture_output=True)
    for sf in specs_dir.glob("*.json"):
        subprocess.run(["scp", str(sf), f"{host}:{remote_run_dir}/specs/"],
                       check=True, capture_output=True)

    # Write and push Slurm script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=enigma-s2
#SBATCH --partition=priority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output={remote_run_dir}/slurm_%j.out
#SBATCH --error={remote_run_dir}/slurm_%j.err

set -euo pipefail
source ~/nanoe/nanochat/.venv/bin/activate
export PYTHONPATH=~/nanoe:~/nanoe/adamopt:${{PYTHONPATH:-}}
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat

echo "=== ENIGMA Stage 2 Real ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

python -m enigma.run_stage2_remote \\
    --specs-dir {remote_run_dir}/specs \\
    --results-dir {remote_run_dir}/results \\
    --steps 20 --eval-every 10 --depth 4

echo "=== Done ==="
ls -la {remote_run_dir}/results/
"""
    slurm_path = run_dir / "slurm_s2.sh"
    slurm_path.write_text(slurm_script)
    subprocess.run(["scp", str(slurm_path), f"{host}:{remote_run_dir}/"],
                   check=True, capture_output=True)

    # Submit
    result = subprocess.run(["ssh", host, f"sbatch {remote_run_dir}/slurm_s2.sh"],
                           capture_output=True, text=True)
    print(result.stdout.strip())
    print(f"\nTo check: ssh {host} squeue -u user54")
    print(f"To pull:  scp '{host}:{remote_run_dir}/results/*_real.json' {run_dir}/evaluations/")


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Enigma Stage 2")
    parser.add_argument("--stage", choices=["toy", "real", "score"], default="toy")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--host", type=str, default=None)
    args = parser.parse_args()

    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_root = Path(__file__).resolve().parent.parent
        args.run_dir = repo_root / "runs" / f"enigma_s2_{timestamp}"
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "toy":
        run_toy_stage(args.run_dir)
    elif args.stage == "real":
        if not args.host:
            print("--host required")
            sys.exit(1)
        run_real_stage(args.run_dir, args.host)
    elif args.stage == "score":
        evals_dir = args.run_dir / "evaluations"
        if not evals_dir.exists():
            evals_dir = args.run_dir / "results"
        for f in sorted(evals_dir.glob("*_real.json")):
            data = json.loads(f.read_text())
            bpb = data.get("final_validation_bpb", "N/A")
            print(f"  {f.stem}: valid={data.get('valid')} bpb={bpb} mutation={data.get('code_mutation', 'none')}")


if __name__ == "__main__":
    main()
