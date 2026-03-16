#!/usr/bin/env python3
"""Enigma Stage 4: Monkey-patch production NanoChat optimizer before training.

This script is imported before base_train.py runs. It reads the ENIGMA_MUTATION
environment variable and patches nanochat.optim accordingly.

Usage:
    ENIGMA_MUTATION=H70_embed_wd python -c "
        import enigma.stage4_patch  # applies patch
        exec(open('nanochat/scripts/base_train.py').read())
    "

Or more robustly, via the wrapper in slurm_s4.sh.
"""
from __future__ import annotations

import math
import os

import torch
from torch import Tensor

mutation = os.environ.get("ENIGMA_MUTATION", "none")


def _get_step_from_context():
    """Extract current step from the AdamW state (first param's step counter)."""
    return int(os.environ.get("_ENIGMA_STEP", "0"))


if mutation != "none":
    import nanochat.optim as optim_module
    from nanochat.optim import polar_express_coeffs

    # Keep references to originals
    _original_muon_step_fused = optim_module.muon_step_fused
    _original_MuonAdamW = optim_module.MuonAdamW

    # ── H81: Add trust ratio to production Muon ──────────────
    if mutation == "H81_trust_ratio":
        @torch.compile(dynamic=False, fullgraph=True)
        def muon_step_with_trust(
            stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
            momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim,
        ):
            momentum = momentum_t.to(stacked_grads.dtype)
            momentum_buffer.lerp_(stacked_grads, 1 - momentum)
            g = stacked_grads.lerp_(momentum_buffer, momentum)

            X = g.bfloat16()
            X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
            if g.size(-2) > g.size(-1):
                for a, b, c in polar_express_coeffs[:ns_steps]:
                    A = X.mT @ X; B = b * A + c * (A @ A); X = a * X + X @ B
            else:
                for a, b, c in polar_express_coeffs[:ns_steps]:
                    A = X @ X.mT; B = b * A + c * (A @ A); X = a * X + B @ X
            g = X

            beta2 = beta2_t.to(g.dtype)
            v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
            red_dim_size = g.size(red_dim)
            v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
            second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
            step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
            scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
            v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
            final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
            g = g * final_scale.to(g.dtype)

            # Trust ratio (layerwise) [0.5, 1.5]
            param_norm = stacked_params.float().flatten(1).norm(dim=1).clamp_min(1e-8)
            update_norm = g.float().flatten(1).norm(dim=1).clamp_min(1e-8)
            trust = (param_norm / update_norm).clamp(0.5, 1.5)
            g = g * trust.view(-1, *([1] * (g.ndim - 1))).to(g.dtype)

            lr = lr_t.to(g.dtype); wd = wd_t.to(g.dtype)
            mask = (g * stacked_params) >= 0
            stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

        optim_module.muon_step_fused = muon_step_with_trust
        print(f"[ENIGMA] Patched muon_step_fused with trust ratio [0.5, 1.5]")

    # ── H70: Embedding weight decay ──────────────────────────
    elif mutation == "H70_embed_wd":
        # Patch applied post-optimizer-creation in the slurm script
        # (sets weight_decay=0.01 on embedding groups)
        print(f"[ENIGMA] H70: Will set embedding weight_decay=0.01 post-optimizer")

    # ── H71: AdamW beta1 warmup ──────────────────────────────
    elif mutation == "H71_beta1_warmup":
        _orig_adamw_fused = optim_module.adamw_step_fused

        @torch.compile(dynamic=False, fullgraph=True)
        def adamw_step_warmup(p, grad, exp_avg, exp_avg_sq, step_t, lr_t,
                               beta1_t, beta2_t, eps_t, wd_t):
            # Warm beta1 from (beta1-0.1) to beta1 over 300 steps
            warmup = 300.0
            frac = (step_t / warmup).clamp(max=1.0)
            effective_beta1 = (beta1_t - 0.1) + 0.1 * frac
            # Standard AdamW with warmed beta1
            p.mul_(1 - lr_t * wd_t)
            exp_avg.lerp_(grad, 1 - effective_beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
            bias1 = 1 - effective_beta1 ** step_t
            bias2 = 1 - beta2_t ** step_t
            denom = (exp_avg_sq / bias2).sqrt() + eps_t
            step_size = lr_t / bias1
            p.add_(exp_avg / denom, alpha=-step_size)

        optim_module.adamw_step_fused = adamw_step_warmup
        print(f"[ENIGMA] Patched adamw_step_fused with beta1 warmup 0.7→0.8/300 steps")

    # ── H73: Epsilon schedule ────────────────────────────────
    elif mutation == "H73_eps_schedule":
        _orig_adamw_fused = optim_module.adamw_step_fused

        @torch.compile(dynamic=False, fullgraph=True)
        def adamw_step_eps_sched(p, grad, exp_avg, exp_avg_sq, step_t, lr_t,
                                   beta1_t, beta2_t, eps_t, wd_t):
            # Log-linear eps from 1e-6 to 1e-10 (using step_t as proxy for progress)
            # Since we don't have total_steps in the fused kernel, we use step count
            # 5000 steps → at step 5000, eps ≈ 1e-10
            log_ratio = -4.0  # log10(1e-10 / 1e-6)
            frac = (step_t / 5000.0).clamp(max=1.0)
            log_eps = -6.0 + log_ratio * frac  # -6 to -10
            eps = 10.0 ** log_eps
            p.mul_(1 - lr_t * wd_t)
            exp_avg.lerp_(grad, 1 - beta1_t)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
            bias1 = 1 - beta1_t ** step_t
            bias2 = 1 - beta2_t ** step_t
            denom = (exp_avg_sq / bias2).sqrt() + eps
            step_size = lr_t / bias1
            p.add_(exp_avg / denom, alpha=-step_size)

        optim_module.adamw_step_fused = adamw_step_eps_sched
        print(f"[ENIGMA] Patched adamw_step_fused with eps schedule 1e-6→1e-10")

    # ── H78: Scalar LR warmup ───────────────────────────────
    elif mutation == "H78_scalar_warmup":
        # Patch applied in training loop (scales scalar group LRs during warmup)
        print(f"[ENIGMA] H78: Will apply scalar LR warmup in training loop")

    # ── H60: Beta2 warmup for NorMuon ───────────────────────
    elif mutation == "H60_beta2_warmup":
        # Patch applied in training loop (modifies group['beta2'] per step)
        print(f"[ENIGMA] H60: Will warm Muon beta2 from 0.8→0.95 over 500 steps")

    # ── H63: Momentum overshoot ──────────────────────────────
    elif mutation == "H63_momentum_overshoot":
        # Patch applied by replacing get_muon_momentum in training loop
        print(f"[ENIGMA] H63: Momentum overshoot 0.85→0.97→0.95")

    else:
        print(f"[ENIGMA] WARNING: Unknown mutation '{mutation}', running unpatched")
