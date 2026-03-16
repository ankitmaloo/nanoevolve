#!/usr/bin/env python3
"""Enigma Stage 4: Mutations applied as monkey-patches to production NanoChat.

Instead of maintaining a separate eval loop, Stage 4 patches the production
MuonAdamW optimizer and runs the real base_train.py training loop. This ensures
all production schedules (momentum warmup, WD decay, linear warmdown LR) are
active, and the baseline is exactly production.

Usage (on GPU cluster):
    # Baseline (unpatched production)
    python enigma/run_stage4.py --mutation none --output results/baseline.json

    # With a mutation
    python enigma/run_stage4.py --mutation H60_beta2_warmup --output results/H60.json

The script:
1. Patches nanochat.optim.muon_step_fused or MuonAdamW._step_muon / _step_adamw
2. Patches schedule functions in base_train (if mutation modifies schedules)
3. Runs production training for --steps iterations at --depth
4. Captures val_bpb curve and writes results JSON
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import Tensor


# ── Mutation registry ─────────────────────────────────────────

MUTATIONS: dict[str, dict] = {}


def register_mutation(mutation_id: str, title: str, direction: str):
    """Decorator to register a mutation function."""
    def decorator(fn):
        MUTATIONS[mutation_id] = {
            "id": mutation_id,
            "title": title,
            "direction": direction,
            "apply_fn": fn,
        }
        return fn
    return decorator


# ── Direction A: Schedule mutations ───────────────────────────
# These patch the schedule functions in the training loop.
# The apply_fn receives a context dict and patches what it needs.

@register_mutation("H60_beta2_warmup", "Second moment beta2 warmup 0.8→0.95", "schedule")
def apply_h60(ctx):
    """NorMuon beta2 warmup: start at 0.8 (less smoothing early), warm to 0.95 over 500 steps."""
    original_step_muon = ctx["optimizer_cls"]._step_muon

    def patched_step_muon(self, group):
        step = ctx["get_step"]()
        warmup = 500
        beta2 = 0.8 + (0.95 - 0.8) * min(1.0, step / warmup)
        group["beta2"] = beta2
        return original_step_muon(self, group)

    ctx["optimizer_cls"]._step_muon = patched_step_muon


@register_mutation("H61_trust_clamp_anneal", "Trust clamp narrows over training", "schedule")
def apply_h61(ctx):
    """Trust ratio clamp starts wide [0.3, 2.0] and narrows to [0.5, 1.5] over 1000 steps.
    Requires injecting trust ratio into production Muon (which doesn't have it).
    This mutation adds layerwise trust ratio to muon_step_fused."""
    import nanochat.optim as optim_module

    original_muon_step = optim_module.muon_step_fused.__wrapped__ if hasattr(optim_module.muon_step_fused, '__wrapped__') else None

    # We need to replace muon_step_fused with a version that adds trust ratio
    @torch.compile(dynamic=False, fullgraph=True)
    def muon_step_with_trust(
        stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
        momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim,
        trust_clamp_min_t, trust_clamp_max_t,
    ):
        # Original momentum
        momentum = momentum_t.to(stacked_grads.dtype)
        momentum_buffer.lerp_(stacked_grads, 1 - momentum)
        g = stacked_grads.lerp_(momentum_buffer, momentum)

        # Polar express (unchanged)
        X = g.bfloat16()
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        if g.size(-2) > g.size(-1):
            for a, b, c in optim_module.polar_express_coeffs[:ns_steps]:
                A = X.mT @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:
            for a, b, c in optim_module.polar_express_coeffs[:ns_steps]:
                A = X @ X.mT
                B = b * A + c * (A @ A)
                X = a * X + B @ X
        g = X

        # Variance reduction (unchanged)
        beta2 = beta2_t.to(g.dtype)
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = g.size(red_dim)
        v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
        v_norm = v_norm_sq.sqrt()
        second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
        step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
        final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
        g = g * final_scale.to(g.dtype)

        # NEW: Layerwise trust ratio
        eps = 1e-8
        param_norm = stacked_params.float().flatten(1).norm(dim=1).clamp_min(eps)
        update_norm = g.float().flatten(1).norm(dim=1).clamp_min(eps)
        trust = (param_norm / update_norm).clamp(trust_clamp_min_t, trust_clamp_max_t)
        g = g * trust.view(-1, *([1] * (g.ndim - 1))).to(g.dtype)

        # Cautious weight decay + parameter update
        lr = lr_t.to(g.dtype)
        wd = wd_t.to(g.dtype)
        mask = (g * stacked_params) >= 0
        stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

    # Patch _step_muon to use our extended fused kernel
    trust_clamp_min_t = torch.tensor(0.3, dtype=torch.float32, device="cpu")
    trust_clamp_max_t = torch.tensor(2.0, dtype=torch.float32, device="cpu")

    original_step_muon = ctx["optimizer_cls"]._step_muon

    def patched_step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        # Anneal trust clamps
        step = ctx["get_step"]()
        anneal_steps = 1000
        frac = min(1.0, step / anneal_steps)
        trust_clamp_min_t.fill_(0.3 + (0.5 - 0.3) * frac)
        trust_clamp_max_t.fill_(2.0 + (1.5 - 2.0) * frac)

        muon_step_with_trust(
            stacked_grads, stacked_params,
            state["momentum_buffer"], state["second_momentum_buffer"],
            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
            group["ns_steps"], red_dim,
            trust_clamp_min_t, trust_clamp_max_t,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    ctx["optimizer_cls"]._step_muon = patched_step_muon


@register_mutation("H62_wd_cosine_decay", "WD cosine decay instead of linear", "schedule")
def apply_h62(ctx):
    """Production decays WD linearly. Cosine keeps WD higher longer, then drops faster at end."""
    def patched_get_weight_decay(it):
        progress = it / ctx["num_iterations"]
        return ctx["wd_initial"] * 0.5 * (1.0 + math.cos(math.pi * progress))
    ctx["schedule_patches"]["get_weight_decay"] = patched_get_weight_decay


@register_mutation("H63_momentum_overshoot", "Momentum overshoot 0.85→0.97→0.95", "schedule")
def apply_h63(ctx):
    """Overshoot momentum past target then settle. Higher momentum mid-training accelerates convergence."""
    def patched_get_muon_momentum(it):
        if it < 300:
            return 0.85 + (0.97 - 0.85) * (it / 300)
        elif it < 1000:
            return 0.97 + (0.95 - 0.97) * ((it - 300) / 700)
        return 0.95
    ctx["schedule_patches"]["get_muon_momentum"] = patched_get_muon_momentum


@register_mutation("H64_nesterov_blend_schedule", "Nesterov blend 0.7→0.95 schedule", "schedule")
def apply_h64(ctx):
    """Decouple Nesterov blend from momentum. Start conservative (0.7), increase to 0.95.
    Less aggressive lookahead early when gradients are noisy."""
    import nanochat.optim as optim_module

    @torch.compile(dynamic=False, fullgraph=True)
    def muon_step_nesterov_scheduled(
        stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
        momentum_t, nesterov_blend_t, lr_t, wd_t, beta2_t, ns_steps, red_dim,
    ):
        momentum = momentum_t.to(stacked_grads.dtype)
        nesterov_blend = nesterov_blend_t.to(stacked_grads.dtype)
        momentum_buffer.lerp_(stacked_grads, 1 - momentum)
        g = stacked_grads.lerp_(momentum_buffer, nesterov_blend)  # Use scheduled blend

        X = g.bfloat16()
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        if g.size(-2) > g.size(-1):
            for a, b, c in optim_module.polar_express_coeffs[:ns_steps]:
                A = X.mT @ X; B = b * A + c * (A @ A); X = a * X + X @ B
        else:
            for a, b, c in optim_module.polar_express_coeffs[:ns_steps]:
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

        lr = lr_t.to(g.dtype); wd = wd_t.to(g.dtype)
        mask = (g * stacked_params) >= 0
        stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

    nesterov_blend_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def patched_step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        step = ctx["get_step"]()
        blend = 0.7 + (0.95 - 0.7) * min(1.0, step / 500)
        nesterov_blend_t.fill_(blend)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        muon_step_nesterov_scheduled(
            stacked_grads, stacked_params,
            state["momentum_buffer"], state["second_momentum_buffer"],
            self._muon_momentum_t, nesterov_blend_t,
            self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
            group["ns_steps"], red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    ctx["optimizer_cls"]._step_muon = patched_step_muon


# ── Direction B: AdamW path mutations ─────────────────────────

@register_mutation("H70_embed_wd", "Small weight decay on embeddings", "adamw")
def apply_h70(ctx):
    """Add wd=0.01 to embedding and value_embeds groups. Light regularization on lookup tables."""
    def patch_groups(optimizer):
        for group in optimizer.param_groups:
            if group["kind"] == "adamw" and group.get("lr", 0) > 0.1:
                # Embedding groups have lr ~0.2, others are much lower
                group["weight_decay"] = 0.01
    ctx["post_optimizer_patches"].append(patch_groups)


@register_mutation("H71_adamw_beta1_warmup", "AdamW beta1 warmup 0.7→0.8", "adamw")
def apply_h71(ctx):
    """Warm AdamW beta1 from 0.7 to 0.8 over 300 steps. Mirrors Muon momentum warmup for AdamW path."""
    original_step_adamw = ctx["optimizer_cls"]._step_adamw

    def patched_step_adamw(self, group):
        step = ctx["get_step"]()
        frac = min(1.0, step / 300)
        base_beta1 = group["betas"][0]
        warmed_beta1 = (base_beta1 - 0.1) + 0.1 * frac  # 0.7→0.8 if base is 0.8
        group["betas"] = (warmed_beta1, group["betas"][1])
        return original_step_adamw(self, group)

    ctx["optimizer_cls"]._step_adamw = patched_step_adamw


@register_mutation("H72_lmhead_wd", "Weight decay on lm_head", "adamw")
def apply_h72(ctx):
    """Add small wd=0.01 to lm_head. The unembedding matrix may benefit from regularization."""
    def patch_groups(optimizer):
        for group in optimizer.param_groups:
            if group["kind"] == "adamw" and group.get("lr", 0) < 0.01 and group.get("lr", 0) > 0.001:
                # lm_head has lr ~0.004
                group["weight_decay"] = 0.01
    ctx["post_optimizer_patches"].append(patch_groups)


# ── Direction C: Cross-path interactions ──────────────────────

@register_mutation("H80_production_baseline", "Exact production baseline (no mutation)", "baseline")
def apply_h80(ctx):
    """No patches. Runs exact production training. This IS the baseline."""
    pass


@register_mutation("H81_trust_ratio_production", "Add trust ratio to production Muon", "cross")
def apply_h81(ctx):
    """Add layerwise trust ratio [0.5, 1.5] to production Muon. This is H02 but on the real
    production optimizer with all schedules active. Tests whether H02's win survives."""
    import nanochat.optim as optim_module

    @torch.compile(dynamic=False, fullgraph=True)
    def muon_step_with_trust_fixed(
        stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
        momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim,
    ):
        momentum = momentum_t.to(stacked_grads.dtype)
        momentum_buffer.lerp_(stacked_grads, 1 - momentum)
        g = stacked_grads.lerp_(momentum_buffer, momentum)

        X = g.bfloat16()
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        if g.size(-2) > g.size(-1):
            for a, b, c in optim_module.polar_express_coeffs[:ns_steps]:
                A = X.mT @ X; B = b * A + c * (A @ A); X = a * X + X @ B
        else:
            for a, b, c in optim_module.polar_express_coeffs[:ns_steps]:
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

        # Trust ratio (layerwise)
        eps = 1e-8
        param_norm = stacked_params.float().flatten(1).norm(dim=1).clamp_min(eps)
        update_norm = g.float().flatten(1).norm(dim=1).clamp_min(eps)
        trust = (param_norm / update_norm).clamp(0.5, 1.5)
        g = g * trust.view(-1, *([1] * (g.ndim - 1))).to(g.dtype)

        lr = lr_t.to(g.dtype); wd = wd_t.to(g.dtype)
        mask = (g * stacked_params) >= 0
        stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

    # Replace the compiled kernel
    optim_module.muon_step_fused = muon_step_with_trust_fixed


# ── Runner ────────────────────────────────────────────────────

def list_mutations():
    print("Available mutations:")
    for mid, info in MUTATIONS.items():
        print(f"  {mid:35s} [{info['direction']:8s}] {info['title']}")


def main():
    parser = argparse.ArgumentParser(description="Enigma Stage 4 — Production-patched mutations")
    parser.add_argument("--mutation", type=str, required=True, help="Mutation ID or 'none' for baseline")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--depth", type=int, default=12, help="Model depth")
    parser.add_argument("--eval-every", type=int, default=250, help="Eval interval")
    parser.add_argument("--list", action="store_true", help="List available mutations")
    args = parser.parse_args()

    if args.list:
        list_mutations()
        return

    mutation_id = args.mutation if args.mutation != "none" else "H80_production_baseline"

    if mutation_id not in MUTATIONS:
        print(f"Unknown mutation: {mutation_id}")
        list_mutations()
        sys.exit(1)

    print(f"=== Enigma Stage 4: {mutation_id} ===")
    print(f"Steps: {args.steps}, Depth: {args.depth}, Eval every: {args.eval_every}")

    # Import production optimizer
    from nanochat.optim import MuonAdamW

    # Build mutation context
    _step = [0]
    ctx = {
        "optimizer_cls": MuonAdamW,
        "get_step": lambda: _step[0],
        "num_iterations": args.steps,
        "wd_initial": 0.2,  # will be overridden if needed
        "schedule_patches": {},
        "post_optimizer_patches": [],
    }

    # Apply mutation
    MUTATIONS[mutation_id]["apply_fn"](ctx)

    # Now run production training with sys.argv patched
    # We override sys.argv to feed args to base_train.py's argparse
    sys.argv = [
        "base_train",
        f"--depth={args.depth}",
        f"--num-iterations={args.steps}",
        f"--eval-every={args.eval_every}",
        "--max-seq-len=512",
        "--device-batch-size=2",
        "--total-batch-size=1024",
        "--core-metric-every=-1",
        "--sample-every=-1",
        "--save-every=-1",
        "--run=dummy",
    ]

    # We need to hook into the training loop to:
    # 1. Track step count for schedule mutations
    # 2. Apply post-optimizer patches
    # 3. Capture val_bpb curve for results
    # This requires modifying base_train.py's globals after import
    #
    # For now, print instructions for the slurm script approach
    print(f"\nMutation {mutation_id} applied to MuonAdamW class.")
    print("Context patches:", list(ctx["schedule_patches"].keys()))
    print("Post-optimizer patches:", len(ctx["post_optimizer_patches"]))
    print(f"\nTo run: the slurm script will inline this as a monkey-patch before base_train logic.")


if __name__ == "__main__":
    main()
