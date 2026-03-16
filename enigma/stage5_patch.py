#!/usr/bin/env python3
"""Stage 4 extension / Stage 5 optimizer monkey-patches for production runs.

This module is imported before the training loop starts. It reads
``ENIGMA_MUTATION`` and patches the production optimizer when a run includes
any of the Stage 4 winner features:

- H64: decoupled Nesterov blend schedule on Muon
- H531: raw-gradient cautious weight decay mask on Muon
- H71: beta1 warmup 0.7 -> 0.8 over 300 steps
- H73: epsilon schedule 1e-6 -> 1e-10 over total training steps
- H538: seed AdamW second moment from the first gradient for embedding-like groups
- H517: shared warmdown phase variable for AdamW epsilon

Schedule-only mutations (H60/H504/H532/H533/H534/H535/H536/H537) are applied in the training loop itself.
"""
from __future__ import annotations

import os
import re

import torch

mutation = os.environ.get("ENIGMA_MUTATION", "none")
features = set(re.findall(r"H\d+", mutation))
total_steps = float(os.environ.get("ENIGMA_TOTAL_STEPS", "5000"))


def _patch_muon_h64(optim_module) -> None:
    polar_express_coeffs = optim_module.polar_express_coeffs

    @torch.compile(dynamic=False, fullgraph=True)
    def muon_step_nesterov_scheduled(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_momentum_buffer,
        momentum_t,
        nesterov_blend_t,
        lr_t,
        wd_t,
        beta2_t,
        ns_steps,
        red_dim,
    ):
        momentum = momentum_t.to(stacked_grads.dtype)
        nesterov_blend = nesterov_blend_t.to(stacked_grads.dtype)
        momentum_buffer.lerp_(stacked_grads, 1 - momentum)
        g = stacked_grads.lerp_(momentum_buffer, nesterov_blend)

        X = g.bfloat16()
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        if g.size(-2) > g.size(-1):
            for a, b, c in polar_express_coeffs[:ns_steps]:
                A = X.mT @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:
            for a, b, c in polar_express_coeffs[:ns_steps]:
                A = X @ X.mT
                B = b * A + c * (A @ A)
                X = a * X + B @ X
        g = X

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

        lr = lr_t.to(g.dtype)
        wd = wd_t.to(g.dtype)
        mask = (g * stacked_params) >= 0
        stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

    nesterov_blend_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def patched_step_muon(self, group):
        params = group["params"]
        if not params:
            return

        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([param.grad for param in params])
        stacked_params = torch.stack(params)

        step = int(os.environ.get("_ENIGMA_STEP", "0"))
        blend = 0.7 + (0.95 - 0.7) * min(1.0, step / 500.0)
        nesterov_blend_t.fill_(blend)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        muon_step_nesterov_scheduled(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            nesterov_blend_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    optim_module.MuonAdamW._step_muon = patched_step_muon
    print("[ENIGMA] Patched MuonAdamW._step_muon with faithful H64 Nesterov blend schedule")


def _patch_muon_raw_grad_wd(optim_module) -> None:
    polar_express_coeffs = optim_module.polar_express_coeffs

    @torch.compile(dynamic=False, fullgraph=True)
    def muon_step_raw_mask(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_momentum_buffer,
        momentum_t,
        lr_t,
        wd_t,
        beta2_t,
        ns_steps,
        red_dim,
    ):
        momentum = momentum_t.to(stacked_grads.dtype)
        momentum_buffer.lerp_(stacked_grads, 1 - momentum)
        g = stacked_grads.lerp_(momentum_buffer, momentum)

        raw_mask_source = stacked_grads

        X = g.bfloat16()
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        if g.size(-2) > g.size(-1):
            for a, b, c in polar_express_coeffs[:ns_steps]:
                A = X.mT @ X
                B = b * A + c * (A @ A)
                X = a * X + X @ B
        else:
            for a, b, c in polar_express_coeffs[:ns_steps]:
                A = X @ X.mT
                B = b * A + c * (A @ A)
                X = a * X + B @ X
        g = X

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

        lr = lr_t.to(g.dtype)
        wd = wd_t.to(g.dtype)
        mask = (raw_mask_source * stacked_params) >= 0
        stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

    def patched_step_muon(self, group):
        params = group["params"]
        if not params:
            return

        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_grads = torch.stack([param.grad for param in params])
        stacked_params = torch.stack(params)

        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        muon_step_raw_mask(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    optim_module.MuonAdamW._step_muon = patched_step_muon
    print("[ENIGMA] Patched MuonAdamW._step_muon with H531 raw-gradient cautious WD mask")


def _patch_adamw_beta1_warmup(optim_module) -> None:
    @torch.compile(dynamic=False, fullgraph=True)
    def adamw_step_beta1_warmup(
        p,
        grad,
        exp_avg,
        exp_avg_sq,
        step_t,
        lr_t,
        beta1_t,
        beta2_t,
        eps_t,
        wd_t,
    ):
        frac = (step_t / 300.0).clamp(max=1.0)
        effective_beta1 = (beta1_t - 0.1) + 0.1 * frac

        p.mul_(1 - lr_t * wd_t)
        exp_avg.lerp_(grad, 1 - effective_beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
        bias1 = 1 - effective_beta1 ** step_t
        bias2 = 1 - beta2_t ** step_t
        denom = (exp_avg_sq / bias2).sqrt() + eps_t
        step_size = lr_t / bias1
        p.add_(exp_avg / denom, alpha=-step_size)

    optim_module.adamw_step_fused = adamw_step_beta1_warmup
    print("[ENIGMA] Patched adamw_step_fused with H71 beta1 warmup")


def _patch_adamw_eps_schedule(optim_module) -> None:
    @torch.compile(dynamic=False, fullgraph=True)
    def adamw_step_eps_schedule(
        p,
        grad,
        exp_avg,
        exp_avg_sq,
        step_t,
        lr_t,
        beta1_t,
        beta2_t,
        eps_t,
        wd_t,
    ):
        frac = (step_t / total_steps).clamp(max=1.0)
        log_eps = -6.0 + (-4.0 * frac)
        eps = 10.0 ** log_eps

        p.mul_(1 - lr_t * wd_t)
        exp_avg.lerp_(grad, 1 - beta1_t)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
        bias1 = 1 - beta1_t ** step_t
        bias2 = 1 - beta2_t ** step_t
        denom = (exp_avg_sq / bias2).sqrt() + eps
        step_size = lr_t / bias1
        p.add_(exp_avg / denom, alpha=-step_size)

    optim_module.adamw_step_fused = adamw_step_eps_schedule
    print("[ENIGMA] Patched adamw_step_fused with H73 eps schedule")


def _patch_adamw_beta1_and_eps(optim_module) -> None:
    @torch.compile(dynamic=False, fullgraph=True)
    def adamw_step_combined(
        p,
        grad,
        exp_avg,
        exp_avg_sq,
        step_t,
        lr_t,
        beta1_t,
        beta2_t,
        eps_t,
        wd_t,
    ):
        frac_beta1 = (step_t / 300.0).clamp(max=1.0)
        effective_beta1 = (beta1_t - 0.1) + 0.1 * frac_beta1

        frac_eps = (step_t / total_steps).clamp(max=1.0)
        log_eps = -6.0 + (-4.0 * frac_eps)
        eps = 10.0 ** log_eps

        p.mul_(1 - lr_t * wd_t)
        exp_avg.lerp_(grad, 1 - effective_beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
        bias1 = 1 - effective_beta1 ** step_t
        bias2 = 1 - beta2_t ** step_t
        denom = (exp_avg_sq / bias2).sqrt() + eps
        step_size = lr_t / bias1
        p.add_(exp_avg / denom, alpha=-step_size)

    optim_module.adamw_step_fused = adamw_step_combined
    print("[ENIGMA] Patched adamw_step_fused with combined H71+H73")


def _patch_adamw_shared_phase(optim_module) -> None:
    warmdown_start = total_steps * 0.5
    warmdown_span = max(total_steps - warmdown_start, 1.0)

    @torch.compile(dynamic=False, fullgraph=True)
    def adamw_step_shared_phase(
        p,
        grad,
        exp_avg,
        exp_avg_sq,
        step_t,
        lr_t,
        beta1_t,
        beta2_t,
        eps_t,
        wd_t,
    ):
        phase = ((step_t - warmdown_start) / warmdown_span).clamp(min=0.0, max=1.0)
        log_eps = -6.0 + (-4.0 * phase)
        eps = 10.0 ** log_eps

        p.mul_(1 - lr_t * wd_t)
        exp_avg.lerp_(grad, 1 - beta1_t)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
        bias1 = 1 - beta1_t ** step_t
        bias2 = 1 - beta2_t ** step_t
        denom = (exp_avg_sq / bias2).sqrt() + eps
        step_size = lr_t / bias1
        p.add_(exp_avg / denom, alpha=-step_size)

    optim_module.adamw_step_fused = adamw_step_shared_phase
    print("[ENIGMA] Patched adamw_step_fused with H517 shared warmdown-phase eps schedule")


def _patch_adamw_seed_first_grad(optim_module) -> None:
    original_step_adamw = optim_module.MuonAdamW._step_adamw

    def patched_step_adamw(self, group):
        if group.get("group_name") not in {"embedding", "value_embeds"}:
            return original_step_adamw(self, group)

        params = group["params"]
        for p in params:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = p.grad.square().detach().clone()

        return original_step_adamw(self, group)

    optim_module.MuonAdamW._step_adamw = patched_step_adamw
    print("[ENIGMA] Patched MuonAdamW._step_adamw with H538 seeded first-gradient second moment")


if features:
    import nanochat.optim as optim_module

    has_h64 = "H64" in features
    has_h71 = "H71" in features
    has_h73 = "H73" in features
    has_h517 = "H517" in features
    has_h531 = "H531" in features
    has_h538 = "H538" in features

    if has_h531:
        _patch_muon_raw_grad_wd(optim_module)
    elif has_h64:
        _patch_muon_h64(optim_module)

    if has_h538:
        _patch_adamw_seed_first_grad(optim_module)

    if has_h517:
        _patch_adamw_shared_phase(optim_module)
    elif has_h71 and has_h73:
        _patch_adamw_beta1_and_eps(optim_module)
    elif has_h71:
        _patch_adamw_beta1_warmup(optim_module)
    elif has_h73:
        _patch_adamw_eps_schedule(optim_module)
