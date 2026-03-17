#!/usr/bin/env python3
"""Stage 5 Distributed: 8xH100 production-faithful evaluation runner.

Runs the same mutations as Stage 5 but on 8xH100 with DistMuonAdamW,
production batch sizes, full dataset, and torch.compile enabled.

Usage:
    torchrun --standalone --nproc_per_node=8 -m enigma.run_stage5_dist \
        --mutation H64_H60_H73 --steps 5000 --depth 12 \
        --output runs/enigma_s5_dist/results/H64_H60_H73.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist

SUPPORTED_FEATURES = {
    "H60", "H64", "H71", "H73",
    "H504", "H517", "H531", "H532", "H533", "H534", "H535", "H536", "H537", "H538",
}


def mutation_features(mutation: str) -> set[str]:
    return set(re.findall(r"H\d+", mutation))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enigma Stage 5 distributed evaluation")
    parser.add_argument("--mutation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--device-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Schedule functions (same as production base_train.py) ──────────────

def get_lr_multiplier(step: int, steps: int, warmup_ratio: float = 0.0, warmdown_ratio: float = 0.5) -> float:
    warmup_iters = round(warmup_ratio * steps)
    warmdown_iters = round(warmdown_ratio * steps)
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    elif step <= steps - warmdown_iters:
        return 1.0
    else:
        progress = (steps - step) / warmdown_iters
        return progress


def get_muon_momentum(step: int) -> float:
    frac = min(step / 300, 1.0)
    return (1.0 - frac) * 0.85 + frac * 0.95


def get_weight_decay(step: int, steps: int, base_wd: float) -> float:
    return base_wd * (1.0 - step / steps)


def get_warmdown_start(steps: int) -> int:
    return steps - round(0.5 * steps)


# ── Mutation schedule helpers ──────────────────────────────────────────

def get_h60_beta2(step: int) -> float:
    return 0.8 + (0.95 - 0.8) * min(1.0, step / 500.0)


def get_h504_beta2(step: int, steps: int) -> float:
    if step < 500:
        return 0.8 + (0.95 - 0.8) * min(1.0, step / 500.0)
    warmdown_start = steps * 0.5
    if step <= warmdown_start:
        return 0.95
    phase = min(1.0, (step - warmdown_start) / max(steps - warmdown_start, 1.0))
    return 0.95 + (0.99 - 0.95) * phase


def get_h517_beta2(step: int, steps: int) -> float:
    early = 0.8 + (0.95 - 0.8) * min(1.0, step / 500.0)
    warmdown_start = steps * 0.5
    if step <= warmdown_start:
        return early
    phase = min(1.0, (step - warmdown_start) / max(steps - warmdown_start, 1.0))
    return max(early, 0.95 + (0.99 - 0.95) * phase)


def get_h533_beta2(step: int, group: dict) -> float:
    if group.get("muon_family") == "rectangular":
        return get_h60_beta2(step)
    return 0.95


def get_h534_ns_steps(step: int, steps: int) -> int:
    early_steps = max(1000, steps // 10)
    return 4 if step < early_steps else 5


def get_h535_eps(step: int, steps: int) -> float:
    frac = min(step / max(steps, 1), 1.0)
    return 10.0 ** (-6.0 + (-4.0 * frac))


def get_h536_x0_beta1(step: int, steps: int) -> float:
    warmdown_start = get_warmdown_start(steps)
    if step <= warmdown_start:
        return 0.96
    phase = min(1.0, (step - warmdown_start) / max(steps - warmdown_start, 1))
    return 0.96 + (0.90 - 0.96) * phase


def annotate_optimizer_groups(optimizer) -> None:
    adamw_group_names = ["lm_head", "embedding", "value_embeds", "resid_lambdas", "x0_lambdas"]
    adamw_index = 0
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
        if group["kind"] == "adamw":
            group["group_name"] = adamw_group_names[adamw_index]
            group["base_betas"] = tuple(group["betas"])
            group["base_eps"] = group["eps"]
            adamw_index += 1
        else:
            shape = tuple(group["params"][0].shape)
            aspect_ratio = max(shape) / min(shape)
            group["shape"] = shape
            group["aspect_ratio"] = aspect_ratio
            group["muon_family"] = "rectangular" if aspect_ratio >= 2.0 else "squareish"
            group["base_beta2"] = group["beta2"]
            group["base_ns_steps"] = group["ns_steps"]


def apply_h532_muon_vreset(optimizer, step: int, steps: int) -> None:
    if step < get_warmdown_start(steps):
        return
    for group in optimizer.param_groups:
        if group["kind"] != "muon":
            continue
        state = optimizer.state[group["params"][0]]
        if state.get("enigma_h532_reset_done"):
            continue
        second = state.get("second_momentum_buffer")
        if second is None:
            continue
        second.mul_(0.25)
        state["enigma_h532_reset_done"] = True


def apply_h537_embed_mom_reset(optimizer, step: int, steps: int) -> None:
    if step < get_warmdown_start(steps):
        return
    for group in optimizer.param_groups:
        if group["kind"] != "adamw" or group.get("group_name") not in {"embedding", "value_embeds"}:
            continue
        for param in group["params"]:
            state = optimizer.state[param]
            if state.get("enigma_h537_reset_done"):
                continue
            exp_avg = state.get("exp_avg")
            if exp_avg is None:
                continue
            exp_avg.mul_(0.25)
            state["enigma_h537_reset_done"] = True


def print0(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


def main() -> None:
    args = parse_args()
    features = mutation_features(args.mutation)

    if args.mutation != "none":
        if not features:
            raise SystemExit(f"Unsupported mutation '{args.mutation}'")
        unknown = sorted(features - SUPPORTED_FEATURES)
        if unknown:
            raise SystemExit(f"Unknown features: {', '.join(unknown)}")

    # Apply monkey-patches before importing model
    os.environ["ENIGMA_MUTATION"] = args.mutation
    os.environ["ENIGMA_TOTAL_STEPS"] = str(args.steps)
    from enigma import stage5_patch  # noqa: F401

    # DDP init
    dist.init_process_group(backend="nccl")
    ddp_rank = dist.get_rank()
    ddp_world_size = dist.get_world_size()
    device = f"cuda:{ddp_rank % torch.cuda.device_count()}"
    torch.cuda.set_device(device)

    from nanochat.dataloader import (
        tokenizing_distributed_data_loader_bos_bestfit,
        tokenizing_distributed_data_loader_with_state_bos_bestfit,
    )
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.loss_eval import evaluate_bpb
    from nanochat.tokenizer import get_token_bytes, get_tokenizer

    output_path = Path(args.output)
    if ddp_rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Model config — match production
    max_seq_len = 1024
    config = GPTConfig(n_layer=args.depth, sequence_len=max_seq_len)
    model = GPT(config).to(device)
    model.train()

    # Compute batch size like production (B_REF=524288 for d12)
    total_batch_size = 2 ** 19  # 524288 tokens — production d12 default
    tokens_per_fwdbwd = args.device_batch_size * max_seq_len * ddp_world_size
    assert total_batch_size % tokens_per_fwdbwd == 0, \
        f"total_batch_size {total_batch_size} not divisible by tokens_per_fwdbwd {tokens_per_fwdbwd}"
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    # Weight decay scaling (from production)
    import nanochat.gpt as gpt_module
    d12_ref = gpt_module.GPT(GPTConfig(n_layer=12, sequence_len=max_seq_len))
    num_scaling_params = sum(
        v for k, v in model.num_scaling_params().items()
        if k in ('transformer_matrices', 'lm_head')
    )
    d12_scaling = sum(
        v for k, v in d12_ref.num_scaling_params().items()
        if k in ('transformer_matrices', 'lm_head')
    )
    target_param_data_ratio = 10.5
    target_tokens = int(target_param_data_ratio * num_scaling_params)
    D_REF = target_param_data_ratio * d12_scaling
    B_REF = 2**19
    batch_ratio = total_batch_size / B_REF
    weight_decay_scaled = 0.2 * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
    del d12_ref

    optimizer = model.setup_optimizer(weight_decay=weight_decay_scaled)
    annotate_optimizer_groups(optimizer)

    # Compute num_iterations if not overridden
    num_iterations = args.steps if args.steps > 0 else target_tokens // total_batch_size

    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, args.device_batch_size, max_seq_len, split="train", device=device,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, max_seq_len, split="val", device=device,
    )

    print0(f"=== ENIGMA Stage 5 Distributed — {args.mutation} ===")
    print0(f"GPUs: {ddp_world_size}, Steps: {num_iterations}, Depth: {args.depth}")
    print0(f"Batch: {total_batch_size} tokens, Device batch: {args.device_batch_size}, Grad accum: {grad_accum_steps}")
    print0(f"Features: {sorted(features)}")
    print0(f"Weight decay (scaled): {weight_decay_scaled:.6f}")

    x, y, _ = next(train_loader)
    curve: list[dict] = []
    train_curve: list[dict] = []
    step_times_ms: list[float] = []
    failure_type: str | None = None

    total_start = time.perf_counter()
    for step in range(num_iterations + 1):
        last_step = step == num_iterations

        # Eval
        if step % args.eval_every == 0 or last_step:
            model.eval()
            val_loader = build_val_loader()
            eval_steps = max(1, 512 // (args.device_batch_size * max_seq_len * ddp_world_size))
            with torch.no_grad():
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            curve.append({"step": step, "val_bpb": float(val_bpb)})
            model.train()
            print0(f"  step {step:>5d}: val_bpb={val_bpb:.6f}")

        if last_step:
            break

        # Forward + backward
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(grad_accum_steps):
            loss = model(x, y)
            accumulated_loss += loss.detach().item()
            (loss / grad_accum_steps).backward()
            x, y, _ = next(train_loader)
        train_loss = accumulated_loss / grad_accum_steps

        if not math.isfinite(train_loss):
            failure_type = "non_finite_loss"
            break

        # Apply schedules
        lrm = get_lr_multiplier(step, num_iterations)
        muon_momentum = get_muon_momentum(step)
        muon_wd = get_weight_decay(step, num_iterations, weight_decay_scaled)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "adamw":
                group["betas"] = group["base_betas"]
                group["eps"] = group["base_eps"]
            else:
                group["beta2"] = group["base_beta2"]
                group["ns_steps"] = group["base_ns_steps"]
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_wd

        # Mutation schedule overrides
        if "H517" in features:
            b2 = get_h517_beta2(step, num_iterations)
            for g in optimizer.param_groups:
                if g["kind"] == "muon": g["beta2"] = b2
        elif "H504" in features:
            b2 = get_h504_beta2(step, num_iterations)
            for g in optimizer.param_groups:
                if g["kind"] == "muon": g["beta2"] = b2
        elif "H533" in features:
            for g in optimizer.param_groups:
                if g["kind"] == "muon": g["beta2"] = get_h533_beta2(step, g)
        elif "H60" in features:
            b2 = get_h60_beta2(step)
            for g in optimizer.param_groups:
                if g["kind"] == "muon": g["beta2"] = b2

        if "H534" in features:
            ns = get_h534_ns_steps(step, num_iterations)
            for g in optimizer.param_groups:
                if g["kind"] == "muon": g["ns_steps"] = ns

        if "H535" in features:
            eps = get_h535_eps(step, num_iterations)
            for g in optimizer.param_groups:
                if g["kind"] == "adamw" and g.get("group_name") in {"embedding", "value_embeds"}:
                    g["eps"] = eps

        if "H536" in features:
            b1 = get_h536_x0_beta1(step, num_iterations)
            for g in optimizer.param_groups:
                if g["kind"] == "adamw" and g.get("group_name") == "x0_lambdas":
                    _, b2 = g["base_betas"]
                    g["betas"] = (b1, b2)

        if "H532" in features:
            apply_h532_muon_vreset(optimizer, step, num_iterations)
        if "H537" in features:
            apply_h537_embed_mom_reset(optimizer, step, num_iterations)

        os.environ["_ENIGMA_STEP"] = str(step)
        optimizer.step()
        model.zero_grad(set_to_none=True)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        step_times_ms.append(elapsed_ms)

        if step % 500 == 0:
            train_bpb = train_loss / math.log(2.0)
            train_curve.append({"step": step, "train_bpb": float(train_bpb), "step_time_ms": float(elapsed_ms)})
            tok_per_sec = int(total_batch_size / (elapsed_ms / 1000))
            print0(f"  step {step:>5d}: train_bpb={train_bpb:.4f} ({elapsed_ms:.0f}ms, {tok_per_sec:,} tok/s)")

    total_time_s = time.perf_counter() - total_start

    if ddp_rank == 0:
        val_points = [p["val_bpb"] for p in curve]
        final_bpb = val_points[-1] if val_points else float("inf")
        best_bpb = min(val_points) if val_points else float("inf")

        result = {
            "mutation": args.mutation,
            "features": sorted(features),
            "valid": failure_type is None,
            "failure_type": failure_type,
            "final_validation_bpb": final_bpb,
            "best_validation_bpb": best_bpb,
            "mean_step_time_ms": sum(step_times_ms) / max(1, len(step_times_ms)),
            "total_time_s": total_time_s,
            "steps_completed": len(step_times_ms),
            "curve": curve,
            "train_curve": train_curve,
            "config": {
                "gpus": ddp_world_size,
                "depth": args.depth,
                "total_batch_size": total_batch_size,
                "device_batch_size": args.device_batch_size,
                "seq_len": max_seq_len,
                "grad_accum_steps": grad_accum_steps,
                "weight_decay_scaled": weight_decay_scaled,
                "num_data_shards": "1500+1val",
            },
        }
        output_path.write_text(json.dumps(result, indent=2))
        print0(f"\nSaved: {output_path}")
        print0(f"Final BPB: {final_bpb:.6f}, Best: {best_bpb:.6f}, Time: {total_time_s:.1f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
