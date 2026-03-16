#!/usr/bin/env python3
"""Stage 5 production evaluation runner for single and composite optimizer mutations."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch

SUPPORTED_FEATURES = {
    "H60",
    "H64",
    "H71",
    "H73",
    "H504",
    "H517",
    "H531",
    "H532",
    "H533",
    "H534",
    "H535",
    "H536",
    "H537",
    "H538",
}


def mutation_features(mutation: str) -> set[str]:
    return set(re.findall(r"H\d+", mutation))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enigma Stage 5 production testing on the real NanoChat optimizer."
    )
    parser.add_argument("--mutation", required=True, help="Compound mutation id or 'none'")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--depth", type=int, default=12, help="Model depth")
    parser.add_argument("--eval-every", type=int, default=250, help="Eval interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def get_lr_multiplier(step: int, steps: int) -> float:
    warmdown_iters = round(0.5 * steps)
    if step <= steps - warmdown_iters:
        return 1.0
    return (steps - step) / warmdown_iters


def get_muon_momentum(step: int) -> float:
    frac = min(step / 300, 1.0)
    return (1.0 - frac) * 0.85 + frac * 0.95


def get_weight_decay(step: int, steps: int) -> float:
    return 0.2 * (1.0 - step / steps)


def get_warmdown_start(steps: int) -> int:
    return steps - round(0.5 * steps)


def get_h60_beta2(step: int) -> float:
    return 0.8 + (0.95 - 0.8) * min(1.0, step / 500.0)


def log(message: str) -> None:
    print(message, flush=True)


def get_h504_beta2(step: int, steps: int) -> float:
    """Two-phase Muon beta2 schedule: H60 early, then stabilize further in warmdown."""
    if step < 500:
        return 0.8 + (0.95 - 0.8) * min(1.0, step / 500.0)

    warmdown_start = steps * 0.5
    if step <= warmdown_start:
        return 0.95

    phase = min(1.0, (step - warmdown_start) / max(steps - warmdown_start, 1.0))
    return 0.95 + (0.99 - 0.95) * phase


def get_h517_beta2(step: int, steps: int) -> float:
    """Shared warmdown-phase controller for Muon beta2."""
    early = 0.8 + (0.95 - 0.8) * min(1.0, step / 500.0)
    warmdown_start = steps * 0.5
    if step <= warmdown_start:
        return early

    phase = min(1.0, (step - warmdown_start) / max(steps - warmdown_start, 1.0))
    return max(early, 0.95 + (0.99 - 0.95) * phase)


def get_h533_beta2(step: int, group: dict[str, object]) -> float:
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
    adamw_group_names = [
        "lm_head",
        "embedding",
        "value_embeds",
        "resid_lambdas",
        "x0_lambdas",
    ]
    adamw_index = 0
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
        if group["kind"] == "adamw":
            group["group_name"] = adamw_group_names[adamw_index]
            group["base_betas"] = tuple(group["betas"])
            group["base_eps"] = group["eps"]
            adamw_index += 1
            continue

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


def main() -> None:
    args = parse_args()
    features = mutation_features(args.mutation)
    if args.mutation != "none":
        if not features:
            raise SystemExit(f"Unsupported mutation '{args.mutation}'. No recognized HXX features found.")
        unknown_features = sorted(features - SUPPORTED_FEATURES)
        if unknown_features:
            raise SystemExit(
                f"Unsupported mutation '{args.mutation}'. Unknown features: {', '.join(unknown_features)}"
            )

    os.environ["ENIGMA_MUTATION"] = args.mutation
    os.environ["ENIGMA_TOTAL_STEPS"] = str(args.steps)
    from enigma import stage5_patch  # noqa: F401

    from nanochat.dataloader import (
        tokenizing_distributed_data_loader_bos_bestfit,
        tokenizing_distributed_data_loader_with_state_bos_bestfit,
    )
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.loss_eval import evaluate_bpb
    from nanochat.tokenizer import get_token_bytes, get_tokenizer

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda"
    device_batch_size = 2
    max_seq_len = 512
    total_batch_size = 1024

    log(f"=== ENIGMA Stage 5 Production — {args.mutation} ===")
    log(f"Steps: {args.steps}, Depth: {args.depth}, Eval every: {args.eval_every}")
    log(f"Features: {sorted(features)}")

    config = GPTConfig(n_layer=args.depth, sequence_len=max_seq_len)
    model = GPT(config).to(device)
    model.train()

    optimizer = model.setup_optimizer(weight_decay=0.2)
    annotate_optimizer_groups(optimizer)

    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer,
        device_batch_size,
        max_seq_len,
        split="train",
        device=device,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        device_batch_size,
        max_seq_len,
        split="val",
        device=device,
    )

    tokens_per_fwdbwd = device_batch_size * max_seq_len
    if total_batch_size % tokens_per_fwdbwd != 0:
        raise ValueError("TOTAL_BATCH_SIZE must be divisible by tokens_per_fwdbwd")
    grad_accum_steps = total_batch_size // tokens_per_fwdbwd

    x, y, _ = next(train_loader)
    curve: list[dict[str, float | int]] = []
    train_curve: list[dict[str, float | int]] = []
    step_times_ms: list[float] = []
    failure_type: str | None = None

    total_start = time.perf_counter()
    for step in range(args.steps + 1):
        last_step = step == args.steps

        if step % args.eval_every == 0 or last_step:
            model.eval()
            val_loader = build_val_loader()
            eval_steps_count = max(1, 512 // tokens_per_fwdbwd)
            with torch.no_grad():
                val_bpb = evaluate_bpb(model, val_loader, eval_steps_count, token_bytes)
            curve.append({"step": step, "val_bpb": float(val_bpb)})
            model.train()
            log(f"  step {step:>5d}: val_bpb={val_bpb:.6f}")

        if last_step:
            break

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

        lrm = get_lr_multiplier(step, args.steps)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(step, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "adamw":
                group["betas"] = group["base_betas"]
                group["eps"] = group["base_eps"]
            else:
                group["beta2"] = group["base_beta2"]
                group["ns_steps"] = group["base_ns_steps"]
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay

        if "H517" in features:
            beta2 = get_h517_beta2(step, args.steps)
            for group in optimizer.param_groups:
                if group["kind"] == "muon":
                    group["beta2"] = beta2
        elif "H504" in features:
            beta2 = get_h504_beta2(step, args.steps)
            for group in optimizer.param_groups:
                if group["kind"] == "muon":
                    group["beta2"] = beta2
        elif "H533" in features:
            for group in optimizer.param_groups:
                if group["kind"] == "muon":
                    group["beta2"] = get_h533_beta2(step, group)
        elif "H60" in features:
            beta2 = get_h60_beta2(step)
            for group in optimizer.param_groups:
                if group["kind"] == "muon":
                    group["beta2"] = beta2

        if "H534" in features:
            ns_steps = get_h534_ns_steps(step, args.steps)
            for group in optimizer.param_groups:
                if group["kind"] == "muon":
                    group["ns_steps"] = ns_steps

        if "H535" in features:
            eps = get_h535_eps(step, args.steps)
            for group in optimizer.param_groups:
                if group["kind"] == "adamw" and group.get("group_name") in {"embedding", "value_embeds"}:
                    group["eps"] = eps

        if "H536" in features:
            beta1 = get_h536_x0_beta1(step, args.steps)
            for group in optimizer.param_groups:
                if group["kind"] == "adamw" and group.get("group_name") == "x0_lambdas":
                    _, beta2 = group["base_betas"]
                    group["betas"] = (beta1, beta2)

        if "H64" in features:
            # Faithful H64 is applied in the patched Muon step. Keep the production
            # momentum schedule for the EMA buffer; the patch separately schedules the
            # Nesterov blend coefficient.
            pass

        if "H532" in features:
            apply_h532_muon_vreset(optimizer, step, args.steps)

        if "H537" in features:
            apply_h537_embed_mom_reset(optimizer, step, args.steps)

        os.environ["_ENIGMA_STEP"] = str(step)
        optimizer.step()
        model.zero_grad(set_to_none=True)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        step_times_ms.append(elapsed_ms)

        if step % 500 == 0:
            train_bpb = train_loss / math.log(2.0)
            train_curve.append(
                {
                    "step": step,
                    "train_bpb": float(train_bpb),
                    "step_time_ms": float(elapsed_ms),
                }
            )
            log(f"  step {step:>5d}: train_bpb={train_bpb:.4f} ({elapsed_ms:.0f}ms)")

    total_time_s = time.perf_counter() - total_start

    val_points = [point["val_bpb"] for point in curve]
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
        "production_baseline": True,
        "baseline_schedules": {
            "lr": "linear_warmdown_50pct",
            "momentum": "warmup_0.85_to_0.95_over_300",
            "weight_decay": "linear_decay_to_0",
        },
        "compound_testing": len(features) > 1,
    }

    output_path.write_text(json.dumps(result, indent=2))
    log(f"\nSaved results to {output_path}")
    log(f"Final BPB: {final_bpb:.6f}")


if __name__ == "__main__":
    main()
