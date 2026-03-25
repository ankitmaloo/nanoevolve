# H619: Muon Variance Reset at Phase Boundaries

## Cross-domain inspiration: Cold Restart in Optimization / SGDR

Loshchilov & Hutter (2017) showed that periodic LR restarts (warm restarts) help SGD escape local minima. The mechanism: resetting the learning rate also effectively resets the optimizer's "exploration radius." Stale momentum and variance estimates from a previous phase can trap the optimizer in a suboptimal basin.

H532 from Stage 5 tested resetting Muon's `second_momentum_buffer` at warmdown start by scaling it by 0.25. It was mediocre solo but appeared in some compounds. H619 refines this with a principled reset strategy.

## Mechanism

At the warmdown boundary, partially reset Muon's second moment to force re-adaptation to the warmdown gradient statistics.

```python
warmdown_start = num_iterations // 2

if step == warmdown_start:
    for group in optimizer.param_groups:
        if group["kind"] == "muon":
            state = optimizer.state[group["params"][0]]
            v = state["second_momentum_buffer"]

            # Option A: Scale down (H532-style, but less aggressive)
            v.mul_(0.5)

            # Option B: Reset to current instantaneous variance
            # (re-seed from the latest gradient, like H538 does for AdamW)
            stacked_grads = torch.stack([p.grad for p in group["params"]])
            X = stacked_grads.bfloat16()
            # ... run polar express to get orthogonal gradient ...
            v_fresh = X.float().square().mean(dim=red_dim, keepdim=True)
            state["second_momentum_buffer"] = v_fresh.to(v.dtype)

            # Option C: Blend old and fresh
            v.lerp_(v_fresh.to(v.dtype), 0.5)
```

## Why warmdown boundary specifically?

During warmdown, LR drops linearly to 0. This fundamentally changes the gradient statistics:
- Pre-warmdown: gradients are "full strength," variance reflects the true loss landscape
- During warmdown: effective gradients shrink with LR, but second_momentum_buffer still holds pre-warmdown scale estimates

The mismatch means Muon's per-neuron scaling (from NorMuon) is calibrated for a regime that no longer exists. A reset forces re-calibration.

## Compounding with H73

H73 helps early phase (eps warmup). H619 helps late phase (variance reset at warmdown). They're orthogonal in time — could compound positively.

## Risk

- H532 already tested this and was mediocre at 3-shard scale. Might be equally mediocre at 1500-shard scale.
- Resetting variance destroys per-neuron scaling information that took thousands of steps to accumulate.
- The warmdown boundary is sharp — a gradual transition might be better.
