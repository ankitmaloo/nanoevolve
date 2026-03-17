# H613: State Resets at Phase Boundaries

## Cross-domain inspiration: Simulated Annealing with Restarts / Cyclical Learning Rates

In simulated annealing, periodic restarts (reheating) help escape local minima. In cyclical LR schedules (Smith 2017), periodic LR increases serve the same purpose. The key insight: optimizer state (momentum buffers, variance estimates) accumulates assumptions about the loss landscape that become stale when the regime changes.

Production NanoChat has a clear phase boundary: the warmdown starts at 50% of training. LR begins dropping linearly to 0. But all optimizer state (momentum buffers, second moments) carries over from the pre-warmdown phase without adjustment.

## Mechanism

At the warmdown boundary, partially reset or rescale optimizer state to prevent stale estimates from dominating the warmdown phase.

```python
warmdown_start = num_iterations // 2

if step == warmdown_start:
    for group in optimizer.param_groups:
        if group["kind"] == "muon":
            state = optimizer.state[group["params"][0]]
            # Muon: second moment was calibrated for high-LR regime
            # Warmdown has different gradient statistics → rescale
            state["second_momentum_buffer"].mul_(0.5)  # Shrink, let it re-adapt

        if group["kind"] == "adamw":
            for p in group["params"]:
                state = optimizer.state[p]
                # AdamW: first moment (momentum) carries pre-warmdown direction
                # As LR drops, old momentum direction becomes stale
                if "exp_avg" in state:
                    state["exp_avg"].mul_(0.25)  # Partial reset
```

## Variants

### H613a: Muon second-moment reset only
Just reset `second_momentum_buffer` at warmdown. This was tested as H532 in Stage 5 composites — it didn't win solo but was part of some compounds. Worth retesting at 1500-shard scale.

### H613b: AdamW momentum reset at warmdown
Reset `exp_avg` (first moment) at warmdown start. The momentum direction accumulated during high-LR training may be suboptimal for the warmdown trajectory.

### H613c: Full state reset + re-warmup
At warmdown boundary, reset all optimizer state and re-run warmup schedules (momentum, beta2, eps) for a short period. Like a mini training restart.

## Why this matters for the speedrun

The speedrun has ~2200 steps. Warmdown starts at step ~1100. If optimizer state from steps 0-1100 is misleading for steps 1100-2200, a reset could significantly improve the warmdown phase. This is different from H73 (which helps early phase) — H613 helps the late phase.

## Risk

- Resetting state destroys information. If the accumulated state is actually helpful during warmdown, this hurts.
- H532 (Muon variance reset) was tested at 3-shard scale and was mediocre. May need to be combined with something else.
