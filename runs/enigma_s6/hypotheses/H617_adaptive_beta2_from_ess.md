# H617: Adaptive Beta2 from Effective Sample Size

## Cross-domain inspiration: Importance Sampling (Monte Carlo Methods)

In Monte Carlo methods, the Effective Sample Size (ESS) measures how many independent samples your weighted sample is worth. If weights are uniform, ESS = N. If one sample dominates, ESS ≈ 1. Practitioners adjust their estimator's smoothing based on ESS — more smoothing when ESS is low (few effective samples), less when ESS is high.

AdamW and Muon both use beta2 to control how fast the second moment EMA adapts. beta2=0.95 means effective window ≈ 20 steps. But not all steps are equally informative. A step with a representative gradient sample is worth more than a step with an outlier.

## Mechanism

Track gradient diversity (how different each new gradient is from the running average). When gradients are diverse, each step is informative → lower beta2 (adapt faster). When gradients are redundant, steps carry less new info → higher beta2 (smooth more).

```python
# For each optimizer group:
for group in optimizer.param_groups:
    for p in group["params"]:
        if p.grad is None:
            continue
        state = optimizer.state[p]

        # Compute "surprise" of this gradient relative to second moment
        g_sq = p.grad.square()
        expected_sq = state["exp_avg_sq"]
        surprise = (g_sq / (expected_sq + 1e-12)).mean()
        # surprise ≈ 1 means gradient matches expectation (low info)
        # surprise >> 1 means gradient is surprising (high info)

        # EMA of surprise
        state["surprise_ema"] = 0.95 * state.get("surprise_ema", 1.0) + 0.05 * surprise.item()

        # Map to beta2: high surprise → low beta2 (adapt fast)
        #               low surprise → high beta2 (smooth)
        # ESS-inspired: beta2 = 1 - 1/ESS, where ESS = f(surprise)
        ess = max(1.0, 20.0 / state["surprise_ema"])
        adaptive_beta2 = 1.0 - 1.0 / ess
        adaptive_beta2 = max(0.8, min(0.999, adaptive_beta2))

    # Apply to group
    if group["kind"] == "muon":
        group["beta2"] = adaptive_beta2
    elif group["kind"] == "adamw":
        b1, _ = group["betas"]
        group["betas"] = (b1, adaptive_beta2)
```

## Why this applies to both Muon and AdamW

Both use beta2-controlled EMAs for their second moments. Both face the same problem: fixed beta2 is suboptimal because gradient informativeness varies over training. H60 (Muon beta2 warmup) proved this on the Muon side. H617 generalizes it to both optimizers and makes it data-driven rather than scheduled.

## Builds on H60

H60 = fixed schedule 0.8→0.95 over 500 steps, Muon only, +0.019 BPB.
H617 = adaptive schedule based on gradient surprise, both optimizers.

If H60's gain comes from the general principle (adapt beta2 to gradient quality), H617 captures that principle without needing to know the right schedule.

## Risk

- Per-element surprise computation is expensive (same cost as an extra gradient square).
- The surprise → ESS → beta2 mapping has arbitrary functional forms.
- Applying the same beta2 to a whole group (from per-element statistics) loses granularity.
