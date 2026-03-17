# H618: Weight Decay ↔ Eps Duality — Co-Schedule WD and Eps

## Cross-domain inspiration: Regularization-Optimization Duality / Bayesian MAP Estimation

In Bayesian inference, weight decay is equivalent to a Gaussian prior (L2 regularization), and the optimizer's step size control (including eps) determines how strongly the likelihood (data) overrides the prior. There's a duality: tightening the prior (more WD) and loosening the likelihood (more eps) have similar effects — both push the optimizer toward simpler solutions.

H73 schedules eps from high (conservative) to low (aggressive). What if we co-schedule WD in the opposite direction? Start with low WD (let parameters explore) when eps is high (which already clips large updates), then increase WD as eps drops (when the optimizer becomes more aggressive, add regularization to compensate).

## Mechanism

```python
# H73: eps = 10^(-6 + -4 * frac)  → high early, low late
# H618: wd = base_wd * (1 + alpha * (1 - frac))  → high when eps is high? No...
# Actually: invert the relationship

# When eps is high (early): updates are clipped → less need for WD
# When eps is low (late): updates are unconstrained → more need for WD

# But production already decays WD linearly to 0...
# H618 proposes: DON'T decay WD to 0. Instead, couple it to eps.

frac = min(1.0, step / total_steps)
eps = 10.0 ** (-6.0 + (-4.0 * frac))

# WD increases as eps decreases (compensating)
# When eps=1e-6 (early): wd = base_wd * 0.5 (light regularization, eps already clips)
# When eps=1e-10 (late): wd = base_wd * 1.5 (heavy regularization, eps no longer clips)
eps_normalized = (math.log10(eps) + 10) / 4.0  # maps [1e-10, 1e-6] → [0, 1]
wd = base_wd * (0.5 + 1.0 * (1 - eps_normalized))
```

## Why this is principled

The effective regularization of AdamW comes from two sources:
1. Explicit weight decay: `p -= lr * wd * p`
2. Implicit regularization from eps: large eps → updates are damped → implicit L2-like effect

When H73 makes eps large early, it's implicitly adding regularization. When eps drops late, that implicit regularization disappears. H618 compensates by adding explicit WD as implicit regularization fades.

This maintains a constant total regularization budget throughout training, just shifting between implicit (eps) and explicit (WD) forms.

## Risk

- Production's linear WD decay to 0 is well-tested. Keeping WD high late in training contradicts the conventional wisdom that regularization should weaken during warmdown.
- The eps-WD coupling adds complexity for potentially marginal gain on top of H73.
- Only applies to AdamW groups. Muon's WD is separate and has its own decay.
