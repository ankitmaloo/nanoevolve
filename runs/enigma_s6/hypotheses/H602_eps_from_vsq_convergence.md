# H602: Eps from Second-Moment Convergence Rate

## Cross-domain inspiration: Adaptive Filter Theory (LMS/RLS)

In adaptive signal processing, the LMS (Least Mean Squares) filter uses a step size that depends on the signal power estimate. When the estimate is volatile (changing rapidly), the filter is conservative. When it stabilizes, the filter becomes aggressive. The RLS (Recursive Least Squares) filter takes this further — it tracks the estimation error covariance explicitly and adjusts gains accordingly.

AdamW's `exp_avg_sq` is an EMA of squared gradients. When it's changing rapidly, it's unreliable. When it's stable, we can trust it. Currently eps is blind to this.

## Mechanism

Track the rate of change of `exp_avg_sq` using a second EMA. When the second moment is volatile, eps is high. When it stabilizes, eps drops.

```python
# New state: track EMA of |delta(exp_avg_sq)|
delta_sq = (grad.square() - exp_avg_sq).abs()
volatility.lerp_(delta_sq, 1 - beta_vol)  # beta_vol ~ 0.99

# Adaptive eps: proportional to relative volatility
relative_vol = volatility / (exp_avg_sq.abs() + 1e-12)
adaptive_eps = eps_base * (1.0 + alpha * relative_vol)
```

When `exp_avg_sq` is changing by 50% per step (early training), `relative_vol ≈ 0.5` → `eps ≈ eps_base * (1 + alpha * 0.5)`.
When `exp_avg_sq` is changing by 1% per step (late training), `relative_vol ≈ 0.01` → `eps ≈ eps_base`.

## Pseudocode

```python
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_h602(p, grad, exp_avg, exp_avg_sq, volatility,
                     step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)

    # Track volatility of second moment
    new_sq = grad.square()
    delta = (new_sq - exp_avg_sq).abs()
    volatility.lerp_(delta, 0.01)  # slow EMA of change rate

    exp_avg_sq.lerp_(new_sq, 1 - beta2_t)

    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t

    # Adaptive eps: high when volatile, low when stable
    rel_vol = volatility / (exp_avg_sq.abs() + 1e-12)
    adaptive_eps = eps_t * (1.0 + 100.0 * rel_vol.mean())  # scalar scaling

    denom = (exp_avg_sq / bias2).sqrt() + adaptive_eps
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)
```

## Why this is interesting

- **Per-parameter awareness**: Different parameters converge at different rates. Embedding variance stabilizes differently from scalar variance.
- **Batch-size adaptive**: Larger batches → less gradient noise → `exp_avg_sq` converges faster → volatility drops faster → eps drops faster.
- **Could work at any scale**: No step count or total_steps needed.

## Risk

- Extra state tensor (volatility) — same shape as exp_avg_sq. 33% more memory for AdamW params.
- The `alpha` scaling factor is a new hyperparameter.
- `rel_vol.mean()` collapses per-element info to a scalar — might lose the per-coordinate benefit.
