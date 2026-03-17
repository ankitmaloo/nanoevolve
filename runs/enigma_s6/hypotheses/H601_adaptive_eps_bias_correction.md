# H601: Adaptive Eps from Bias Correction

## Cross-domain inspiration: Kalman Filter Gain Scheduling

In Kalman filtering, the gain matrix K starts large (trust measurements, distrust model) and shrinks as the state estimate converges. The gain is derived from the estimation error covariance — you don't schedule it with a timer, you compute it from how uncertain you actually are.

AdamW's bias correction `1 - beta2^step` already encodes "how reliable is the second moment estimate?" But this factor is only applied to the numerator (`exp_avg_sq / bias2`), never to epsilon. H73 discovered that scheduling eps helps — but the schedule is time-based and doesn't know when the second moment has actually converged.

## Mechanism

Apply the inverse bias correction to eps itself:

```python
bias2 = 1 - beta2_t ** step_t
# Standard: denom = (exp_avg_sq / bias2).sqrt() + eps_t
# H601:     denom = (exp_avg_sq / bias2).sqrt() + eps_base / bias2
```

When `step=1`, `bias2 ≈ 0.05` (for beta2=0.95), so effective eps = `1e-10 / 0.05 = 2e-9`.
When `step=100`, `bias2 ≈ 0.994`, so effective eps ≈ `1e-10`.

This naturally gives large eps early (when second moments are unreliable) and converges to `base_eps` as bias correction converges to 1. No schedule, no step count needed, adapts to any batch size because larger batches fill the EMA faster.

## Pseudocode

```python
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_h601(p, grad, exp_avg, exp_avg_sq, step_t, lr_t,
                     beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Key change: eps scaled by inverse bias correction
    adaptive_eps = eps_t / bias2
    denom = (exp_avg_sq / bias2).sqrt() + adaptive_eps
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)
```

## Why this might beat H73

- **Batch-size invariant**: Larger batches mean gradients are less noisy, so `exp_avg_sq` fills in faster. The bias correction tracks this automatically. H73's fixed 20k-step schedule doesn't.
- **No hyperparameters**: Zero new knobs. Just applies existing math consistently.
- **Principled**: The bias correction is exactly the right quantity — it measures "how much of the true second moment have I seen?"

## Risk

- The bias correction converges very fast (by step ~60 for beta2=0.95). H73 schedules over 20k steps. If the real issue is not just initialization but ongoing variance heterogeneity, H601 might not help enough in the later phase.
- Might need `eps_base` higher than 1e-10 to see any effect (e.g. 1e-8).
