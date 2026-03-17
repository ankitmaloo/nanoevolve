# H607: Homeostatic Eps — Maintain Target Effective Step Size

## Cross-domain inspiration: Homeostatic Plasticity (Neuroscience)

Biological neurons maintain a target firing rate through homeostatic mechanisms. When a neuron fires too much, it downregulates its sensitivity. When it fires too little, it upregulates. The neuron doesn't follow a schedule — it measures its own activity and adjusts to maintain homeostasis.

This is fundamentally different from "schedule-based" approaches. The system has a setpoint and a feedback loop, not a timer.

In AdamW, the "effective step size" for parameter p_i at step t is:

```
effective_step_i = lr * exp_avg_i / (sqrt(exp_avg_sq_i / bias2) + eps)
```

Early training: `exp_avg_sq` is tiny → effective step is huge (or capped by eps).
Late training: `exp_avg_sq` has converged → effective step reflects true curvature.

H73 handles this with a fixed schedule. H607 asks: what if eps self-adjusts to maintain a target effective step size magnitude?

## Mechanism

Track the average effective step size. If it's too large, increase eps (clip harder). If it's too small, decrease eps (allow more adaptivity).

```python
# After computing the update but before applying:
update_magnitude = (exp_avg / denom).abs().mean()

# Target: should decrease smoothly with lr (as lr warms down, updates shrink)
target_magnitude = lr_t * target_update_scale  # target_update_scale ~ 1e-3

# Feedback: adjust eps to push magnitude toward target
if update_magnitude > target_magnitude * 2.0:
    # Updates too large → increase eps
    eps_multiplier *= 1.01
elif update_magnitude < target_magnitude * 0.5:
    # Updates too small → decrease eps
    eps_multiplier *= 0.99

eps_multiplier = clamp(eps_multiplier, 1.0, 1e4)
adaptive_eps = base_eps * eps_multiplier
```

## Elegant formulation

A cleaner version using direct feedback:

```python
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_h607(p, grad, exp_avg, exp_avg_sq, eps_state,
                     step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t

    # Compute update with current eps
    denom = (exp_avg_sq / bias2).sqrt() + eps_state
    update = exp_avg / denom

    # Homeostatic feedback: adjust eps_state based on update magnitude
    update_rms = update.square().mean().sqrt()
    target_rms = lr_t * 0.001  # target scale

    # Multiplicative update to eps_state (slow, stable)
    ratio = (update_rms / (target_rms + 1e-12)).clamp(0.5, 2.0)
    eps_state.mul_(ratio ** 0.01)  # very slow adjustment
    eps_state.clamp_(min=1e-12, max=1e-4)

    step_size = lr_t / bias1
    p.add_(update, alpha=-step_size)
```

## Why this is powerful

- **No schedule, no step count, no batch size knowledge**: Pure feedback control
- **Per-parameter group adaptation**: Different groups (embedding vs lm_head vs scalars) naturally find different eps values based on their update dynamics
- **Self-correcting**: If something goes wrong (loss spike, gradient explosion), update magnitudes jump, eps increases automatically to dampen the response
- **Works at any scale**: The feedback loop adapts to whatever the optimizer encounters

## Analogy to PID control

This is essentially a P-controller (proportional only) on the effective step size. Could extend to PID:
- P: current update magnitude vs target
- I: accumulated deviation over time (prevents steady-state error)
- D: rate of change of update magnitude (anticipates spikes)

## Risk

- `target_update_scale` is a new hyperparameter. If set wrong, could chronically over- or under-regulate.
- The feedback loop adds coupling — could oscillate if the gain (0.01 exponent) is too high.
- Per-element eps_state requires same memory as exp_avg_sq. But could use a single scalar per param group instead.
