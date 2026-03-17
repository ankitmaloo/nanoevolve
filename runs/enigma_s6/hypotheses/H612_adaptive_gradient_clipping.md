# H612: Gradient Clipping from Loss Curvature Estimate

## Cross-domain inspiration: Trust Region Methods (Optimization Theory)

Trust region methods (Levenberg-Marquardt, TRPO in RL) don't just follow the gradient — they estimate how far the local linear approximation is valid and limit the step to that region. When the loss landscape is highly curved, take small steps. When flat, take large steps.

Current production has no gradient clipping. Muon's orthogonalization implicitly normalizes update magnitude, but AdamW has no such mechanism. Early in training, a few gradient outliers on embedding dimensions can cause disproportionate updates.

## Mechanism

Estimate local curvature using the change in gradient between consecutive steps (finite-difference Hessian-vector product). Clip gradients proportionally.

```python
# For AdamW groups, track previous gradient:
for group in optimizer.param_groups:
    if group["kind"] == "adamw":
        for p in group["params"]:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            g = p.grad

            if "prev_grad" in state:
                # Finite-difference curvature estimate: ||g_t - g_{t-1}|| / ||p_t - p_{t-1}||
                grad_diff = g - state["prev_grad"]
                param_diff = p - state["prev_param"]
                curvature = grad_diff.norm() / (param_diff.norm() + 1e-12)

                # Clip threshold inversely proportional to curvature
                clip_threshold = group["lr"] / (curvature + 1e-8)
                grad_norm = g.norm()
                if grad_norm > clip_threshold:
                    g.mul_(clip_threshold / grad_norm)

            state["prev_grad"] = g.clone()
            state["prev_param"] = p.clone()
```

## Why this complements H73

H73 addresses the denominator (eps controls effective step size). H612 addresses the numerator (clip the gradient itself). These are orthogonal — H73 says "don't trust the variance estimate early," H612 says "don't trust the gradient magnitude when the landscape is curved."

## Risk

- Storing prev_grad and prev_param doubles AdamW memory.
- Finite-difference curvature is noisy and may not be meaningful with stochastic gradients.
- A simpler approach (just global gradient norm clipping) might capture 90% of the benefit with none of the complexity.
