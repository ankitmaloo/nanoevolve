# H610: Adaptive Weight Decay from Parameter-Gradient Correlation

## Cross-domain inspiration: Regularization Theory / Overfitting Detection

In classical statistics, you increase regularization (ridge penalty) when the model is overfitting — when it's tracking noise rather than signal. The sign of overfitting: the model's parameters are large and the gradients are pushing them larger (positive correlation between param direction and gradient direction).

Production Muon already uses "cautious" weight decay: `mask = (g * params) >= 0`. It only applies WD when the gradient and parameter agree in sign. But the magnitude is fixed on a schedule (`wd * (1 - step/steps)`).

What if WD magnitude adapts to how much the optimizer is "agreeing with itself"?

## Mechanism

When gradient consistently aligns with parameters (pushing them larger), increase WD. When gradient opposes parameters (self-correcting), decrease WD.

```python
# For Muon groups:
for group in optimizer.param_groups:
    if group["kind"] == "muon":
        for p in group["params"]:
            if p.grad is not None:
                # Cosine similarity between param and gradient
                alignment = (p * p.grad).sum() / (p.norm() * p.grad.norm() + 1e-8)
                # When alignment > 0: params growing in gradient direction → overfit risk
                # When alignment < 0: self-correcting → less regularization needed

        # EMA of alignment across the group
        group["alignment_ema"] = 0.95 * group.get("alignment_ema", 0.0) + 0.05 * alignment.item()

        # Scale WD: high alignment → more WD, low → less
        wd_scale = 1.0 + group["alignment_ema"]  # range [0, 2] roughly
        group["weight_decay"] = base_wd * wd_scale * (1.0 - step / steps)
```

## Why this matters

Production WD decays linearly to 0, which is a time-based assumption. But the actual need for regularization depends on training dynamics. Early training might need more WD (random params, gradients push everywhere). Late training with warmdown might need less (already converging).

The cautious mask is binary — apply WD or don't. This makes it continuous and proportional.

## Risk

- WD is already on a linear decay schedule that works well. Adding adaptation on top of a working schedule might add noise.
- Alignment metric might be dominated by a few large parameters.
