# H615: Sharpness-Aware Muon (SAM for Matrix Parameters)

## Cross-domain inspiration: SAM / Loss Geometry / Flat Minima

Sharpness-Aware Minimization (Foret et al. 2021) seeks flat minima by first perturbing parameters to the worst-case neighbor within an epsilon ball, then computing gradients at that perturbed point. Flat minima generalize better than sharp ones.

SAM is typically applied to all parameters uniformly. But Muon already does something related — orthogonalization removes the "sharp" directions from the gradient (high singular values get compressed). H615 asks: what if we explicitly seek flat minima for the matrix parameters before orthogonalization?

## Mechanism

Before computing Muon's gradient, perturb matrix parameters in the ascent direction, then compute gradients at the perturbed point.

```python
# Standard SAM perturbation:
# p_perturbed = p + rho * grad / ||grad||

# For Muon, applied to stacked params:
def muon_step_sam(self, group):
    params = group["params"]
    # Forward pass already computed gradients at current params
    # Now: perturb in gradient direction
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                perturbation = group["sam_rho"] * p.grad / (p.grad.norm() + 1e-12)
                p.add_(perturbation)

    # Recompute gradients at perturbed point (requires second forward+backward)
    # ... this is expensive ...

    # Undo perturbation
    with torch.no_grad():
        for p in params:
            p.sub_(perturbation)

    # Now proceed with standard Muon step using the SAM gradients
    # ...
```

## Cheap alternative: use second moment as sharpness proxy

Instead of a second forward pass, estimate sharpness from the variance of recent gradients (which Muon already tracks in `second_momentum_buffer`). High variance = sharp landscape.

```python
# Muon's second_momentum_buffer already captures per-neuron variance
# Scale the update inversely: sharp neurons get smaller updates
sharpness_scale = second_momentum_buffer.rsqrt()
# This is... exactly what NorMuon variance reduction already does.
```

Wait — Muon's variance reduction IS already a form of sharpness-awareness. The `second_momentum_buffer → step_size → final_scale` pipeline normalizes updates so that high-variance (sharp) neurons get smaller updates. H615 might be redundant with existing NorMuon.

## What's NOT captured by NorMuon

NorMuon normalizes per-neuron update magnitude, but doesn't change the direction. SAM changes the direction by computing gradients at the worst-case neighbor. The directional change is what makes SAM seek flat minima.

## Risk

- Full SAM requires 2x forward+backward passes = 2x compute. Incompatible with speedrun goal.
- Cheap proxies might not capture the directional benefit of SAM.
- Muon's orthogonalization might already be doing something similar to the directional component of SAM.
