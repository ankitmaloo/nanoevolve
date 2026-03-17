# H609: Layer-wise LR Scaling from Gradient Norms

## Cross-domain inspiration: Thermodynamic Depth (Physics) / Heat Equation

In the heat equation, temperature changes propagate through a material at a rate that depends on local thermal conductivity. Deep layers of a material equilibrate last. In deep networks, the same thing happens — gradients attenuate or explode through layers, so deep layers receive different signal quality than shallow layers.

Production Muon applies the same LR to all matrix param groups regardless of layer position. But gradient norms vary significantly across layers — early layers (close to embeddings) see larger gradients, deep layers see attenuated ones. A flat LR means deep layers are under-updated relative to shallow layers.

## Mechanism

Compute per-layer gradient RMS and scale LR inversely — layers with smaller gradients get proportionally larger LR to maintain uniform effective update magnitude.

```python
# Per Muon group (each group is one layer's matrices):
for group in optimizer.param_groups:
    if group["kind"] == "muon":
        grad_rms = torch.stack([p.grad.norm() for p in group["params"]]).mean()
        # Track with EMA for stability
        group["grad_rms_ema"] = 0.95 * group.get("grad_rms_ema", grad_rms.item()) + 0.05 * grad_rms.item()

# Compute median as reference
all_rms = [g["grad_rms_ema"] for g in optimizer.param_groups if g["kind"] == "muon"]
median_rms = sorted(all_rms)[len(all_rms) // 2]

# Scale LR: layers with small gradients get boosted, large ones get damped
for group in optimizer.param_groups:
    if group["kind"] == "muon":
        ratio = median_rms / (group["grad_rms_ema"] + 1e-12)
        # Clamp to prevent extreme scaling
        ratio = max(0.5, min(2.0, ratio))
        group["lr"] = group["initial_lr"] * lrm * ratio
```

## Why this is different from existing approaches

- Production applies one LR multiplier to all Muon groups
- LARS/LAMB scale by param norm, not gradient norm — different signal
- This is gradient-norm equalization, like batch norm but for optimizer steps

## Risk

- Muon's orthogonalization already normalizes gradients — the per-layer variance might be small post-orthogonalization
- Could destabilize training if layer-wise ratios oscillate
