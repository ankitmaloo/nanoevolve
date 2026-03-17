# H603: Muon's Variance Signal → AdamW Eps

## Cross-domain inspiration: Cascade Control (Chemical/Process Engineering)

In industrial process control, cascade control uses an inner loop and an outer loop. The inner loop (fast, measures process directly) stabilizes fast dynamics. The outer loop (slower, measures output quality) sets the setpoint for the inner loop. The key insight: the inner loop's state contains information about disturbances that the outer loop can't measure directly.

In our case:
- **Muon = inner loop**: Operates on matrices, computes rich variance information via `second_momentum_buffer`, updates with orthogonalization
- **AdamW = outer loop**: Operates on embeddings/scalars, has only per-element `exp_avg_sq`, no structural information

Muon's `second_momentum_buffer` (per-neuron variance of orthogonal gradients) is a **direct measurement of loss landscape conditioning**. AdamW currently ignores this. If Muon says "the landscape is poorly conditioned" (high variance spread), AdamW should be conservative.

## Mechanism

Every N steps, compute a global conditioning metric from Muon's second_momentum_buffer. Use it to scale AdamW's eps.

```python
# Extract Muon conditioning signal
muon_variances = []
for group in optimizer.param_groups:
    if group["kind"] == "muon":
        state = optimizer.state[group["params"][0]]
        v = state["second_momentum_buffer"]
        muon_variances.append(v.float())

# Compute conditioning ratio (how spread are per-neuron variances?)
all_v = torch.cat([v.flatten() for v in muon_variances])
conditioning = all_v.max() / all_v.median().clamp_min(1e-12)
# conditioning ~ 1 means well-conditioned, >>1 means poorly conditioned

# Map to AdamW eps: poorly conditioned → high eps (conservative)
eps_scale = conditioning.clamp(min=1.0, max=1000.0)
adaptive_eps = base_eps * eps_scale
```

## Why this is the first true cross-optimizer coordination

- **Information flows from Muon → AdamW** through an actual signal, not a schedule
- Muon's orthogonalization + variance reduction gives it structural information about the loss landscape that AdamW can't compute from per-element gradients alone
- The conditioning metric captures something fundamentally different from what AdamW sees — the relative scales of features after removing rotation

## Pseudocode (training loop integration)

```python
# In training loop, every step or every N steps:
conditioning = compute_muon_conditioning(optimizer)

for group in optimizer.param_groups:
    if group["kind"] == "adamw":
        # Scale eps by Muon's conditioning signal
        group["eps"] = base_eps * min(conditioning, 1000.0)
```

## Variants

### H603a: Use Muon variance rate-of-change
Instead of the conditioning ratio, track how fast Muon's variance is changing. If Muon's second moment is still moving, AdamW should be conservative.

### H603b: Per-group Muon → AdamW mapping
Different AdamW groups (embedding vs lm_head vs scalars) may need different sensitivity to Muon's signal. Embeddings feed directly into Muon's matrices — they should be most sensitive.

## Risk

- Muon's second_momentum_buffer is computed post-orthogonalization, which removes information about the raw gradient landscape. The conditioning signal may be too processed.
- Requires reaching into optimizer state from the training loop — breaks the clean kernel abstraction.
- Conditioning metric is a single scalar from a rich tensor — information compression may lose the signal.
