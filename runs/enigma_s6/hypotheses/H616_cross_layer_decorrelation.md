# H616: Cross-Layer Gradient Decorrelation

## Cross-domain inspiration: Whitening in Statistics / Decorrelated Batch Normalization

In statistics, whitening transforms correlated inputs into uncorrelated ones, improving the efficiency of downstream estimators. Decorrelated Batch Normalization (Huang et al. 2018) whitens activations across layers to speed up training.

But nobody whitens the gradients across layers. If layer 3 and layer 7 receive highly correlated gradient updates, they're doing redundant work — the optimizer's total update budget is wasted on moving the same direction in two places.

## Mechanism

Compute correlation between gradient directions across Muon param groups (each group is typically one layer's matrices). Penalize correlated updates by scaling down the correlated component.

```python
# Collect per-group gradient directions
directions = []
for group in optimizer.param_groups:
    if group["kind"] == "muon":
        g = torch.stack([p.grad.flatten() for p in group["params"]])
        directions.append(g.flatten() / (g.norm() + 1e-8))

# Compute pairwise cosine similarities
n = len(directions)
for i in range(n):
    for j in range(i+1, n):
        cos_sim = (directions[i] * directions[j]).sum()
        if cos_sim > 0.5:  # highly correlated
            # Remove the correlated component from the later layer
            # (Gram-Schmidt-like projection)
            directions[j] -= cos_sim * directions[i]
            directions[j] /= (directions[j].norm() + 1e-8)

# Apply decorrelated directions back to gradients
# (Scale original gradients by the decorrelation factor)
```

## Why this might help

In transformers, attention and MLP matrices at different layers can develop similar update patterns, especially early in training when representations haven't differentiated. Decorrelation forces each layer to contribute unique information to the overall update, potentially speeding up feature diversification.

## Risk

- O(n^2) pairwise comparisons across layers. For d12 (12 layers × ~3 groups each = 36 groups), that's 630 pairs. Could be expensive.
- Flattening matrices for cosine similarity loses structural information.
- Gradient decorrelation is not well-studied and might hurt — correlated updates might be correlated for a good reason (shared useful gradient signal).
