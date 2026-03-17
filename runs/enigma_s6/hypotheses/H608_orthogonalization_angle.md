# H608: Muon Orthogonalization Angle → Embedding Confidence

## Cross-domain inspiration: Residual Analysis (Statistics) / Model Diagnostics

In regression, you fit a model and examine the residuals. Large residuals mean the model is wrong about those data points. Systematically structured residuals (not random) indicate model mis-specification.

Muon's polar express orthogonalization transforms the raw gradient `G` into a quasi-orthogonal matrix `U ≈ UV^T` from the SVD of `G = USV^T`. The "residual" of this transformation — how much the direction changed — tells you about the structure of the gradient.

If the raw gradient is already nearly orthogonal (small angle change), the loss landscape is well-conditioned for matrices. If it requires heavy rotation (large angle change), something is distorting the gradient — and that something is often the embeddings feeding in bad representations.

## Mechanism

Compute the cosine similarity between the pre- and post-orthogonalization gradient. Use this as a confidence signal for embedding updates.

```python
# Inside muon_step_fused (or a wrapper):
# Before orthogonalization:
g_pre = stacked_grads.clone()  # or just momentum-blended g

# After orthogonalization (polar express):
g_post = X  # the orthogonalized result

# Cosine similarity per matrix
cos_angle = (g_pre * g_post).sum(dim=(-2,-1)) / (
    g_pre.norm(dim=(-2,-1)) * g_post.norm(dim=(-2,-1)) + 1e-8
)  # shape (K,) — one value per matrix param group

# Average across all Muon groups
mean_cos = cos_angle.mean().item()
# Range: 0 (completely different direction) to 1 (same direction)

# Expose to training loop
optimizer._muon_gradient_alignment = mean_cos
```

In the training loop:
```python
alignment = getattr(optimizer, '_muon_gradient_alignment', 1.0)

for group in optimizer.param_groups:
    if group["kind"] == "adamw" and group.get("group_name") in {"embedding", "value_embeds"}:
        # Low alignment → embeddings might be causing bad gradients → be conservative
        if alignment < 0.5:
            group["lr"] *= alignment  # scale down embedding LR
        # High alignment → embeddings are helping → full speed
```

## Why embedding LR specifically?

Embeddings are the **input** to the transformer matrices. If Muon's gradients are "wrong" (need heavy orthogonalization correction), the most likely cause is that embeddings are producing representations that create ill-conditioned gradient matrices. Slowing down embedding updates when this happens gives the matrices time to adjust.

This is the reverse direction of H603 (Muon → AdamW). H603 shares Muon's variance signal. H608 shares Muon's directional signal.

## Variants

### H608a: Gate weight decay instead of LR
When alignment is low, increase embedding weight decay to regularize them back toward a more stable manifold.

### H608b: Use angle to control Muon's own momentum
Low alignment → gradients are very different from momentum buffer → reduce momentum (trust the new gradient more). This is similar to H64 but adaptive rather than scheduled.

## Risk

- Computing `g_pre.clone()` inside the compiled kernel might break `torch.compile` or add memory overhead.
- The cosine angle might be naturally low early in training (random initialization) and naturally high late — could just recapitulate a schedule.
- Need to verify the angle is actually informative and not noise.
