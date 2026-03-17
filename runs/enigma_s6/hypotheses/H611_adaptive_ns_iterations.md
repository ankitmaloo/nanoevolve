# H611: Adaptive Polar Express Iteration Count from Condition Number

## Cross-domain inspiration: Iterative Solvers in Numerical Linear Algebra

Iterative methods (conjugate gradient, GMRES, Newton-Schulz) converge faster on well-conditioned problems. The standard approach: run a fixed number of iterations. The smart approach: run until convergence or until a residual threshold is met. Krylov solvers adaptively choose iteration count based on the residual.

Muon uses 5 iterations of Polar Express (Newton-Schulz variant) to orthogonalize the gradient. This is fixed. But early in training, gradients are nearly random and 5 iterations might be overkill (or too few). Late in training, gradients have more structure and fewer iterations might suffice.

## Mechanism

After each Polar Express step, check how close the result is to orthogonal. Stop early if already converged, or flag if not converged enough.

```python
# Inside muon_step_fused, after each polar express iteration:
for i, (a, b, c) in enumerate(polar_express_coeffs[:max_ns_steps]):
    if g.size(-2) > g.size(-1):
        A = X.mT @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B
    else:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    # Check orthogonality: for perfect orthogonal X, X^T X = I
    # Residual = ||X^T X - I||_F / sqrt(n)
    if i >= min_ns_steps - 1:
        if g.size(-2) > g.size(-1):
            residual = (X.mT @ X - eye).norm(dim=(-2,-1)).mean()
        else:
            residual = (X @ X.mT - eye).norm(dim=(-2,-1)).mean()
        if residual < threshold:
            break
```

## Simpler version: schedule ns_steps inversely to Muon's second_momentum_buffer stability

```python
# In training loop:
muon_v_change = compute_second_moment_change_rate(optimizer)
if muon_v_change > 0.1:
    ns_steps = 5  # noisy → need full orthogonalization
elif muon_v_change < 0.01:
    ns_steps = 3  # stable → less orthogonalization needed
else:
    ns_steps = 4

for group in optimizer.param_groups:
    if group["kind"] == "muon":
        group["ns_steps"] = ns_steps
```

## Why this matters

Each Polar Express iteration is expensive — it's the dominant cost in Muon (matrix multiplications). Going from 5→3 iterations saves 40% of Muon's compute. If the gradient is already well-conditioned (late training), those extra iterations are wasted FLOPs.

Note: H534 (staged NS depth) from Stage 5 tried a fixed schedule (4 early, 5 late). It didn't win. But H534 was time-based. An adaptive version based on actual orthogonality residual might be different.

## Risk

- Dynamic iteration count breaks `torch.compile` (graph must be static). Would need to compile separate kernels for ns=3,4,5 and dispatch.
- Fewer iterations = less orthogonal = update quality degrades. The tradeoff might not be worth it for BPB even if it saves FLOPs.
- For the speedrun (time-to-GPT2), FLOPs savings matter. For BPB-only experiments, they don't.
