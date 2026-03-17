# H620: Embedding-Matrix Gradient Coherence Penalty

## Cross-domain inspiration: Multi-Task Gradient Alignment / PCGrad / GradNorm

In multi-task learning, different task losses can produce conflicting gradients that cancel each other out. PCGrad (Yu et al. 2020) detects conflicting gradients between tasks and projects them to be non-conflicting. GradNorm (Chen et al. 2018) balances gradient magnitudes across tasks.

In our setup, embeddings and matrices aren't separate tasks, but they ARE separate parameter subspaces with separate optimizers. If the embedding gradient (via AdamW) and the matrix gradient (via Muon) are pulling in conflicting directions through the shared computation graph, neither optimizer achieves its full potential.

## Mechanism

Measure the coherence between embedding updates and matrix updates via their effect on the shared loss. When they're coherent (both reducing loss), proceed normally. When they conflict (one increases what the other decreases), reduce the conflicting update.

### Cheap proxy: track loss contribution from each parameter type

```python
# Every N steps, compute per-group loss contribution:
loss_total = model(x, y)

# Approximate: which parameter group's update reduced loss more?
# Use the inner product of gradient with update direction as proxy
for group in optimizer.param_groups:
    update_alignment = 0.0
    for p in group["params"]:
        if p.grad is not None:
            # How aligned is this step's gradient with the previous update?
            if "prev_update" in optimizer.state[p]:
                alignment = (p.grad * optimizer.state[p]["prev_update"]).sum()
                update_alignment += alignment.item()
    group["coherence"] = update_alignment

# If Muon and AdamW have opposite coherence signs, one is fighting the other
muon_coherence = sum(g["coherence"] for g in optimizer.param_groups if g["kind"] == "muon")
adamw_coherence = sum(g["coherence"] for g in optimizer.param_groups if g["kind"] == "adamw")

if muon_coherence * adamw_coherence < 0:
    # Conflict! Scale down the fighter
    if abs(muon_coherence) < abs(adamw_coherence):
        muon_lr_scale = 0.5  # Muon is the weaker fighter, slow it down
    else:
        adamw_lr_scale = 0.5
```

### Simpler version: just measure gradient magnitude ratio stability

```python
muon_grad_norm = sum(p.grad.norm().item() for g in optimizer.param_groups
                     if g["kind"] == "muon" for p in g["params"] if p.grad is not None)
adamw_grad_norm = sum(p.grad.norm().item() for g in optimizer.param_groups
                      if g["kind"] == "adamw" for p in g["params"] if p.grad is not None)

ratio = muon_grad_norm / (adamw_grad_norm + 1e-12)
ratio_ema = 0.95 * ratio_ema + 0.05 * ratio

# If ratio is volatile (embeddings and matrices alternately dominating gradients),
# both are fighting. Dampen the volatile one.
```

## Why this is architecturally interesting

Transformers have a specific structure: embeddings → attention/MLP layers → output. The gradient flows backward through this chain. If embeddings produce bad representations, all downstream matrix gradients are corrupted. If matrices overfit, embedding gradients are misleading.

This is the only hypothesis that treats the optimizer as a multi-agent system where the agents (Muon, AdamW) can interfere constructively or destructively. H603/H606/H608 share signals but don't detect or resolve conflicts.

## Risk

- Tracking per-group coherence adds significant overhead (inner products every step).
- The conflict detection might fire too often (some conflict is natural and healthy).
- LR scaling is a blunt response to a subtle signal.
- Storing prev_update per parameter doubles state memory.
