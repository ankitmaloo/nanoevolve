# Stage 4 Context — Schedule-Aware & AdamW Mutations

## Critical Discovery: Production vs Evaluation Gap

**Production NanoChat (base_train.py:373-381) already has THREE schedules our eval loop LACKS:**
```python
# 1. Momentum warmup 0.85→0.95 over 300 steps
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# 2. Weight decay linear decay to 0
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)

# 3. LR: linear warmup → constant → linear warmdown (last 50%)
def get_lr_multiplier(it):
    ...  # warmup_ratio=0.0, warmdown_ratio=0.5, final_lr_frac=0.0
```

**Our eval loop (slurm scripts) only has cosine LR schedule.** No momentum warmup, no WD decay.
This means H41's Stage 3 win (+0.032 BPB) was partly because it added something production already has.

**Implication:** Stage 4 must first establish a CORRECT baseline that includes all production schedules, then mutate from there. Otherwise we're measuring gap-filling, not actual improvement.

## Two Attack Directions

### Direction A: Programmatic Schedules (Annealing)

**What is already scheduled in production:**
- LR: linear warmdown over last 50% to 0 (NOT cosine)
- Momentum (Muon only): linear warmup 0.85→0.95 over 300 steps
- Weight decay (Muon only): linear decay to 0 over full training

**What is NOT scheduled (fixed constants):**
- AdamW beta1=0.8 (all AdamW groups) — NEVER changes
- AdamW beta2=0.95 (all AdamW groups) — NEVER changes
- AdamW eps=1e-10 — NEVER changes
- Second moment beta2=0.95 (NorMuon variance reduction) — NEVER changes
- Trust ratio clamp bounds [0.5, 1.5] — NEVER changes
- Orthogonalization mix=1.0 — NEVER changes
- NS iterations=5 — NEVER changes
- Nesterov blend coefficient = momentum (coupled) — NEVER changes independently

**Attack surfaces for scheduling:**

1. **Second moment beta2 schedule** — Production uses fixed 0.95. Early training has high variance; late training is smoother. A warmup from 0.8→0.95 or decay from 0.95→0.99 could help.

2. **Trust ratio clamp schedule** — Wider clamps early (more adaptive), tighter late (more stable). E.g., [0.3, 2.0] → [0.5, 1.5] over first 500 steps.

3. **Weight decay schedule** — Production linearly decays to 0. Could try: cosine decay, sqrt decay, or step-function (full WD for first 80%, then zero).

4. **Momentum overshoot then settle** — Production warms 0.85→0.95. What about 0.85→0.97→0.95 (overshoot then cool)?

5. **LR schedule shape** — Our eval uses cosine. Production uses linear warmdown. Which is actually better? Test both.

6. **Nesterov blend schedule** — Decouple from momentum beta. H46 used fixed 0.8. What about scheduled: 0.7→0.9 (less aggressive early, more late)?

7. **Orthogonal mix schedule** — Start with partial orthogonalization, increase to full. E.g., 0.5→1.0 over first 200 steps. (Easier optimization landscape early.)

### Direction B: AdamW Path (Untouched Territory)

**The AdamW path has NEVER been mutated.** It handles:
- `lm_head` (unembedding): lr=0.004, betas=(0.8, 0.95), wd=0.0
- `embedding` (wte): lr=0.2, betas=(0.8, 0.95), wd=0.0
- `value_embeds`: lr=0.2, betas=(0.8, 0.95), wd=0.0
- `block_non_matrix` (LayerNorm): lr=0.005, betas=(0.8, 0.95), wd=0.0
- `resid_scalars` (resid_lambdas): lr=0.005, betas=(0.8, 0.95), wd=0.0
- `x0_scalars` (x0_lambdas): lr=0.5, betas=(0.96, 0.95), wd=0.0

**Key observation:** ALL AdamW groups have weight_decay=0.0 and fixed betas. No schedules at all.

**Attack surfaces:**

1. **AdamW beta1 warmup** — Production uses fixed 0.8. H41 showed momentum warmup helps Muon. Does beta1 warmup help AdamW? E.g., 0.7→0.8 over 200 steps for embeddings.

2. **Embedding weight decay** — All embedding groups have wd=0.0. Small WD (0.01-0.05) on embeddings is common in other frameworks. Could help regularization.

3. **lm_head / embedding LR coupling** — lm_head (0.004) and embedding (0.2) have 50x LR ratio. Could test tied LR or different ratios.

4. **AdamW eps schedule** — eps=1e-10 is very small. Early training may benefit from larger eps (1e-8) for stability, decaying to 1e-10.

5. **x0_scalars beta1** — Currently 0.96, different from all other groups (0.8). Why? Could test aligning it.

6. **Embedding gradient scaling** — Embeddings have sparse gradients (only active tokens get updates). Could normalize by active token count.

7. **Separate LR warmup for embeddings** — Embeddings might benefit from slower warmup since they're high-dimensional lookup tables.

## Code Reference: _step_adamw (candidate_optimizer.py:239-264)

```python
def _step_adamw(self, group: dict[str, object]) -> float:
    max_grad_norm = 0.0
    for param in group["params"]:
        if param.grad is None:
            continue
        state = self.state[param]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
        state["step"] += 1
        grad = param.grad
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        beta1, beta2 = group["betas"]
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2)
        bias_correction1 = 1 - beta1 ** state["step"]
        bias_correction2 = 1 - beta2 ** state["step"]
        denom = exp_avg_sq.div(bias_correction2).sqrt().add_(group["eps"])
        step_size = group["lr"] / bias_correction1
        if group["weight_decay"]:
            param.mul_(1 - group["lr"] * group["weight_decay"])
        param.addcdiv_(exp_avg, denom, value=-step_size)
        max_grad_norm = max(max_grad_norm, float(grad.float().norm().item()))
    return max_grad_norm
```

Note: This AdamW already has bias correction (unlike the Muon path). The mutation target is the hyperparameters and schedules, not the algorithm itself.

## Code Reference: Production training loop schedule application (base_train.py:515-522)

```python
lrm = get_lr_multiplier(step)
muon_momentum = get_muon_momentum(step)
muon_weight_decay = get_weight_decay(step)
for group in optimizer.param_groups:
    group["lr"] = group["initial_lr"] * lrm
    if group['kind'] == 'muon':
        group["momentum"] = muon_momentum
        group["weight_decay"] = muon_weight_decay
```

**Key:** Only Muon groups get momentum/WD schedules. AdamW groups only get LR schedule.

## Code Reference: Production param group setup (gpt.py:356-394)

```python
# dmodel_lr_scale for muP extrapolation
dmodel_lr_scale = (model.config.n_embd / 768) ** -0.5

# AdamW groups
param_groups = [
    dict(kind='adamw', name='lm_head', params=..., lr=unembedding_lr * dmodel_lr_scale,
         betas=(adam_beta1, adam_beta2), eps=1e-10, weight_decay=0.0),
    dict(kind='adamw', name='wte', params=..., lr=embedding_lr * dmodel_lr_scale, ...),
    dict(kind='adamw', name='value_embeds', params=..., lr=embedding_lr * dmodel_lr_scale, ...),
    dict(kind='adamw', name='resid_lambdas', params=..., lr=scalar_lr * 0.01, ...),
    dict(kind='adamw', name='x0_lambdas', params=..., lr=scalar_lr, betas=(0.96, adam_beta2), ...),
]

# Muon groups (by shape)
for shape in sorted({p.shape for p in matrix_params}):
    group_params = [p for p in matrix_params if p.shape == shape]
    param_groups.append(dict(
        kind='muon', params=group_params, lr=matrix_lr,
        momentum=0.95, ns_steps=5, beta2=0.95,
        weight_decay=weight_decay_scaled,
    ))
```

## What we know works (all stages)

| ID | Mutation | BPB Δ | Horizon | Status |
|----|----------|-------|---------|--------|
| H02 | Layerwise trust ratio [0.5, 1.5] | +0.006 | 1000 | Deployed |
| H41 | Momentum warmup 0.85→0.95/200 steps | +0.032 | 5000 | **Best ever** |
| H39 | Per-neuron trust (aspect-aware) | +0.014 | 5000 | Stable |
| H42 | Second moment bias correction | +0.007 | 1000 | Untested 5k |
| H54 | Raw gradient WD alignment | +0.007 | 1000 | Untested 5k |

## What DOES NOT work (negative knowledge)

| ID | What | Why |
|----|------|-----|
| NK01 | Post-orthogonal momentum | -0.031. Pipeline order is sacred. |
| NK02 | ns_steps=3 | -0.013. Quality matters. |
| NK03 | Remove norm-preserving rescale | -0.049. Worst mutation ever. |
| NK12 | Post-ortho scale preservation | -0.066. Polar Express distortion is beneficial. |
| NK13 | Compounding top 3 mutations | Compound < best single. Interactions cancel. |
| NK14 | H41 + H46 together | -0.010. Same-method mutations interfere. |

## Meta-lessons

1. **Single clean mutations beat compounds** — always test individually first.
2. **Rankings shift with horizon** — must test at 5x the screening horizon.
3. **Momentum is highest leverage** — H41 (+0.032) > all other mutations combined.
4. **Production already has schedules we were missing** — H41 partially rediscovered momentum warmup.
5. **Deep hypothesis generation with coverage analysis >> random exploration**.
6. **The AdamW path is completely unexplored** — 100% of evolution has been on Muon matrices.

## Architectural constraints

- `_step_adamw` runs per-parameter (not stacked). Mutations must work per-param.
- `_step_matrix_group` runs stacked by shape. Schedule mutations can use `self._step_context`.
- `set_step_context(loss_value, step, total_steps)` is called before every `optimizer.step()`.
- External LR schedule multiplies `group['lr']` before `step()`. Internal schedules must not conflict.
- The training loop also sets `group['momentum']` and `group['weight_decay']` for Muon groups in production. Our eval loop currently does NOT do this — only LR.

## File references

- Candidate optimizer: `adamopt/optim_search/candidate_optimizer.py`
- Production Muon+AdamW: `nanochat/nanochat/optim.py`
- Production training loop: `nanochat/scripts/base_train.py` (schedules at lines 362-381, application at 515-522)
- Model setup_optimizer: `nanochat/nanochat/gpt.py` (lines 356-394)
- DSL spec: `adamopt/optim_search/spec.py`
- Stage 3 runner: `enigma/run_stage3.py`
- Stage 3 postmortem: `runs/enigma_stage3_postmortem.json`

## IMPORTANT: Eval loop gap to fix FIRST

Before any Stage 4 mutations, the eval loop MUST match production schedules:
1. Add momentum warmup 0.85→0.95 over 300 steps (match production, not H41's 200)
2. Add weight decay linear decay to 0
3. Consider switching from cosine LR to linear warmdown (match production)

Then measure a **corrected baseline**. H41's +0.032 will shrink or disappear since production already does this.
The TRUE Stage 4 baseline should include all production schedules.
