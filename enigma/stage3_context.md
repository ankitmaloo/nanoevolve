# Stage 3 Context — Code-Level Optimizer Mutation Hypothesis Generation

## Goal
Generate code-level mutations to `SpecCandidateOptimizer` (adamopt/optim_search/candidate_optimizer.py)
that improve BPB on real NanoChat GPT-2 training (12 layers, 768 dim, ClimbMix-400B dataset, B200 GPU).

The ONLY mutation that survived 1000-step real GPT-2 validation is H02: adding layerwise trust ratio with clamp [0.5, 1.5].
Everything else (including all Stage 2 code-level mutations) either faded or reversed at long horizon.
We need BETTER code-level hypotheses.

## What the optimizer does (pipeline order in `_step_matrix_group`)

```
grads -> momentum (Nesterov, beta=0.95) -> orthogonalize (Polar Express, 5 iters, float32)
       -> second_moment (factored RMS, norm-preserving rescale, beta2=0.95)
       -> trust_ratio (layerwise param_norm/update_norm, clamp [0.5, 1.5])
       -> clip (update_rms, threshold=1.0)
       -> scale by update_multiplier
       -> lr_aspect_scale (sqrt(rows/cols))
       -> cautious weight decay (only where update aligns with param, wd=0.2)
       -> copy back to params
```

## Production Muon (nanochat/nanochat/optim.py) — what we're trying to beat

```
grads -> Nesterov momentum (beta=0.95) -> Polar Express (5 iters, BFLOAT16)
       -> NorMuon variance reduction (factored RMS, norm-preserving rescale, beta2=0.95)
       -> cautious weight decay (wd=0.2)
       -> copy back
```

Key: Production is `@torch.compile(dynamic=False, fullgraph=True)` fused into a single kernel.
Production does NOT have: trust_ratio, clip, stateful control, update_multiplier.
Production DOES use: bfloat16 for Polar Express (candidate uses float32).

## Candidate optimizer code — the mutation target

### `_apply_momentum` (line 266-270)
```python
def _apply_momentum(self, update: Tensor, momentum_buffer: Tensor) -> Tensor:
    momentum_buffer.lerp_(update, 1 - self.spec.momentum)  # EMA update
    if self.spec.nesterov:
        return update.lerp(momentum_buffer, self.spec.momentum)  # Nesterov lookahead
    return momentum_buffer
```

### `_orthogonalize` (line 272-290)
```python
def _orthogonalize(self, update: Tensor, *, orthogonal_mix: float = 1.0) -> Tensor:
    if self.spec.orthogonalization == "none":
        return update
    raw_update = update
    x = update.to(dtype=torch.float32)  # DIFFERS: production uses bfloat16
    x = x / (x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6) * 1.02)
    for a, b, c in POLAR_EXPRESS_COEFFS[:self.spec.ns_steps]:
        if x.size(-2) > x.size(-1):  # tall matrix
            gram = x.mT @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
        else:  # wide matrix
            gram = x @ x.mT
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    orthogonalized = x.to(dtype=update.dtype)
    if orthogonal_mix >= 1.0:
        return orthogonalized
    if orthogonal_mix <= 0.0:
        return raw_update
    return orthogonal_mix * orthogonalized + (1.0 - orthogonal_mix) * raw_update
```

### `_apply_second_moment` (line 292-305) — NorMuon variance reduction
```python
def _apply_second_moment(self, update: Tensor, second_moment_buffer: Tensor, *, beta2_override=None) -> Tensor:
    if self.spec.second_moment.mode == "none":
        return update
    red_dim = -1 if update.size(-2) >= update.size(-1) else -2
    v_mean = update.float().square().mean(dim=red_dim, keepdim=True)
    beta2 = self.spec.second_moment.beta2 if beta2_override is None else beta2_override
    second_moment_buffer.lerp_(v_mean.to(dtype=second_moment_buffer.dtype), 1 - beta2)
    step_size = second_moment_buffer.clamp_min(self.spec.second_moment.eps).rsqrt()
    # Norm-preserving rescale
    red_dim_size = update.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(self.spec.second_moment.eps)
    final_scale = step_size * (v_norm / v_norm_new)
    return update * final_scale.to(dtype=update.dtype)
```

### `_apply_trust_ratio` (line 307-316) — THE WINNING MUTATION
```python
def _apply_trust_ratio(self, update: Tensor, params: Tensor, *, trust_mix: float = 1.0) -> tuple[Tensor, Tensor]:
    if self.spec.trust_ratio.mode == "none":
        return update, ones
    param_norm = params.float().flatten(1).norm(dim=1).clamp_min(eps)
    update_norm = update.float().flatten(1).norm(dim=1).clamp_min(eps)
    trust = (param_norm / update_norm).clamp(clamp_min, clamp_max)
    trust = 1.0 + trust_mix * (trust - 1.0)
    reshape = trust.view(-1, *([1] * (update.ndim - 1)))
    return update * reshape.to(dtype=update.dtype), trust
```
This computes ONE trust ratio per matrix (whole-layer Frobenius norm ratio).

### `_step_matrix_group` pipeline (line 332-399) — the full step
```python
# Shape-group all same-shape matrix params, stack into single tensor
grads = torch.stack([p.grad.detach() for p in params])
stacked_params = torch.stack([p.detach() for p in params])

# Gate + actuators (if stateful control enabled)
gate = self._gate_value()

# Momentum + orthogonalization
if pre_orthogonal:
    update = _apply_momentum(grads, momentum_buffer)
    update = _orthogonalize(update, orthogonal_mix=...)
else:
    update = _orthogonalize(grads, ...)
    update = _apply_momentum(update, momentum_buffer)

# Post-processing pipeline
update = _apply_second_moment(update, second_moment_buffer, ...)
update, trust = _apply_trust_ratio(update, stacked_params, ...)
update = _apply_clip(update, ...)
update *= update_multiplier

# Weight decay + param update
lr *= sqrt(max(1, rows/cols))  # aspect ratio scaling
if cautious:
    mask = (update * stacked_params) >= 0  # alignment mask
    stacked_params -= lr * update + lr * wd * stacked_params * mask
elif decoupled:
    stacked_params *= (1 - lr * wd)
    stacked_params += update * (-lr)
```

### `_step_adamw` (line 239-264) — for embeddings, scalars, lm_head
Standard AdamW. NOT a mutation target (embeddings/scalars stay on AdamW by design).

## What we know works (positive knowledge)
- Trust ratio (layerwise): +0.0016 BPB at 1000 steps. Simple, robust, no overhead.
- Pre-orthogonal momentum: critical, must not change.
- 5 Newton-Schulz iterations: critical, must not reduce.
- Norm-preserving rescale in second_moment: critical at depth, must not remove.
- Cautious weight decay: production default, works.

## What we know DOES NOT work (negative knowledge — DO NOT REPEAT)
- NK01: Post-orthogonal momentum: -0.031 BPB. Catastrophic.
- NK02: ns_steps=3: -0.013 BPB. Quality matters.
- NK03: Removing norm-preserving rescale: -0.049 BPB. Worst mutation.
- NK04: Disabling second_moment entirely: +0.011 at 200 steps but -0.005 at 1000 steps. Short-horizon illusion.
- NK05: bfloat16 Polar Express: +0.001 BPB but not significant at 1000 steps. Speed-only.
- NK06: Compounding trust+clip+stateful gate: no better than trust alone at 1000 steps.
- NK07: Stateful gate alone: loses.
- NK08: Wider trust clamp [0.1, 8.0]: neutral/loses (narrow [0.5, 1.5] is better).
- NK09: Reordering trust before second_moment: neutral.
- NK10: Trust-gated cautious WD: loses.
- NK11: Amplified gate annealing: loses.

## Dimensions NOT yet explored (gaps from Stage 2)

1. **Per-neuron trust ratio** instead of per-layer (finer granularity)
2. **Momentum schedule** — beta varies over training (warmup from 0.8 to 0.95)
3. **Adaptive weight decay schedule** — WD annealing or cosine
4. **Gradient noise injection** — regularization during orthogonalization
5. **Second moment bias correction** — production doesn't do it, candidate doesn't either
6. **Separate beta2 per shape group** — different variance reduction rates for attention vs MLP
7. **Orthogonalization warmup** — fewer NS iterations early, more later (or vice versa)
8. **Trust ratio with momentum** — use momentum buffer norm instead of raw param norm
9. **Nesterov interpolation coefficient** — the `update.lerp(buf, momentum)` uses the same momentum for both EMA and Nesterov blending
10. **Gradient accumulation awareness** — optimizer doesn't know about grad_accum_steps
11. **Per-shape-group learning rate multiplier** — different LR scales for attention Q/K/V/O vs MLP
12. **Update direction filtering** — project out directions that have been consistently harmful
13. **Cautious WD alignment threshold** — current is (update * params) >= 0, could be a softer threshold
14. **EMA of orthogonalized updates** — track running statistics of the post-ortho update for better trust calibration

## Key architectural constraints
- All matrix params are grouped by shape and stacked into (N, rows, cols) tensors
- Trust ratio operates per-matrix (dim 0 of the stack), NOT per-neuron
- Second moment is factored: either (N, rows, 1) or (N, 1, cols) depending on aspect ratio
- Cautious WD uses element-wise alignment, not per-matrix
- The whole pipeline runs under `@torch.no_grad()`
- LR aspect scaling: lr *= sqrt(max(1, rows/cols)) — only scales up for tall matrices
- Production Muon mutates `stacked_grads` in-place for the momentum step (`g = stacked_grads.lerp_(...)`)

## DSL spec (frozen dataclass)
- MatrixOptimizerSpec: name, momentum, nesterov, momentum_placement, orthogonalization,
  ns_steps, trust_ratio, clip, decay, second_moment, stateful_control, update_multiplier,
  lr_aspect_scale, matrix_only, metadata
- TrustRatioConfig: mode (none|layerwise), clamp_min, clamp_max, eps
- ClipConfig: mode (none|update_rms|global_norm), threshold
- DecayConfig: mode (none|decoupled|cautious), weight_decay
- SecondMomentConfig: mode (none|factored_rms), beta2, eps
- StatefulControlConfig: enabled, ema_beta, gate (GateConfig), actuators (AdaptiveActuatorConfig)

## File references
- Candidate optimizer: adamopt/optim_search/candidate_optimizer.py
- Production Muon: nanochat/nanochat/optim.py
- DSL spec: adamopt/optim_search/spec.py
- Stage 2 runner: enigma/run_stage2.py (Stage2Optimizer subclass pattern)
- Postmortem: runs/enigma_stage2_postmortem.json
- Stage 2 results (200-step GPT-2): runs/enigma_gpt2_full/results/
- Stage 2 results (1000-step GPT-2): runs/enigma_top3_1k/results/

## Important: what "code-level mutation" means
Override a method in Stage2Optimizer (subclass of SpecCandidateOptimizer).
The method signature and the rest of the pipeline stay the same.
You change the INTERNALS of one method — the math, the dtype, the reduction strategy, etc.
Example: override `_apply_trust_ratio` to compute per-neuron trust instead of per-layer.
Example: override `_apply_momentum` to use a warmup schedule on the momentum coefficient.
Example: override `_step_matrix_group` to change the pipeline ordering or add a new post-processing step.
