# Stage 4 Hypotheses: Cross-Path Interactions & Eval-Production Parity

---

## H80 — Corrected Baseline: Production Schedule Parity

**What it changes:** The eval loop currently applies only cosine LR. This adds all three production schedules: (1) Muon momentum warmup 0.85->0.95 over 300 steps, (2) Muon WD linear decay to 0, (3) linear warmup + constant + linear warmdown LR (replacing cosine).

**Exact implementation:**
In `eval_candidate.py`, inside the training loop (after `optimizer.set_step_context(...)`, before `optimizer.step()`), add schedule application matching `base_train.py:515-522`:

```python
# --- Production schedule parity ---
warmup_ratio = 0.0  # production default
warmdown_ratio = 0.5
final_lr_frac = 0.0
total = self.eval_config.steps

# LR multiplier: linear warmup -> constant -> linear warmdown
warmup_iters = round(warmup_ratio * total)
warmdown_iters = round(warmdown_ratio * total)
if warmup_iters > 0 and step < warmup_iters:
    lrm = (step + 1) / warmup_iters
elif step <= total - warmdown_iters:
    lrm = 1.0
else:
    progress = (total - step) / warmdown_iters
    lrm = progress * 1.0 + (1 - progress) * final_lr_frac

# Muon momentum warmup: 0.85 -> 0.95 over 300 steps
muon_momentum = (1 - min(step / 300, 1)) * 0.85 + min(step / 300, 1) * 0.95

# Muon WD decay: linear to 0
weight_decay_base = 0.2  # from spec
muon_wd = weight_decay_base * (1 - step / total)

for group in optimizer.param_groups:
    group["lr"] = group.get("initial_lr", group["lr"]) * lrm
    if group["kind"] == "matrix_candidate":
        # Muon-equivalent groups: apply momentum and WD schedules
        # Momentum is baked into spec.momentum; override via _step_context or group
        pass  # See note below on how momentum/WD flow through SpecCandidateOptimizer
```

**Critical subtlety:** `SpecCandidateOptimizer` reads `self.spec.momentum` (frozen dataclass), not `group["momentum"]`. The corrected baseline must either: (a) store `initial_lr` on group init and apply `lrm` externally, handling momentum/WD via `_step_context` additions, or (b) subclass and override `_apply_momentum` to read scheduled momentum from `_step_context`. Option (b) is cleaner since it keeps the spec frozen while allowing runtime schedule overrides.

Add to `_step_context`: `"muon_momentum": float`, `"muon_wd": float`. Then in `_apply_momentum`, use `self._step_context.get("muon_momentum", self.spec.momentum)` instead of `self.spec.momentum`. Similarly for WD in `_step_matrix_group`.

**Mechanistic reasoning:** H41 got +0.032 BPB partly by adding momentum warmup that production already has. The corrected baseline reveals the TRUE improvement ceiling by closing the eval-production gap. Without this, every subsequent hypothesis is measured against a handicapped baseline, inflating deltas.

**Risk level:** LOW. This is not a mutation; it is a measurement correction. The only risk is implementation bugs.

**Dependencies:** None. This IS the foundation.

---

## H81 — AdamW Beta1 Warmup Synchronized with Muon Momentum

**What it changes:** AdamW groups use fixed beta1=0.8. Production warms Muon momentum 0.85->0.95 over 300 steps. This hypothesis warms AdamW beta1 in sync: 0.7->0.8 over the same 300 steps.

**Exact implementation:**
In `_step_adamw`, replace `beta1, beta2 = group["betas"]` with:

```python
beta1_base, beta2 = group["betas"]
step_num = self._step_context.get("step", 1)
warmup_steps = 300
frac = min(step_num / warmup_steps, 1.0)
beta1 = (1 - frac) * (beta1_base - 0.1) + frac * beta1_base  # 0.7 -> 0.8
```

Everything else stays the same (exp_avg.lerp_, bias_correction, etc.).

**Mechanistic reasoning:** Early training has high-variance gradients for ALL parameter types, not just matrices. If Muon benefits from lower initial momentum (less reliance on noisy history), AdamW's first moment should too. The 0.7->0.8 range is conservative (smaller delta than Muon's 0.85->0.95) because AdamW already has bias correction which partially compensates for early noise, while Muon does not.

**Risk level:** LOW. Small perturbation to a well-understood hyperparameter. Worst case: negligible change.

**Dependencies:** H80 (corrected baseline must include production Muon momentum warmup, otherwise we conflate two effects).

---

## H82 — Embedding LR Warmup (Slower Than Matrix LR)

**What it changes:** Embedding LR follows the same schedule as all other groups (production: instant warmup since warmup_ratio=0.0). This adds a 100-step embedding-specific linear warmup from 0.1x to 1.0x of the group LR.

**Exact implementation:**
In the schedule application code (from H80), after computing `lrm`, add:

```python
for group in optimizer.param_groups:
    base_lrm = lrm
    if group.get("group_name") in ("embedding", "value_embeds"):
        embed_warmup = 100
        embed_frac = min(step / embed_warmup, 1.0)
        embed_scale = 0.1 + 0.9 * embed_frac
        base_lrm *= embed_scale
    group["lr"] = group["initial_lr"] * base_lrm
```

**Mechanistic reasoning:** Embeddings are lookup tables with sparse gradients (only active tokens get updates). Early in training, the token distribution seen so far is narrow, so embedding gradients are biased toward high-frequency tokens. High LR on biased early gradients pushes embeddings into poor initial geometry that takes many steps to recover from. Muon matrices don't have this problem because orthogonalization normalizes update direction regardless of gradient sparsity. A slower embedding warmup lets the token distribution diversify before committing to large embedding updates.

**Risk level:** LOW-MEDIUM. Embedding LR is already very high (0.2) relative to other groups. Slowing initial updates is conservative.

**Dependencies:** H80 (need corrected LR schedule as baseline).

---

## H83 — Coordinated WD Decay: Embedding Regularization Tied to Muon WD

**What it changes:** All AdamW groups have weight_decay=0.0 (never changes). This adds small WD to embedding and lm_head groups, decaying in sync with Muon WD: `wd = 0.02 * (1 - step/total_steps)`.

**Exact implementation:**
In `_step_adamw`, after the existing WD block, add:

```python
# Coordinated embedding WD (only for embedding-like groups)
group_name = group.get("group_name", "")
if group_name in ("embedding", "value_embeds", "lm_head"):
    step_num = self._step_context.get("step", 1)
    total = self._step_context.get("total_steps", 1)
    emb_wd = 0.02 * (1 - step_num / total)
    if emb_wd > 0:
        param.mul_(1 - group["lr"] * emb_wd)
```

**Mechanistic reasoning:** Muon's WD decay means matrix parameters gradually become less regularized as training progresses. If embeddings have zero WD throughout, there is a growing regularization mismatch: late in training, matrices are free to grow while embeddings are already unconstrained. This can cause the embedding-to-matrix interface to become poorly conditioned. Small coordinated WD on embeddings keeps the regularization pressure ratio stable across the Muon/AdamW boundary. The 0.02 value is 10x smaller than Muon's 0.2 because embeddings are not orthogonalized and have no norm-preserving rescale.

**Risk level:** MEDIUM. Adding WD to embeddings that previously had none changes the loss landscape. Could help or hurt.

**Dependencies:** H80 (need Muon WD decay in eval to observe the coordination effect).

---

## H84 — Loss-Gated Orthogonal Mix: Reduce Orthogonalization When Loss Plateaus

**What it changes:** Orthogonal_mix is fixed at 1.0. This makes it reactive to loss: when loss improvement stalls (loss_improvement_ema < threshold), reduce ortho_mix to 0.7 to allow more gradient-aligned updates.

**Exact implementation:**
Modify `_step_matrix_group` to compute a loss-aware ortho_mix:

```python
# Before existing orthogonal_mix computation:
if not self._stateful_enabled():
    # Lightweight loss-gated ortho (no full stateful control needed)
    loss_imp = self.training_signals.loss_improvement_ema
    if self.training_signals.initialized and loss_imp < 0.001:
        orthogonal_mix = 0.7  # partial ortho when plateauing
    else:
        orthogonal_mix = 1.0
```

Alternatively, use the existing stateful control machinery by enabling it with a minimal config:
```python
StatefulControlConfig(
    enabled=True,
    gate=GateConfig(
        coefficients=GateCoefficients(loss_improvement_ema=-2.0),  # low improvement -> high gate
        bias=0.0, sharpness=1.5,
    ),
    actuators=AdaptiveActuatorConfig(
        orthogonal_mix=AdaptiveRange(aggressive=1.0, conservative=0.7),
        # All other actuators set to neutral (aggressive=conservative)
        update_multiplier=AdaptiveRange(aggressive=1.0, conservative=1.0),
        trust_ratio_mix=AdaptiveRange(aggressive=1.0, conservative=1.0),
        clip_threshold=AdaptiveRange(aggressive=1.0, conservative=1.0),
        beta2=AdaptiveRange(aggressive=0.95, conservative=0.95),
    ),
)
```

**Mechanistic reasoning:** Full orthogonalization projects updates onto the Stiefel manifold, which is excellent for feature learning (diverse gradient directions map to orthogonal weight updates, maximizing representational capacity). But during loss plateaus, the model needs to fine-tune existing features, not discover new ones. Partial orthogonalization (mix=0.7) lets the raw gradient direction leak through, enabling subtle parameter adjustments that full ortho would project away. The AdamW path is unaffected, so embeddings continue fine-tuning normally — the asymmetry is intentional.

**Risk level:** MEDIUM. NK03 showed removing ortho rescaling is catastrophic (-0.049), but that was removing the rescale, not reducing ortho_mix. Partial ortho at 0.7 preserves the manifold structure while adding flexibility. Still, the plateau detector (loss_improvement_ema < 0.001) needs tuning.

**Dependencies:** H80 (the LR schedule shape affects when plateaus appear; cosine vs linear warmdown creates different plateau patterns).

---

## H85 — Phase-Aware Trust Ratio: Widen Early, Tighten Late

**What it changes:** Trust ratio clamps are fixed at [0.5, 1.5]. This schedules them: [0.3, 2.0] for the first 20% of training, linearly transitioning to [0.6, 1.3] for the remaining 80%.

**Exact implementation:**
In `_apply_trust_ratio` (or overridden version), replace fixed clamps:

```python
step_frac = self._step_context.get("step", 1) / self._step_context.get("total_steps", 1)
transition_point = 0.2
if step_frac < transition_point:
    t = step_frac / transition_point
    clamp_min = 0.3 + t * (0.6 - 0.3)
    clamp_max = 2.0 + t * (1.3 - 2.0)
else:
    clamp_min = 0.6
    clamp_max = 1.3

trust = (param_norm / update_norm).clamp(clamp_min, clamp_max)
```

**Mechanistic reasoning:** Early training has high gradient variance; updates for different matrix layers have wildly different norms relative to their parameters. Wide trust clamps let the optimizer adapt per-layer learning rates aggressively, which is critical when some layers initialize far from their final operating point. Late in training, all layers are in the same ballpark, and tight clamps prevent any single layer from overshooting during fine-tuning. This interacts with the AdamW path indirectly: because trust ratios modulate matrix update magnitudes, they affect the gradient signal that AdamW-optimized embeddings and lm_head see at the next forward pass. Tighter late-stage trust means more predictable gradient flow through the network, which stabilizes AdamW's accumulation.

**Risk level:** LOW-MEDIUM. This is a schedule over proven-beneficial trust ratios. The [0.3, 2.0] early range is wider than tested but bounded. The [0.6, 1.3] late range is tighter than the H02 winner [0.5, 1.5] but within NK-safe territory.

**Dependencies:** H80 (trust ratio interacts with WD decay and momentum warmup; all must be present for realistic phase behavior).

---

## H86 — Muon Update Norm Feedback to AdamW LR

**What it changes:** AdamW groups use a fixed LR multiplier. This scales AdamW LR by the inverse of Muon's mean update-to-parameter ratio, smoothed by EMA. When Muon makes large updates (high ratio), AdamW LR decreases; when Muon makes small updates, AdamW LR increases.

**Exact implementation:**
Add to `SpecCandidateOptimizer`:

```python
def __init__(self, ...):
    ...
    self._muon_update_ratio_ema = 1.0  # normalized to 1.0

# In step(), after matrix groups are processed:
if matrix_stats:
    raw_ratio = sum(s.mean_update_param_ratio for s in matrix_stats) / len(matrix_stats)
    self._muon_update_ratio_ema = 0.95 * self._muon_update_ratio_ema + 0.05 * raw_ratio

# In _step_adamw, scale the effective LR:
adamw_lr_scale = 1.0 / max(0.5, min(2.0, self._muon_update_ratio_ema / self._muon_update_ratio_baseline))
step_size = group["lr"] * adamw_lr_scale / bias_correction1
```

Where `_muon_update_ratio_baseline` is the ratio at step 1 (or a fixed expected value like 0.01).

**Mechanistic reasoning:** Muon's orthogonalized updates change the loss landscape that embedding and lm_head parameters see. When Muon makes a large step (moving matrices far from their previous position), the gradient signal at the next step will be very different from what AdamW's momentum buffers predict. Reducing AdamW's LR when Muon is volatile prevents AdamW from overshooting on stale momentum estimates. Conversely, when Muon is making small updates (late training), AdamW can safely take larger steps to continue improving. This creates implicit coordination without explicit schedule coupling.

**Risk level:** MEDIUM-HIGH. This is a novel feedback mechanism between optimizer paths. The clamping [0.5, 2.0] prevents catastrophic scaling, but the normalization baseline needs careful tuning. If the EMA lags too much, the scaling becomes anti-correlated with the actual state.

**Dependencies:** H80 (Muon update ratios change significantly with momentum warmup and WD decay).

---

## H87 — Loss-Reactive Second Moment Beta2 Schedule

**What it changes:** Second moment beta2 is fixed at 0.95. This makes it loss-reactive: when loss is decreasing rapidly (early training), use lower beta2=0.85 (more responsive); when loss plateaus, use higher beta2=0.99 (more stable).

**Exact implementation:**
In `_step_matrix_group`, compute dynamic beta2:

```python
# After self._update_loss_signals() has run (in step()):
loss_imp = self.training_signals.loss_improvement_ema
if self.training_signals.initialized:
    # Normalize improvement to [0, 1] range
    imp_signal = max(0, min(1, loss_imp / 0.05))  # 0.05 = expected max improvement
    # Map: high improvement -> low beta2, low improvement -> high beta2
    beta2_dynamic = 0.85 + (0.99 - 0.85) * (1 - imp_signal)
else:
    beta2_dynamic = 0.85  # start responsive

# Pass as override:
update = self._apply_second_moment(update, second_moment_buffer, beta2_override=beta2_dynamic)
```

**Mechanistic reasoning:** The second moment in NorMuon acts as a per-neuron adaptive learning rate. During rapid loss decrease, gradient statistics are non-stationary — a high beta2 averages over outdated variance estimates, causing step sizes to lag behind the actual gradient scale. Lower beta2 during rapid learning lets the variance tracker keep up. During plateaus, gradient statistics stabilize and a higher beta2 produces smoother, more reliable step sizes that prevent oscillation. This mirrors the intuition behind the "learning rate warmup for Adam" paper (Ma & Yarats, 2021) but applied to the variance tracker rather than the LR itself.

**Risk level:** MEDIUM. The beta2 range [0.85, 0.99] is wider than tested. The loss_improvement_ema signal is already computed and available. Risk is primarily in the normalization constant (0.05) being wrong for longer runs.

**Dependencies:** H80 (loss dynamics change significantly with proper LR schedule).

---

## H88 — Phase Transition Detector: Switch Decay Mode at Inflection

**What it changes:** The spec uses a single decay mode (cautious) throughout training. This detects the feature-learning-to-refinement phase transition (loss curve inflection point) and switches from cautious WD to decoupled WD at that point.

**Exact implementation:**
Add a phase detector to the optimizer:

```python
# In SpecCandidateOptimizer.__init__:
self._loss_history = []
self._phase = "feature_learning"  # or "refinement"
self._inflection_step = None

# In _update_loss_signals, after EMA update:
self._loss_history.append(self._step_context["loss"])
if len(self._loss_history) >= 50 and self._inflection_step is None:
    # Compute second derivative of smoothed loss
    window = 20
    recent = self._loss_history[-window:]
    deltas = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
    # First derivative of deltas (second derivative of loss)
    second_deriv = [deltas[i+1] - deltas[i] for i in range(len(deltas)-1)]
    avg_2nd = sum(second_deriv) / len(second_deriv)
    # Inflection: second derivative goes from negative to near-zero
    if avg_2nd > -1e-5:
        self._phase = "refinement"
        self._inflection_step = self._step_context["step"]

# In _step_matrix_group, replace decay block:
if self._phase == "feature_learning":
    # Cautious: only decay when update and param agree
    mask = (update * stacked_params) >= 0
    stacked_params.sub_(lr * update + lr * wd * stacked_params * mask)
else:
    # Decoupled: uniform decay, simpler dynamics for refinement
    stacked_params.mul_(1 - lr * wd)
    stacked_params.add_(update, alpha=-lr)
```

**Mechanistic reasoning:** Cautious WD is beneficial during feature learning because it avoids shrinking parameters that the optimizer is actively growing (update and param have opposite signs = the feature is being restructured). But during refinement, features are established and the dominant failure mode is overfitting, not undergrowth. Decoupled WD provides uniform regularization that is more appropriate for fine-tuning. The phase transition typically happens when the loss curve's second derivative crosses zero (going from concave to linear/convex). Cautious WD during refinement can actually be harmful because it selectively preserves large-magnitude parameters, creating an implicit bias toward "confident" features even when the model should be distributing capacity more evenly.

**Risk level:** MEDIUM-HIGH. The inflection point detector is noisy on short runs. The mode switch is discontinuous — could cause a loss spike at the transition. A soft blend over 50 steps (cautious_weight * (1-t) + decoupled_weight * t) would be safer.

**Dependencies:** H80 (WD decay schedule affects when and whether the inflection point appears; also, without WD decay, "switching from cautious to decoupled" is less meaningful because WD magnitude is constant).

---

## H89 — Cross-Path Gradient Norm Balancing

**What it changes:** Currently, Muon and AdamW operate on completely independent gradient scales. This normalizes the effective update magnitude ratio between the two paths, ensuring neither path dominates the model's forward pass changes.

**Exact implementation:**
In `step()`, after all groups are processed, compute and store a correction factor:

```python
# At the end of step(), after all groups processed:
adamw_update_norms = []
for group in self.param_groups:
    if group["kind"] == "adamw":
        for param in group["params"]:
            if param.grad is not None:
                state = self.state[param]
                # AdamW effective update = exp_avg / denom * lr
                beta1, beta2 = group["betas"]
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                denom = state["exp_avg_sq"].div(bc2).sqrt().add_(group["eps"])
                eff_update = state["exp_avg"] / denom * (group["lr"] / bc1)
                adamw_update_norms.append(float(eff_update.float().norm().item() / max(1e-8, param.float().norm().item())))

if adamw_update_norms and matrix_stats:
    adamw_ratio = sum(adamw_update_norms) / len(adamw_update_norms)
    muon_ratio = self.last_step_stats.mean_update_param_ratio
    # Store the imbalance for next step's correction
    self._path_imbalance = muon_ratio / max(adamw_ratio, 1e-8)
```

Then at the START of the next `_step_adamw`, apply correction:

```python
imbalance = getattr(self, '_path_imbalance', 1.0)
# If Muon is making proportionally larger updates, boost AdamW slightly
target_ratio = 1.0  # desired Muon:AdamW update ratio
correction = min(1.5, max(0.67, target_ratio / max(imbalance, 1e-8)))
step_size = group["lr"] * correction / bias_correction1
```

**Mechanistic reasoning:** In the current setup, Muon updates are orthogonalized and norm-preserving while AdamW updates scale with gradient magnitude. As training progresses and Muon WD decays toward 0, Muon matrices grow in norm while their update-to-parameter ratio shrinks. Meanwhile, AdamW parameters (especially embeddings at LR=0.2) can have update-to-parameter ratios that diverge from the matrix path. This creates an implicit "who's steering" effect: whichever path has higher effective update ratio dominates the next-step gradient signal for the other path. Balancing the ratio prevents one path from becoming passive. This is particularly important at the embedding<->matrix interface (first layer attention) where both paths directly interact.

**Risk level:** HIGH. This is a novel feedback loop between the two optimizer paths. The correction factor must be heavily clamped [0.67, 1.5] to prevent oscillation. The one-step lag means the correction is always slightly stale. However, if it works, it addresses a fundamental coordination problem that no existing optimizer handles.

**Dependencies:** H80 (the imbalance dynamics change completely with WD decay and momentum warmup). Also benefits from H83 (embedding WD) since without it, AdamW parameters have no WD and their norms grow unbounded, making the ratio computation less meaningful.

---

## Summary Table

| ID | Title | Risk | Depends On | Path |
|----|-------|------|------------|------|
| H80 | Corrected baseline: production schedule parity | LOW | None | Eval loop |
| H81 | AdamW beta1 warmup synced with Muon | LOW | H80 | Cross-path |
| H82 | Embedding-specific LR warmup (100 steps) | LOW-MED | H80 | AdamW |
| H83 | Coordinated embedding WD tied to Muon WD | MEDIUM | H80 | Cross-path |
| H84 | Loss-gated orthogonal mix reduction | MEDIUM | H80 | Muon + loss |
| H85 | Phase-aware trust ratio schedule | LOW-MED | H80 | Muon schedule |
| H86 | Muon update norm feedback to AdamW LR | MED-HIGH | H80 | Cross-path |
| H87 | Loss-reactive second moment beta2 | MEDIUM | H80 | Muon + loss |
| H88 | Phase transition detector: cautious->decoupled WD | MED-HIGH | H80 | Muon + phase |
| H89 | Cross-path gradient norm balancing | HIGH | H80, H83 | Cross-path |

**Recommended execution order:**
1. H80 (mandatory first — establishes true baseline)
2. H81, H82, H85 (low-risk, independent, can run in parallel)
3. H83, H84, H87 (medium-risk, independent)
4. H86, H88, H89 (high-risk, may benefit from insights from earlier results)
