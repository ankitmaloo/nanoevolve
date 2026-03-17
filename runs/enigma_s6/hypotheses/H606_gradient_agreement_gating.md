# H606: Gradient Agreement Gating Between Muon and AdamW

## Cross-domain inspiration: Ensemble Agreement in Gradient Boosting / Multi-Agent RL

In gradient boosting (XGBoost, LightGBM), each tree in the ensemble votes on the direction. When trees agree, the combined update is large. When they disagree, the update is conservative. This is implicitly a confidence mechanism — agreement = high confidence.

In multi-agent RL, agents that observe different parts of the state can share "confidence signals." When an agent's local observation is consistent with global signals, it acts boldly. When inconsistent, it hedges.

Muon and AdamW see gradients from different parameter types but the same loss. When their "implied loss landscape curvature" agrees, both can be aggressive. When they disagree, something is off — maybe embeddings and matrices are fighting each other.

## Mechanism

Compute a scalar "agreement" between Muon and AdamW's view of the loss landscape, then use it to gate update magnitudes.

```python
# Muon signal: average per-neuron variance (how curved is the matrix loss?)
muon_curvature = []
for group in optimizer.param_groups:
    if group["kind"] == "muon":
        state = optimizer.state[group["params"][0]]
        v = state["second_momentum_buffer"]
        muon_curvature.append(v.mean().item())
avg_muon_curvature = sum(muon_curvature) / len(muon_curvature)

# AdamW signal: average second moment (how curved is the embedding loss?)
adamw_curvature = []
for group in optimizer.param_groups:
    if group["kind"] == "adamw":
        for p in group["params"]:
            state = optimizer.state[p]
            if "exp_avg_sq" in state:
                adamw_curvature.append(state["exp_avg_sq"].mean().item())
avg_adamw_curvature = sum(adamw_curvature) / max(len(adamw_curvature), 1)

# Agreement: are they seeing similar curvature trends?
# Track ratio over time with EMA
curvature_ratio = avg_muon_curvature / (avg_adamw_curvature + 1e-12)
agreement_ema = 0.95 * agreement_ema + 0.05 * curvature_ratio

# If ratio is stable (close to its EMA), optimizers agree → be aggressive
# If ratio is volatile, they disagree → be conservative
ratio_stability = abs(curvature_ratio - agreement_ema) / (agreement_ema + 1e-12)

if ratio_stability < 0.1:  # stable agreement
    lr_gate = 1.0  # full learning rate
elif ratio_stability > 0.5:  # strong disagreement
    lr_gate = 0.5  # halve learning rate for both
else:
    lr_gate = 1.0 - 0.5 * (ratio_stability - 0.1) / 0.4

# Apply gate to both optimizers
for group in optimizer.param_groups:
    group["lr"] = group["initial_lr"] * lrm * lr_gate
```

## Why this is different from all Stage 5 mutations

Every Stage 5 mutation was **unilateral** — one optimizer changes, the other is oblivious. H606 is **bilateral** — both optimizers respond to a shared signal about their agreement. This is the first hypothesis where the interaction between Muon and AdamW is the mechanism, not a side effect.

## Risk

- Curvature ratio between different parameter types might be naturally non-stationary (e.g., embeddings converge while matrices are still learning) without indicating a problem.
- LR gating is blunt — better to gate eps or beta2 which are more targeted.
- Computation overhead of scanning all optimizer states every step.
