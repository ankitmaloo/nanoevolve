# H604: Shared Noise-Phase Detector

## Cross-domain inspiration: Regime Detection in Finance / Hidden Markov Models

Financial markets alternate between low-volatility ("risk-on") and high-volatility ("risk-off") regimes. Quant strategies detect these regimes using running volatility estimates, then adjust position sizes and hedging accordingly. A common approach: compute realized volatility over a rolling window, compare to a threshold, and switch strategy.

Neural network training has similar regimes:
- **Early phase**: Gradients are noisy, second moments unreliable, loss is high and changing fast
- **Stable phase**: Gradients are informative, second moments converged, loss descends smoothly
- **Warmdown phase**: LR drops, gradients become small, different optimization dynamics

Currently these phases are encoded as fixed time intervals. But the actual transitions depend on batch size, model scale, and data. A detector that reads the actual state would generalize.

## Mechanism

Compute a scalar "training noise level" from both optimizers' states, then use it to control hyperparams for both.

```python
# Noise metric from AdamW: how much is exp_avg_sq changing?
adamw_noise = 0.0
adamw_count = 0
for group in optimizer.param_groups:
    if group["kind"] == "adamw":
        for p in group["params"]:
            state = optimizer.state[p]
            sq = state["exp_avg_sq"]
            # Compare current second moment to its EMA
            adamw_noise += (grad.square() / (sq + 1e-12) - 1.0).abs().mean().item()
            adamw_count += 1
adamw_noise /= max(adamw_count, 1)

# Noise metric from Muon: how much is second_momentum_buffer changing?
muon_noise = 0.0
muon_count = 0
for group in optimizer.param_groups:
    if group["kind"] == "muon":
        state = optimizer.state[group["params"][0]]
        v = state["second_momentum_buffer"]
        # Track step-to-step change (need to store previous)
        if "prev_v" in state:
            muon_noise += (v - state["prev_v"]).abs().mean().item() / (v.abs().mean().item() + 1e-12)
            muon_count += 1
        state["prev_v"] = v.clone()
muon_noise /= max(muon_count, 1)

# Combined noise metric (EMA-smoothed)
noise_level = 0.95 * noise_level + 0.05 * (adamw_noise + muon_noise) / 2

# Regime-dependent hyperparams
if noise_level > threshold_high:
    # Noisy regime: both conservative
    adamw_eps_scale = 100.0   # high eps
    muon_beta2 = 0.8          # fast variance tracking
elif noise_level < threshold_low:
    # Stable regime: both aggressive
    adamw_eps_scale = 1.0     # low eps
    muon_beta2 = 0.95         # standard
else:
    # Transition: interpolate
    t = (noise_level - threshold_low) / (threshold_high - threshold_low)
    adamw_eps_scale = 1.0 + 99.0 * t
    muon_beta2 = 0.95 - 0.15 * t
```

## Why both optimizers should see the same signal

Muon and AdamW optimize different parameters but they share the backward pass. If the loss landscape is noisy for matrices, it's likely noisy for embeddings too — they're connected through the same computation graph. A shared detector avoids the failure mode where one optimizer is aggressive while the other is conservative, creating conflicting update scales.

## Analogy to HMM

The noise_level acts like a hidden state in an HMM. We observe gradient statistics (emissions) and infer the latent regime. The regime then controls the policy (hyperparameters). This is exactly the Baum-Welch / forward algorithm, simplified to a scalar state with EMA transitions.

## Risk

- Two thresholds (threshold_high, threshold_low) are new hyperparameters.
- The noise metric computation adds overhead every step.
- May be redundant with simpler approaches like H601 if the phase transition is driven purely by second-moment convergence.
