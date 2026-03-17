# H614: Muon Momentum from Gradient Signal-to-Noise Ratio

## Cross-domain inspiration: Wiener Filter (Signal Processing)

The Wiener filter is the optimal linear filter for extracting signal from noise. Its key parameter: the signal-to-noise ratio (SNR). When SNR is high, the filter passes the signal through. When SNR is low, it attenuates (smooths). The filter adapts its bandwidth based on the actual noise level, not a schedule.

Muon's momentum serves the same purpose — it smooths noisy gradients. High momentum (0.95) = heavy smoothing = good when gradients are noisy. Low momentum (0.85) = light smoothing = good when gradients are informative. Production warms from 0.85→0.95 over 300 steps, a fixed schedule.

But the actual SNR of gradients depends on batch size, data diversity, and training phase — not just step count.

## Mechanism

Estimate gradient SNR online, set momentum to match.

```python
# For each Muon group, track:
# signal = EMA of gradient (the "expected" direction)
# noise = EMA of (gradient - signal)^2

for group in optimizer.param_groups:
    if group["kind"] == "muon":
        state = optimizer.state[group["params"][0]]
        g = state["momentum_buffer"]  # current momentum buffer ≈ signal estimate

        # The gradient deviation from the momentum buffer is noise
        stacked_grads = torch.stack([p.grad for p in group["params"]])
        deviation = (stacked_grads - g).square().mean()
        signal_power = g.square().mean()

        # SNR estimate
        snr = signal_power / (deviation + 1e-12)

        # Map SNR to momentum: high SNR → low momentum (trust gradient)
        #                       low SNR → high momentum (smooth more)
        # Wiener optimal: momentum = noise / (signal + noise) = 1 / (1 + SNR)
        optimal_momentum = 1.0 / (1.0 + snr)
        # Clamp to reasonable range
        group["momentum"] = max(0.80, min(0.98, optimal_momentum.item()))
```

## Why this is principled

The Wiener filter's `1/(1+SNR)` formula is provably optimal for minimum mean-squared-error signal extraction from additive white noise. Momentum in SGD serves exactly this role — extracting the true gradient from stochastic estimates. Using the Wiener-optimal momentum is literally the best linear estimator for the gradient direction.

## Risk

- The deviation from momentum buffer isn't pure noise — it includes real gradient changes from a changing loss landscape. Could over-smooth during phase transitions.
- SNR computation adds overhead (gradient norms every step).
- The Wiener filter assumes stationary noise, which doesn't hold in training.
