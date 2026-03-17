# H605: Batch-Size Aware Eps Warmup

## Cross-domain inspiration: Central Limit Theorem / Statistical Estimation Theory

The variance of the sample mean scales as `sigma^2 / n`. A batch of size B gives gradient estimates with variance proportional to `1/B`. This means:
- Small batch (B=1024): gradient variance is high, second moments need many steps to converge
- Large batch (B=524288): gradient variance is 512x lower, second moments converge 512x faster

H73 schedules eps over `total_steps`, which doesn't account for this. The dangerous window (where eps matters) should scale with how fast second moments converge, which depends on batch size.

## Mechanism

Scale the eps warmup duration by effective sample count rather than step count.

```python
# How many gradient samples has exp_avg_sq seen?
# With beta2=0.95, effective window ≈ 1/(1-0.95) = 20 steps
# Each step sees B tokens, so effective samples ≈ 20 * B
# At B=1024: ~20k samples. At B=524288: ~10M samples.

# Warmup eps based on effective sample count
effective_samples = step * batch_size
reference_samples = 20_000 * 1024  # calibrated to H73's sweet spot at small batch

frac = min(1.0, effective_samples / reference_samples)
eps = 10.0 ** (-6.0 + (-4.0 * frac))
```

At batch=1024, this matches H73 (20k steps to converge).
At batch=524288, eps reaches 1e-10 after ~40 steps instead of 20k.

## Simpler version

Just scale the schedule duration:

```python
warmup_steps = int(20000 * (1024 / batch_size))  # ~40 steps at production batch
frac = min(1.0, step / warmup_steps)
eps = 10.0 ** (-6.0 + (-4.0 * frac))
```

## Why this matters for the speedrun

The Time-to-GPT-2 target uses batch=524288. If H73's gain comes from early-phase protection, the relevant window at production batch is only ~40 steps, not 20k. Running the schedule over 20k steps at large batch means eps stays unnecessarily high for most of training, which might actually hurt.

Conversely, if you port H73 to production with the same 20k-step schedule, you're being overly conservative for 99.8% of training. H605 avoids this.

## Risk

- If H73's benefit is not just "early phase protection" but ongoing (diverse data requires perpetually higher eps), then shortening the schedule kills the gain.
- Need to validate at both batch sizes to confirm the batch-scaling hypothesis.
- The `reference_samples` calibration point is somewhat arbitrary.
