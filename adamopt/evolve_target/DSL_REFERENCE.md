# Optimizer DSL Reference

This document describes the MatrixOptimizerSpec DSL — the search space for NanoChat optimizer evolution.

## Core Parameters

| Parameter | Type | Range | Default | Effect |
|-----------|------|-------|---------|--------|
| `momentum` | float | [0, 1) | 0.95 | EMA decay for gradient momentum buffer |
| `nesterov` | bool | — | true | Look-ahead momentum (usually better) |
| `momentum_placement` | str | pre_orthogonal, post_orthogonal | pre_orthogonal | When to apply momentum relative to NS orthogonalization |
| `orthogonalization` | str | polar_express, none | polar_express | Newton-Schulz orthogonalization for matrix params |
| `ns_steps` | int | > 0 | 5 | Newton-Schulz iterations (more = more accurate, slower) |
| `update_multiplier` | float | > 0 | 1.0 | Scales the entire update |
| `lr_aspect_scale` | bool | — | true | Scale LR by sqrt(fan_in/fan_out) for matrix params |

## Trust Ratio

Controls layerwise learning rate adaptation (like LARS/LAMB).

| Field | Range | Default | Effect |
|-------|-------|---------|--------|
| `mode` | "none", "layerwise" | "none" | Enable trust ratio |
| `clamp_min` | > 0 | 0.25 | Minimum trust ratio |
| `clamp_max` | >= clamp_min | 4.0 | Maximum trust ratio |

## Gradient Clipping

| Field | Range | Default | Effect |
|-------|-------|---------|--------|
| `mode` | "none", "update_rms", "global_norm" | "none" | Clipping strategy |
| `threshold` | > 0 | 1.0 | Clip threshold value |

## Weight Decay

| Field | Range | Default | Effect |
|-------|-------|---------|--------|
| `mode` | "none", "decoupled", "cautious" | "cautious" | Decay strategy. "cautious" only decays when gradient and param agree in sign. |
| `weight_decay` | >= 0 | 0.2 | Decay strength |

## Second Moment (adaptive LR like Adam)

| Field | Range | Default | Effect |
|-------|-------|---------|--------|
| `mode` | "none", "factored_rms" | "factored_rms" | Second moment estimation |
| `beta2` | [0, 1) | 0.95 | EMA decay for second moment |
| `eps` | > 0 | 1e-10 | Numerical stability |

## Stateful Control (Adaptive Gate)

When `enabled: true`, the optimizer reads training signals at each step and uses a learned gate function to blend between "aggressive" and "conservative" actuator settings.

### Gate Function

The gate output `g ∈ [0,1]` is computed as:
```
z = bias + Σ(coefficient_i × normalized_sensor_i)
g = sigmoid(sharpness × z)
```

Where `g=1` → fully aggressive, `g=0` → fully conservative.

### State Sensors (inputs to the gate)

| Sensor | What it measures |
|--------|-----------------|
| `loss_ema` | Smoothed training loss |
| `loss_improvement_ema` | Rate of loss decrease |
| `grad_norm_ema` | Smoothed gradient norm |
| `update_ratio_ema` | update_norm / param_norm ratio |
| `grad_alignment_ema` | Cosine similarity between consecutive gradients |
| `step_fraction` | Current step / total steps (0→1 over training) |

Each coefficient can be in [-8, 8]. The gate bias can also be in [-8, 8].

### Actuators (what the gate controls)

Each actuator has an `aggressive` and `conservative` value. The actual value used is:
```
actual = conservative + g × (aggressive - conservative)
```

| Actuator | Effect |
|----------|--------|
| `update_multiplier` | Scales the update (aggressive > 1 = bigger steps) |
| `trust_ratio_mix` | Blends trust ratio with identity (1 = full trust ratio) |
| `clip_threshold` | How aggressively to clip (lower = more clipping) |
| `beta2` | Second moment decay (lower = faster adaptation) |
| `orthogonal_mix` | Blend between orthogonalized and raw update |

## High-Impact Mutation Ideas

1. **Enable stateful control** with `loss_improvement_ema` and `step_fraction` coefficients — anneals behavior over training
2. **Try `clip: "update_rms"`** to prevent exploding updates
3. **Try `trust_ratio: "layerwise"`** for per-layer LR adaptation
4. **Lower momentum** (0.85-0.90) for faster adaptation to new gradients
5. **Post-orthogonal momentum** — apply momentum after NS orthogonalization
6. **Increase ns_steps** to 7-10 for better orthogonalization quality
7. **Tune beta2** — lower (0.8-0.9) = faster second moment adaptation
