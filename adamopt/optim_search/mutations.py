from __future__ import annotations

import random
from dataclasses import replace

from .spec import (
    AdaptiveActuatorConfig,
    AdaptiveRange,
    ClipConfig,
    DecayConfig,
    GateCoefficients,
    GateConfig,
    MatrixOptimizerSpec,
    SecondMomentConfig,
    StatefulControlConfig,
    TrustRatioConfig,
)


def mutate_spec(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    operators = [
        _toggle_momentum_placement,
        _toggle_trust_ratio,
        _adjust_trust_ratio_clamp,
        _toggle_clip_policy,
        _toggle_decay_mode,
        _scale_update_multiplier,
        _adjust_ns_steps,
        _adjust_second_moment_beta,
        _toggle_stateful_control,
        _adjust_gate_bias,
        _adjust_gate_weight,
        _adjust_stateful_ema_beta,
        _adjust_stateful_range,
    ]
    op = rng.choice(operators)
    mutated, lineage = op(spec, rng)
    lineage["parent_spec"] = spec.name
    return mutated, lineage


def _toggle_momentum_placement(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    placement = "post_orthogonal" if spec.momentum_placement == "pre_orthogonal" else "pre_orthogonal"
    name = f"{spec.name}_mom_{placement}"
    return replace(spec, name=name, momentum_placement=placement), {"mutation": "toggle_momentum_placement", "value": placement}


def _toggle_trust_ratio(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    enabled = spec.trust_ratio.mode == "none"
    trust = TrustRatioConfig(mode="layerwise", clamp_min=0.5, clamp_max=1.5, eps=1e-8) if enabled else TrustRatioConfig(mode="none")
    suffix = "trust_on" if enabled else "trust_off"
    return replace(spec, name=f"{spec.name}_{suffix}", trust_ratio=trust), {"mutation": "toggle_trust_ratio", "enabled": enabled}


def _adjust_trust_ratio_clamp(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    trust = spec.trust_ratio
    if trust.mode == "none":
        trust = TrustRatioConfig(mode="layerwise", clamp_min=0.5, clamp_max=1.5, eps=1e-8)
    delta = rng.choice([-0.25, 0.25])
    clamp_min = max(0.1, trust.clamp_min + delta)
    clamp_max = max(clamp_min, trust.clamp_max + delta)
    updated = replace(trust, clamp_min=clamp_min, clamp_max=clamp_max)
    return replace(spec, name=f"{spec.name}_trust_clamp", trust_ratio=updated), {"mutation": "adjust_trust_ratio_clamp", "clamp_min": clamp_min, "clamp_max": clamp_max}


def _toggle_clip_policy(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    if spec.clip.mode == "none":
        clip = ClipConfig(mode="update_rms", threshold=1.0)
    elif spec.clip.mode == "update_rms":
        clip = ClipConfig(mode="global_norm", threshold=2.5)
    else:
        clip = ClipConfig(mode="none")
    return replace(spec, name=f"{spec.name}_clip_{clip.mode}", clip=clip), {"mutation": "toggle_clip_policy", "mode": clip.mode}


def _toggle_decay_mode(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    next_mode = {"cautious": "decoupled", "decoupled": "none", "none": "cautious"}[spec.decay.mode]
    decay = DecayConfig(mode=next_mode, weight_decay=spec.decay.weight_decay)
    return replace(spec, name=f"{spec.name}_decay_{next_mode}", decay=decay), {"mutation": "toggle_decay_mode", "mode": next_mode}


def _scale_update_multiplier(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    factor = rng.choice([0.85, 0.95, 1.05, 1.15])
    multiplier = max(0.1, spec.update_multiplier * factor)
    return replace(spec, name=f"{spec.name}_mul_{multiplier:.2f}", update_multiplier=multiplier), {"mutation": "scale_update_multiplier", "multiplier": multiplier}


def _adjust_ns_steps(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    ns_steps = min(7, max(1, spec.ns_steps + rng.choice([-2, -1, 1, 2])))
    return replace(spec, name=f"{spec.name}_ns_{ns_steps}", ns_steps=ns_steps), {"mutation": "adjust_ns_steps", "ns_steps": ns_steps}


def _adjust_second_moment_beta(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    if spec.second_moment.mode == "none":
        second = SecondMomentConfig(mode="factored_rms", beta2=0.95, eps=1e-10)
    else:
        beta2 = min(0.99, max(0.85, spec.second_moment.beta2 + rng.choice([-0.02, 0.02])))
        second = replace(spec.second_moment, beta2=beta2)
    return replace(spec, name=f"{spec.name}_beta2_{second.beta2:.2f}", second_moment=second), {"mutation": "adjust_second_moment_beta", "beta2": second.beta2}


def _default_stateful_control() -> StatefulControlConfig:
    return StatefulControlConfig(
        enabled=True,
        ema_beta=0.9,
        gate=GateConfig(
            coefficients=GateCoefficients(
                loss_ema=0.3,
                loss_improvement_ema=-0.5,
                grad_norm_ema=0.2,
                update_ratio_ema=0.2,
                grad_alignment_ema=0.5,
                step_fraction=-0.8,
            ),
            bias=0.0,
            sharpness=1.0,
        ),
        actuators=AdaptiveActuatorConfig(
            update_multiplier=AdaptiveRange(aggressive=1.1, conservative=0.85),
            trust_ratio_mix=AdaptiveRange(aggressive=0.7, conservative=0.1),
            clip_threshold=AdaptiveRange(aggressive=1.3, conservative=0.7),
            beta2=AdaptiveRange(aggressive=0.9, conservative=0.985),
            orthogonal_mix=AdaptiveRange(aggressive=1.0, conservative=0.5),
        ),
    )


def _toggle_stateful_control(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    if spec.stateful_control.enabled:
        ctrl = replace(spec.stateful_control, enabled=False)
        enabled = False
    else:
        ctrl = _default_stateful_control()
        enabled = True
    return replace(spec, name=f"{spec.name}_stateful_{'on' if enabled else 'off'}", stateful_control=ctrl), {
        "mutation": "toggle_stateful_control",
        "enabled": enabled,
    }


def _adjust_gate_bias(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    ctrl = spec.stateful_control if spec.stateful_control.enabled else _default_stateful_control()
    bias = max(-8.0, min(8.0, ctrl.gate.bias + rng.choice([-0.5, 0.5])))
    gate = replace(ctrl.gate, bias=bias)
    return replace(spec, name=f"{spec.name}_gate_bias_{bias:+.1f}", stateful_control=replace(ctrl, enabled=True, gate=gate)), {
        "mutation": "adjust_gate_bias",
        "bias": bias,
    }


def _adjust_gate_weight(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    ctrl = spec.stateful_control if spec.stateful_control.enabled else _default_stateful_control()
    sensor = rng.choice(sorted(("loss_ema", "loss_improvement_ema", "grad_norm_ema", "update_ratio_ema", "grad_alignment_ema", "step_fraction")))
    delta = rng.choice([-0.4, 0.4])
    coefficients = dict(ctrl.gate.coefficients.items())
    coefficients[sensor] = max(-8.0, min(8.0, coefficients[sensor] + delta))
    gate = replace(ctrl.gate, coefficients=GateCoefficients(**coefficients))
    return replace(spec, name=f"{spec.name}_gate_{sensor}", stateful_control=replace(ctrl, enabled=True, gate=gate)), {
        "mutation": "adjust_gate_weight",
        "sensor": sensor,
        "weight": coefficients[sensor],
    }


def _adjust_stateful_ema_beta(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    ctrl = spec.stateful_control if spec.stateful_control.enabled else _default_stateful_control()
    ema_beta = min(0.99, max(0.5, ctrl.ema_beta + rng.choice([-0.05, 0.05])))
    return replace(spec, name=f"{spec.name}_ema_{ema_beta:.2f}", stateful_control=replace(ctrl, enabled=True, ema_beta=ema_beta)), {
        "mutation": "adjust_stateful_ema_beta",
        "ema_beta": ema_beta,
    }


def _adjust_stateful_range(spec: MatrixOptimizerSpec, rng: random.Random) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    ctrl = spec.stateful_control if spec.stateful_control.enabled else _default_stateful_control()
    actuator_name = rng.choice(["update_multiplier", "trust_ratio_mix", "clip_threshold", "beta2", "orthogonal_mix"])
    actuator = getattr(ctrl.actuators, actuator_name)
    field_name = rng.choice(["aggressive", "conservative"])
    delta = rng.choice([-0.1, 0.1])
    updated_value = getattr(actuator, field_name) + delta
    if actuator_name == "beta2":
        updated_value = min(0.999, max(0.0, updated_value))
    elif actuator_name in {"trust_ratio_mix", "orthogonal_mix"}:
        updated_value = min(1.0, max(0.0, updated_value))
    else:
        updated_value = max(0.1 if actuator_name == "update_multiplier" else 1e-3, updated_value)
    updated_actuator = replace(actuator, **{field_name: updated_value})
    actuators = replace(ctrl.actuators, **{actuator_name: updated_actuator})
    return replace(spec, name=f"{spec.name}_{actuator_name}_{field_name}", stateful_control=replace(ctrl, enabled=True, actuators=actuators)), {
        "mutation": "adjust_stateful_range",
        "actuator": actuator_name,
        "field": field_name,
        "value": updated_value,
    }
