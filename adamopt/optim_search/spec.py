from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


ALLOWED_ORTHOGONALIZATION = {"polar_express", "none"}
ALLOWED_MOMENTUM_PLACEMENT = {"pre_orthogonal", "post_orthogonal"}
ALLOWED_TRUST_RATIO = {"none", "layerwise"}
ALLOWED_CLIP = {"none", "update_rms", "global_norm"}
ALLOWED_DECAY = {"none", "decoupled", "cautious"}
ALLOWED_SECOND_MOMENT = {"none", "factored_rms"}
ALLOWED_STATE_SENSORS = {
    "loss_ema",
    "loss_improvement_ema",
    "grad_norm_ema",
    "update_ratio_ema",
    "grad_alignment_ema",
    "step_fraction",
}


@dataclass(frozen=True)
class TrustRatioConfig:
    mode: str = "none"
    clamp_min: float = 0.25
    clamp_max: float = 4.0
    eps: float = 1e-8


@dataclass(frozen=True)
class ClipConfig:
    mode: str = "none"
    threshold: float = 1.0


@dataclass(frozen=True)
class DecayConfig:
    mode: str = "cautious"
    weight_decay: float = 0.2


@dataclass(frozen=True)
class SecondMomentConfig:
    mode: str = "factored_rms"
    beta2: float = 0.95
    eps: float = 1e-10


@dataclass(frozen=True)
class GateCoefficients:
    loss_ema: float = 0.0
    loss_improvement_ema: float = 0.0
    grad_norm_ema: float = 0.0
    update_ratio_ema: float = 0.0
    grad_alignment_ema: float = 0.0
    step_fraction: float = 0.0

    def items(self) -> tuple[tuple[str, float], ...]:
        return tuple((name, getattr(self, name)) for name in sorted(ALLOWED_STATE_SENSORS))


@dataclass(frozen=True)
class GateConfig:
    coefficients: GateCoefficients = field(default_factory=GateCoefficients)
    bias: float = 0.0
    sharpness: float = 1.0


@dataclass(frozen=True)
class AdaptiveRange:
    aggressive: float
    conservative: float


@dataclass(frozen=True)
class AdaptiveActuatorConfig:
    update_multiplier: AdaptiveRange = field(default_factory=lambda: AdaptiveRange(aggressive=1.15, conservative=0.85))
    trust_ratio_mix: AdaptiveRange = field(default_factory=lambda: AdaptiveRange(aggressive=1.0, conservative=0.0))
    clip_threshold: AdaptiveRange = field(default_factory=lambda: AdaptiveRange(aggressive=2.0, conservative=0.5))
    beta2: AdaptiveRange = field(default_factory=lambda: AdaptiveRange(aggressive=0.90, conservative=0.99))
    orthogonal_mix: AdaptiveRange = field(default_factory=lambda: AdaptiveRange(aggressive=1.0, conservative=0.35))


@dataclass(frozen=True)
class StatefulControlConfig:
    enabled: bool = False
    ema_beta: float = 0.9
    loss_normalizer: float = 2.0
    improvement_normalizer: float = 0.05
    grad_norm_normalizer: float = 1.0
    update_ratio_normalizer: float = 0.05
    gate: GateConfig = field(default_factory=GateConfig)
    actuators: AdaptiveActuatorConfig = field(default_factory=AdaptiveActuatorConfig)


@dataclass(frozen=True)
class MatrixOptimizerSpec:
    name: str
    momentum: float = 0.95
    nesterov: bool = True
    momentum_placement: str = "pre_orthogonal"
    orthogonalization: str = "polar_express"
    ns_steps: int = 5
    trust_ratio: TrustRatioConfig = field(default_factory=TrustRatioConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    second_moment: SecondMomentConfig = field(default_factory=SecondMomentConfig)
    stateful_control: StatefulControlConfig = field(default_factory=StatefulControlConfig)
    update_multiplier: float = 1.0
    lr_aspect_scale: bool = True
    matrix_only: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.matrix_only:
            raise ValueError("Phase 1 only supports matrix-only optimizer search.")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        if self.momentum_placement not in ALLOWED_MOMENTUM_PLACEMENT:
            raise ValueError(f"unsupported momentum placement: {self.momentum_placement}")
        if self.orthogonalization not in ALLOWED_ORTHOGONALIZATION:
            raise ValueError(f"unsupported orthogonalization: {self.orthogonalization}")
        if self.orthogonalization != "none" and self.ns_steps <= 0:
            raise ValueError("ns_steps must be positive when orthogonalization is enabled.")
        if self.trust_ratio.mode not in ALLOWED_TRUST_RATIO:
            raise ValueError(f"unsupported trust ratio mode: {self.trust_ratio.mode}")
        if self.trust_ratio.clamp_min <= 0 or self.trust_ratio.clamp_max < self.trust_ratio.clamp_min:
            raise ValueError("invalid trust ratio clamp range.")
        if self.clip.mode not in ALLOWED_CLIP:
            raise ValueError(f"unsupported clip mode: {self.clip.mode}")
        if self.clip.mode != "none" and self.clip.threshold <= 0:
            raise ValueError("clip threshold must be positive.")
        if self.decay.mode not in ALLOWED_DECAY:
            raise ValueError(f"unsupported decay mode: {self.decay.mode}")
        if self.decay.weight_decay < 0:
            raise ValueError("weight decay must be non-negative.")
        if self.second_moment.mode not in ALLOWED_SECOND_MOMENT:
            raise ValueError(f"unsupported second moment mode: {self.second_moment.mode}")
        if self.second_moment.mode != "none" and not 0.0 <= self.second_moment.beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1).")
        if self.second_moment.eps <= 0:
            raise ValueError("second moment epsilon must be positive.")
        self._validate_stateful_control()
        if self.update_multiplier <= 0:
            raise ValueError("update_multiplier must be positive.")

    def _validate_stateful_control(self) -> None:
        ctrl = self.stateful_control
        if not 0.0 <= ctrl.ema_beta < 1.0:
            raise ValueError("stateful_control.ema_beta must be in [0, 1).")
        if ctrl.loss_normalizer <= 0 or ctrl.improvement_normalizer <= 0 or ctrl.grad_norm_normalizer <= 0 or ctrl.update_ratio_normalizer <= 0:
            raise ValueError("stateful control normalizers must be positive.")
        if ctrl.gate.sharpness <= 0:
            raise ValueError("gate sharpness must be positive.")
        for name, value in ctrl.gate.coefficients.items():
            if name not in ALLOWED_STATE_SENSORS:
                raise ValueError(f"unsupported gate sensor: {name}")
            if abs(value) > 8.0:
                raise ValueError(f"gate coefficient for {name} is out of bounds.")
        if abs(ctrl.gate.bias) > 8.0:
            raise ValueError("gate bias is out of bounds.")
        self._validate_range(ctrl.actuators.update_multiplier, min_value=0.1, name="update_multiplier")
        self._validate_range(ctrl.actuators.trust_ratio_mix, min_value=0.0, max_value=1.0, name="trust_ratio_mix")
        self._validate_range(ctrl.actuators.clip_threshold, min_value=1e-3, name="clip_threshold")
        self._validate_range(ctrl.actuators.beta2, min_value=0.0, max_value=0.999, name="beta2")
        self._validate_range(ctrl.actuators.orthogonal_mix, min_value=0.0, max_value=1.0, name="orthogonal_mix")

    @staticmethod
    def _validate_range(value: AdaptiveRange, *, min_value: float, name: str, max_value: float | None = None) -> None:
        for mode_name in ("aggressive", "conservative"):
            raw = getattr(value, mode_name)
            if raw < min_value:
                raise ValueError(f"{name}.{mode_name} must be >= {min_value}.")
            if max_value is not None and raw > max_value:
                raise ValueError(f"{name}.{mode_name} must be <= {max_value}.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "MatrixOptimizerSpec":
        payload = dict(payload)
        payload["trust_ratio"] = TrustRatioConfig(**payload.get("trust_ratio", {}))
        payload["clip"] = ClipConfig(**payload.get("clip", {}))
        payload["decay"] = DecayConfig(**payload.get("decay", {}))
        payload["second_moment"] = SecondMomentConfig(**payload.get("second_moment", {}))
        payload["stateful_control"] = _stateful_control_from_dict(payload.get("stateful_control", {}))
        return cls(**payload)

    @classmethod
    def baseline_nanochat(cls, weight_decay: float = 0.2) -> "MatrixOptimizerSpec":
        return cls(
            name="nanochat_muon_baseline",
            momentum=0.95,
            nesterov=True,
            momentum_placement="pre_orthogonal",
            orthogonalization="polar_express",
            ns_steps=5,
            trust_ratio=TrustRatioConfig(mode="none"),
            clip=ClipConfig(mode="none"),
            decay=DecayConfig(mode="cautious", weight_decay=weight_decay),
            second_moment=SecondMomentConfig(mode="factored_rms", beta2=0.95, eps=1e-10),
            update_multiplier=1.0,
            lr_aspect_scale=True,
            metadata={"baseline_target": "nanochat_muon"},
        )

    @classmethod
    def trust_ratio_variant(cls, weight_decay: float = 0.2) -> "MatrixOptimizerSpec":
        return cls(
            name="muon_with_layerwise_trust_ratio",
            momentum=0.95,
            nesterov=True,
            momentum_placement="pre_orthogonal",
            orthogonalization="polar_express",
            ns_steps=5,
            trust_ratio=TrustRatioConfig(mode="layerwise", clamp_min=0.5, clamp_max=1.5, eps=1e-8),
            clip=ClipConfig(mode="none"),
            decay=DecayConfig(mode="cautious", weight_decay=weight_decay),
            second_moment=SecondMomentConfig(mode="factored_rms", beta2=0.95, eps=1e-10),
            update_multiplier=1.0,
            lr_aspect_scale=True,
            metadata={"parent": "nanochat_muon_baseline", "mutation": "add_layerwise_trust_ratio"},
        )

    @classmethod
    def stateful_annealing_variant(cls, weight_decay: float = 0.2) -> "MatrixOptimizerSpec":
        return cls(
            name="muon_with_stateful_annealing_gate",
            momentum=0.95,
            nesterov=True,
            momentum_placement="pre_orthogonal",
            orthogonalization="polar_express",
            ns_steps=5,
            trust_ratio=TrustRatioConfig(mode="layerwise", clamp_min=0.5, clamp_max=1.5, eps=1e-8),
            clip=ClipConfig(mode="update_rms", threshold=1.0),
            decay=DecayConfig(mode="cautious", weight_decay=weight_decay),
            second_moment=SecondMomentConfig(mode="factored_rms", beta2=0.95, eps=1e-10),
            stateful_control=StatefulControlConfig(
                enabled=True,
                ema_beta=0.9,
                loss_normalizer=2.0,
                improvement_normalizer=0.05,
                grad_norm_normalizer=1.0,
                update_ratio_normalizer=0.05,
                gate=GateConfig(
                    coefficients=GateCoefficients(
                        loss_ema=0.4,
                        loss_improvement_ema=-0.8,
                        grad_norm_ema=0.5,
                        update_ratio_ema=0.3,
                        grad_alignment_ema=0.7,
                        step_fraction=-1.0,
                    ),
                    bias=-0.2,
                    sharpness=1.2,
                ),
                actuators=AdaptiveActuatorConfig(
                    update_multiplier=AdaptiveRange(aggressive=1.15, conservative=0.8),
                    trust_ratio_mix=AdaptiveRange(aggressive=0.8, conservative=0.25),
                    clip_threshold=AdaptiveRange(aggressive=1.5, conservative=0.65),
                    beta2=AdaptiveRange(aggressive=0.9, conservative=0.985),
                    orthogonal_mix=AdaptiveRange(aggressive=1.0, conservative=0.45),
                ),
            ),
            update_multiplier=1.0,
            lr_aspect_scale=True,
            metadata={"parent": "nanochat_muon_baseline", "mutation": "stateful_annealing_gate"},
        )

    def stable_id(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _adaptive_range_from_dict(payload: dict[str, object]) -> AdaptiveRange:
    return AdaptiveRange(**payload)


def _adaptive_actuators_from_dict(payload: dict[str, object]) -> AdaptiveActuatorConfig:
    payload = dict(payload)
    payload["update_multiplier"] = _adaptive_range_from_dict(payload.get("update_multiplier", {"aggressive": 1.15, "conservative": 0.85}))
    payload["trust_ratio_mix"] = _adaptive_range_from_dict(payload.get("trust_ratio_mix", {"aggressive": 1.0, "conservative": 0.0}))
    payload["clip_threshold"] = _adaptive_range_from_dict(payload.get("clip_threshold", {"aggressive": 2.0, "conservative": 0.5}))
    payload["beta2"] = _adaptive_range_from_dict(payload.get("beta2", {"aggressive": 0.9, "conservative": 0.99}))
    payload["orthogonal_mix"] = _adaptive_range_from_dict(payload.get("orthogonal_mix", {"aggressive": 1.0, "conservative": 0.35}))
    return AdaptiveActuatorConfig(**payload)


def _stateful_control_from_dict(payload: dict[str, object]) -> StatefulControlConfig:
    payload = dict(payload)
    gate_payload = dict(payload.get("gate", {}))
    gate_payload["coefficients"] = GateCoefficients(**gate_payload.get("coefficients", {}))
    payload["gate"] = GateConfig(**gate_payload)
    payload["actuators"] = _adaptive_actuators_from_dict(payload.get("actuators", {}))
    return StatefulControlConfig(**payload)
