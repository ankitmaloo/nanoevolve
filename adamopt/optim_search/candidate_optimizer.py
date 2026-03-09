from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from .spec import MatrixOptimizerSpec, StatefulControlConfig


POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@dataclass
class OptimizerStepStats:
    mean_update_param_ratio: float = 0.0
    max_update_param_ratio: float = 0.0
    mean_trust_ratio: float = 1.0
    max_grad_norm: float = 0.0
    mean_gate: float = 0.0
    groups_touched: int = 0


@dataclass
class TrainingStateSignals:
    loss_ema: float = 0.0
    loss_improvement_ema: float = 0.0
    grad_norm_ema: float = 0.0
    update_ratio_ema: float = 0.0
    grad_alignment_ema: float = 0.0
    step_fraction: float = 0.0
    initialized: bool = False


class ToyBlock(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(model_dim, model_dim, bias=False)
        self.mlp = nn.Linear(model_dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, model_dim, bias=False)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor) -> Tensor:
        y = torch.tanh(self.attn(x))
        y = torch.relu(self.mlp(y))
        y = self.proj(y)
        return self.norm(x + y)


class ToyNanoChatModel(nn.Module):
    """
    Small deterministic model that mirrors NanoChat's optimizer fault lines:
    transformer blocks are 2D matrix-heavy, while embeddings/lm_head/scalars stay separate.
    """

    def __init__(self, vocab_size: int, model_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(vocab_size, model_dim),
                "h": nn.ModuleList([ToyBlock(model_dim, hidden_dim) for _ in range(layers)]),
            }
        )
        self.value_embeds = nn.Embedding(4, model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(layers))
        self.x0_lambdas = nn.Parameter(torch.ones(layers))
        self.layers = layers
        self.vocab_size = vocab_size

    def forward(self, idx: Tensor) -> Tensor:
        x = self.transformer["wte"](idx)
        side = self.value_embeds(idx % 4)
        x = x + 0.1 * side
        for layer_idx, block in enumerate(self.transformer["h"]):
            x0 = x.mean(dim=1, keepdim=True)
            mixed = block(x)
            x = x + self.resid_lambdas[layer_idx] * mixed + 0.1 * self.x0_lambdas[layer_idx] * x0
        return self.lm_head(x)


def _group_by_shape(params: Iterable[nn.Parameter]) -> list[list[nn.Parameter]]:
    shape_to_params: dict[tuple[int, ...], list[nn.Parameter]] = {}
    for param in params:
        shape_to_params.setdefault(tuple(param.shape), []).append(param)
    return [shape_to_params[shape] for shape in sorted(shape_to_params)]


def build_nanochat_param_groups(
    model: nn.Module,
    *,
    unembedding_lr: float = 0.004,
    embedding_lr: float = 0.2,
    matrix_lr: float = 0.02,
    weight_decay: float = 0.2,
    adam_betas: tuple[float, float] = (0.8, 0.95),
    scalar_lr: float = 0.5,
) -> list[dict[str, object]]:
    if hasattr(model, "transformer") and isinstance(getattr(model, "transformer"), nn.ModuleDict):
        transformer = model.transformer
        wte_params = list(transformer["wte"].parameters())
        block_params = list(transformer["h"].parameters())
        matrix_params = [param for param in block_params if param.ndim >= 2]
        block_scalar_params = [param for param in block_params if param.ndim < 2]
        value_embeds_params = list(getattr(model, "value_embeds", nn.Module()).parameters())
        lm_head_params = list(getattr(model, "lm_head", nn.Module()).parameters())
        resid_params = [getattr(model, "resid_lambdas")] if hasattr(model, "resid_lambdas") else []
        x0_params = [getattr(model, "x0_lambdas")] if hasattr(model, "x0_lambdas") else []
        model_dim = getattr(model.lm_head, "in_features", 768)
    else:
        named = list(model.named_parameters())
        matrix_params = [p for _, p in named if p.ndim >= 2]
        non_matrix = [p for _, p in named if p.ndim < 2]
        wte_params = []
        value_embeds_params = []
        lm_head_params = []
        resid_params = non_matrix
        x0_params = []
        block_scalar_params = []
        model_dim = 768

    dmodel_lr_scale = (model_dim / 768) ** -0.5
    param_groups: list[dict[str, object]] = [
        dict(kind="adamw", group_name="lm_head", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", group_name="embedding", params=wte_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", group_name="value_embeds", params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", group_name="block_non_matrix", params=block_scalar_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", group_name="resid_scalars", params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", group_name="x0_scalars", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for group_index, grouped in enumerate(_group_by_shape(matrix_params)):
        param_groups.append(
            dict(
                kind="matrix_candidate",
                group_name=f"matrix_shape_{group_index}",
                params=grouped,
                lr=matrix_lr,
                weight_decay=weight_decay,
            )
        )
    return [group for group in param_groups if group["params"]]


class SpecCandidateOptimizer(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict[str, object]], spec: MatrixOptimizerSpec) -> None:
        super().__init__(param_groups, defaults={})
        self.spec = spec
        self.last_step_stats = OptimizerStepStats()
        self.training_signals = TrainingStateSignals()
        self._step_context: dict[str, float] = {"loss": 0.0, "step": 0.0, "total_steps": 1.0}

    @property
    def estimated_state_bytes(self) -> int:
        total = 0
        for state in self.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total

    def set_step_context(self, *, loss_value: float, step: int, total_steps: int) -> None:
        self._step_context = {
            "loss": float(loss_value),
            "step": float(step),
            "total_steps": float(max(1, total_steps)),
        }

    def _stateful_enabled(self) -> bool:
        return self.spec.stateful_control.enabled

    def _actuator_value(self, value_name: str, gate: float) -> float:
        ctrl = self.spec.stateful_control
        value = getattr(ctrl.actuators, value_name)
        return gate * value.aggressive + (1.0 - gate) * value.conservative

    def _normalized_signals(self) -> dict[str, float]:
        ctrl = self.spec.stateful_control
        return {
            "loss_ema": self.training_signals.loss_ema / ctrl.loss_normalizer,
            "loss_improvement_ema": self.training_signals.loss_improvement_ema / ctrl.improvement_normalizer,
            "grad_norm_ema": self.training_signals.grad_norm_ema / ctrl.grad_norm_normalizer,
            "update_ratio_ema": self.training_signals.update_ratio_ema / ctrl.update_ratio_normalizer,
            "grad_alignment_ema": self.training_signals.grad_alignment_ema,
            "step_fraction": self.training_signals.step_fraction,
        }

    def _gate_value(self) -> float:
        if not self._stateful_enabled():
            return 1.0
        ctrl = self.spec.stateful_control
        signals = self._normalized_signals()
        score = ctrl.gate.bias
        for name, weight in ctrl.gate.coefficients.items():
            score += weight * signals[name]
        score *= ctrl.gate.sharpness
        return float(torch.sigmoid(torch.tensor(score)).item())

    def _update_loss_signals(self) -> None:
        ctrl = self.spec.stateful_control
        loss_value = self._step_context["loss"]
        prev_loss_ema = self.training_signals.loss_ema
        if not self.training_signals.initialized:
            self.training_signals.loss_ema = loss_value
            self.training_signals.loss_improvement_ema = 0.0
            self.training_signals.initialized = True
        else:
            self.training_signals.loss_ema = ctrl.ema_beta * self.training_signals.loss_ema + (1.0 - ctrl.ema_beta) * loss_value
            improvement = prev_loss_ema - self.training_signals.loss_ema
            self.training_signals.loss_improvement_ema = (
                ctrl.ema_beta * self.training_signals.loss_improvement_ema + (1.0 - ctrl.ema_beta) * improvement
            )
        self.training_signals.step_fraction = min(1.0, self._step_context["step"] / self._step_context["total_steps"])

    def _update_post_step_signals(self, *, grad_norm: float, update_ratio: float, grad_alignment: float) -> None:
        if not self._stateful_enabled():
            return
        ctrl = self.spec.stateful_control
        if not self.training_signals.initialized:
            self.training_signals.grad_norm_ema = grad_norm
            self.training_signals.update_ratio_ema = update_ratio
            self.training_signals.grad_alignment_ema = grad_alignment
            self.training_signals.initialized = True
            return
        beta = ctrl.ema_beta
        self.training_signals.grad_norm_ema = beta * self.training_signals.grad_norm_ema + (1.0 - beta) * grad_norm
        self.training_signals.update_ratio_ema = beta * self.training_signals.update_ratio_ema + (1.0 - beta) * update_ratio
        self.training_signals.grad_alignment_ema = beta * self.training_signals.grad_alignment_ema + (1.0 - beta) * grad_alignment

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

    def _apply_momentum(self, update: Tensor, momentum_buffer: Tensor) -> Tensor:
        momentum_buffer.lerp_(update, 1 - self.spec.momentum)
        if self.spec.nesterov:
            return update.lerp(momentum_buffer, self.spec.momentum)
        return momentum_buffer

    def _orthogonalize(self, update: Tensor, *, orthogonal_mix: float = 1.0) -> Tensor:
        if self.spec.orthogonalization == "none":
            return update
        raw_update = update
        x = update.to(dtype=torch.float32)
        x = x / (x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6) * 1.02)
        for a, b, c in POLAR_EXPRESS_COEFFS[: self.spec.ns_steps]:
            if x.size(-2) > x.size(-1):
                gram = x.mT @ x
                x = a * x + x @ (b * gram + c * (gram @ gram))
            else:
                gram = x @ x.mT
                x = a * x + (b * gram + c * (gram @ gram)) @ x
        orthogonalized = x.to(dtype=update.dtype)
        if orthogonal_mix >= 1.0:
            return orthogonalized
        if orthogonal_mix <= 0.0:
            return raw_update
        return orthogonal_mix * orthogonalized + (1.0 - orthogonal_mix) * raw_update

    def _apply_second_moment(self, update: Tensor, second_moment_buffer: Tensor, *, beta2_override: float | None = None) -> Tensor:
        if self.spec.second_moment.mode == "none":
            return update
        red_dim = -1 if update.size(-2) >= update.size(-1) else -2
        v_mean = update.float().square().mean(dim=red_dim, keepdim=True)
        beta2 = self.spec.second_moment.beta2 if beta2_override is None else beta2_override
        second_moment_buffer.lerp_(v_mean.to(dtype=second_moment_buffer.dtype), 1 - beta2)
        step_size = second_moment_buffer.clamp_min(self.spec.second_moment.eps).rsqrt()
        red_dim_size = update.size(red_dim)
        v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(self.spec.second_moment.eps)
        final_scale = step_size * (v_norm / v_norm_new)
        return update * final_scale.to(dtype=update.dtype)

    def _apply_trust_ratio(self, update: Tensor, params: Tensor, *, trust_mix: float = 1.0) -> tuple[Tensor, Tensor]:
        if self.spec.trust_ratio.mode == "none":
            ones = torch.ones(update.size(0), device=update.device, dtype=update.dtype)
            return update, ones
        param_norm = params.float().flatten(1).norm(dim=1).clamp_min(self.spec.trust_ratio.eps)
        update_norm = update.float().flatten(1).norm(dim=1).clamp_min(self.spec.trust_ratio.eps)
        trust = (param_norm / update_norm).clamp(self.spec.trust_ratio.clamp_min, self.spec.trust_ratio.clamp_max)
        trust = 1.0 + trust_mix * (trust - 1.0)
        reshape = trust.view(-1, *([1] * (update.ndim - 1)))
        return update * reshape.to(dtype=update.dtype), trust

    def _apply_clip(self, update: Tensor, *, threshold_override: float | None = None) -> Tensor:
        if self.spec.clip.mode == "none":
            return update
        threshold = self.spec.clip.threshold if threshold_override is None else threshold_override
        if self.spec.clip.mode == "update_rms":
            rms = update.float().square().mean().sqrt()
            if float(rms.item()) <= threshold:
                return update
            return update * (threshold / float(rms.item()))
        norm = update.float().norm().clamp_min(1e-8)
        if float(norm.item()) <= threshold:
            return update
        return update * (threshold / float(norm.item()))

    def _step_matrix_group(self, group: dict[str, object]) -> OptimizerStepStats:
        params = [param for param in group["params"] if param.grad is not None]
        if not params:
            return OptimizerStepStats()

        anchor = params[0]
        state = self.state[anchor]
        shape = tuple(anchor.shape)
        if state.get("shape") != shape or state.get("count") != len(params):
            state["shape"] = shape
            state["count"] = len(params)
            state["momentum_buffer"] = torch.zeros((len(params), *shape), dtype=anchor.dtype, device=anchor.device)
            red_dim = -1 if shape[-2] >= shape[-1] else -2
            state_shape = (len(params), shape[-2], 1) if red_dim == -1 else (len(params), 1, shape[-1])
            state["second_moment_buffer"] = torch.zeros(state_shape, dtype=anchor.dtype, device=anchor.device)

        momentum_buffer = state["momentum_buffer"]
        second_moment_buffer = state["second_moment_buffer"]
        grads = torch.stack([param.grad.detach() for param in params])
        stacked_params = torch.stack([param.detach() for param in params])
        gate = self._gate_value()
        orthogonal_mix = self._actuator_value("orthogonal_mix", gate) if self._stateful_enabled() else 1.0
        trust_mix = self._actuator_value("trust_ratio_mix", gate) if self._stateful_enabled() else 1.0
        clip_threshold = self._actuator_value("clip_threshold", gate) if self._stateful_enabled() else None
        beta2_override = self._actuator_value("beta2", gate) if self._stateful_enabled() else None
        update_multiplier = self.spec.update_multiplier
        if self._stateful_enabled():
            update_multiplier *= self._actuator_value("update_multiplier", gate)

        if self.spec.momentum_placement == "pre_orthogonal":
            update = self._apply_momentum(grads, momentum_buffer)
            update = self._orthogonalize(update, orthogonal_mix=orthogonal_mix)
        else:
            update = self._orthogonalize(grads, orthogonal_mix=orthogonal_mix)
            update = self._apply_momentum(update, momentum_buffer)

        update = self._apply_second_moment(update, second_moment_buffer, beta2_override=beta2_override)
        update, trust = self._apply_trust_ratio(update, stacked_params, trust_mix=trust_mix)
        update = self._apply_clip(update, threshold_override=clip_threshold)
        update = update * update_multiplier

        lr = float(group["lr"])
        if self.spec.lr_aspect_scale:
            lr *= math.sqrt(max(1.0, stacked_params.size(-2) / stacked_params.size(-1)))

        if self.spec.decay.mode == "decoupled" and self.spec.decay.weight_decay:
            stacked_params.mul_(1 - lr * self.spec.decay.weight_decay)
            stacked_params.add_(update, alpha=-lr)
        elif self.spec.decay.mode == "cautious" and self.spec.decay.weight_decay:
            mask = (update * stacked_params) >= 0
            stacked_params.sub_(lr * update + lr * self.spec.decay.weight_decay * stacked_params * mask)
        else:
            stacked_params.add_(update, alpha=-lr)

        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

        param_norm = stacked_params.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        update_norm = update.float().flatten(1).norm(dim=1)
        ratios = update_norm / param_norm
        grad_norm = grads.float().flatten(1).norm(dim=1)
        return OptimizerStepStats(
            mean_update_param_ratio=float(ratios.mean().item()),
            max_update_param_ratio=float(ratios.max().item()),
            mean_trust_ratio=float(trust.float().mean().item()),
            max_grad_norm=float(grad_norm.max().item()),
            mean_gate=gate,
            groups_touched=1,
        )

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        if closure is not None:
            with torch.enable_grad():
                closure()
        self._update_loss_signals()

        matrix_stats: list[OptimizerStepStats] = []
        max_grad_norm = 0.0
        mean_update_ratio = 0.0
        mean_grad_alignment = 0.0
        for group in self.param_groups:
            if group["kind"] == "adamw":
                max_grad_norm = max(max_grad_norm, self._step_adamw(group))
            elif group["kind"] == "matrix_candidate":
                stats = self._step_matrix_group(group)
                matrix_stats.append(stats)
                max_grad_norm = max(max_grad_norm, stats.max_grad_norm)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        if matrix_stats:
            mean_update_ratio = sum(item.mean_update_param_ratio for item in matrix_stats) / len(matrix_stats)
            mean_grad_alignment = self.training_signals.grad_alignment_ema
            self.last_step_stats = OptimizerStepStats(
                mean_update_param_ratio=mean_update_ratio,
                max_update_param_ratio=max(item.max_update_param_ratio for item in matrix_stats),
                mean_trust_ratio=sum(item.mean_trust_ratio for item in matrix_stats) / len(matrix_stats),
                max_grad_norm=max_grad_norm,
                mean_gate=sum(item.mean_gate for item in matrix_stats) / len(matrix_stats),
                groups_touched=len(matrix_stats),
            )
        else:
            self.last_step_stats = OptimizerStepStats(max_grad_norm=max_grad_norm)

        if matrix_stats:
            grad_norm_value = sum(item.max_grad_norm for item in matrix_stats) / len(matrix_stats)
            grad_alignment_value = self._compute_grad_alignment()
            self._update_post_step_signals(
                grad_norm=grad_norm_value,
                update_ratio=mean_update_ratio,
                grad_alignment=grad_alignment_value,
            )

        return None

    def _compute_grad_alignment(self) -> float:
        alignments: list[float] = []
        for group in self.param_groups:
            if group["kind"] != "matrix_candidate":
                continue
            params = [param for param in group["params"] if param.grad is not None]
            if not params:
                continue
            anchor = params[0]
            state = self.state[anchor]
            grads = torch.stack([param.grad.detach() for param in params]).float().flatten(1)
            current_mean = grads.mean(dim=0)
            prev_mean = state.get("prev_grad_mean")
            if prev_mean is not None:
                denom = current_mean.norm().item() * prev_mean.norm().item()
                if denom > 0:
                    alignments.append(float(torch.dot(current_mean, prev_mean).item() / denom))
            state["prev_grad_mean"] = current_mean.detach()
        if not alignments:
            return self.training_signals.grad_alignment_ema
        return sum(alignments) / len(alignments)


def build_candidate_optimizer(
    model: nn.Module,
    spec: MatrixOptimizerSpec,
    *,
    unembedding_lr: float = 0.004,
    embedding_lr: float = 0.2,
    matrix_lr: float = 0.02,
    weight_decay: float = 0.2,
    adam_betas: tuple[float, float] = (0.8, 0.95),
    scalar_lr: float = 0.5,
) -> SpecCandidateOptimizer:
    param_groups = build_nanochat_param_groups(
        model,
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
        adam_betas=adam_betas,
        scalar_lr=scalar_lr,
    )
    return SpecCandidateOptimizer(param_groups, spec)
