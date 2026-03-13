from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .candidate_optimizer import ToyNanoChatModel, build_candidate_optimizer
from .config import ComparisonConfig, EvaluationConfig
from .spec import MatrixOptimizerSpec
from .types import CurvePoint, EvaluationOutcome, StepTelemetry, TrialMetrics


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_batch(
    *,
    base_seed: int,
    step: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    split: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    offset = 0 if split == "train" else 10_000
    generator = torch.Generator(device="cpu")
    generator.manual_seed(base_seed * 100_003 + offset + step)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)
    prev = torch.roll(x, shifts=1, dims=1)
    pos = torch.arange(seq_len).unsqueeze(0)
    y = (3 * x + 2 * prev + pos) % vocab_size
    return x.to(device), y.to(device)


def _loss_to_bpb(loss: torch.Tensor) -> float:
    return float(loss.item() / math.log(2.0))


def _curve_auc(points: list[CurvePoint], attr: str) -> float:
    filtered = [(point.step, getattr(point, attr)) for point in points if getattr(point, attr) is not None]
    if len(filtered) < 2:
        return filtered[0][1] if filtered else float("inf")
    area = 0.0
    for (x0, y0), (x1, y1) in zip(filtered, filtered[1:]):
        area += (x1 - x0) * (float(y0) + float(y1)) * 0.5
    return area


class ToyNanoChatBackend:
    def __init__(self, eval_config: EvaluationConfig) -> None:
        self.eval_config = eval_config

    def build_model(self, seed: int) -> ToyNanoChatModel:
        _seed_everything(seed)
        return ToyNanoChatModel(
            vocab_size=self.eval_config.vocab_size,
            model_dim=self.eval_config.model_dim,
            hidden_dim=self.eval_config.hidden_dim,
            layers=self.eval_config.layers,
        ).to(self.eval_config.device)

    @torch.no_grad()
    def _evaluate_split(self, model: ToyNanoChatModel, seed: int, split: str, batches: int) -> float:
        losses: list[float] = []
        model.eval()
        for index in range(batches):
            x, y = _make_batch(
                base_seed=seed,
                step=index,
                batch_size=self.eval_config.batch_size,
                seq_len=self.eval_config.seq_len,
                vocab_size=self.eval_config.vocab_size,
                split=split,
                device=self.eval_config.device,
            )
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(_loss_to_bpb(loss))
        return sum(losses) / len(losses)

    def evaluate(self, spec: MatrixOptimizerSpec, *, seed: int, candidate_id: str) -> EvaluationOutcome:
        _seed_everything(seed)
        model = self.build_model(seed)
        optimizer = build_candidate_optimizer(model, spec)

        curve: list[CurvePoint] = []
        telemetry: list[StepTelemetry] = []
        step_times_ms: list[float] = []
        update_ratios: list[float] = []
        max_ratio = 0.0
        grad_norm_spikes = 0
        running_grad_norm = 0.0
        nan_failures = 0
        inf_failures = 0
        failure_type: str | None = None

        for step in range(1, self.eval_config.steps + 1):
            model.train()
            x, y = _make_batch(
                base_seed=seed,
                step=step,
                batch_size=self.eval_config.batch_size,
                seq_len=self.eval_config.seq_len,
                vocab_size=self.eval_config.vocab_size,
                split="train",
                device=self.eval_config.device,
            )
            start = time.perf_counter()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if not torch.isfinite(loss):
                failure_type = "non_finite_loss"
                nan_failures += int(torch.isnan(loss).item())
                inf_failures += int(torch.isinf(loss).item())
                break

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.set_step_context(
                loss_value=float(loss.detach().item()),
                step=step,
                total_steps=self.eval_config.steps,
            )

            grad_sq = 0.0
            for param in model.parameters():
                if param.grad is None:
                    continue
                grad_sq += float(param.grad.float().pow(2).sum().item())
            grad_norm = math.sqrt(grad_sq)
            if running_grad_norm > 0 and grad_norm > self.eval_config.grad_spike_factor * running_grad_norm:
                grad_norm_spikes += 1
            running_grad_norm = grad_norm if running_grad_norm == 0 else 0.95 * running_grad_norm + 0.05 * grad_norm

            optimizer.step()
            step_times_ms.append((time.perf_counter() - start) * 1000.0)
            update_ratios.append(optimizer.last_step_stats.mean_update_param_ratio)
            max_ratio = max(max_ratio, optimizer.last_step_stats.max_update_param_ratio)

            telemetry.append(optimizer.snapshot_telemetry(
                step=step,
                loss=float(loss.detach().item()),
                grad_norm=grad_norm,
            ))

            invalid_params = False
            for param in model.parameters():
                if torch.isnan(param).any():
                    nan_failures += 1
                    invalid_params = True
                    break
                if torch.isinf(param).any():
                    inf_failures += 1
                    invalid_params = True
                    break
            if invalid_params:
                failure_type = "non_finite_parameters"
                break

            train_bpb = _loss_to_bpb(loss.detach())
            point = CurvePoint(step=step, train_bpb=train_bpb, tokens_seen=step * self.eval_config.tokens_per_step)
            if step % self.eval_config.eval_every == 0 or step == self.eval_config.steps:
                point.val_bpb = self._evaluate_split(model, seed, "val", self.eval_config.val_eval_batches)
            curve.append(point)

        if not curve:
            return EvaluationOutcome(
                candidate_id=candidate_id,
                spec_name=spec.name,
                seed=seed,
                valid=False,
                failure_type=failure_type or "empty_run",
                diagnostics={"spec": spec.to_dict()},
            )

        val_points = [point.val_bpb for point in curve if point.val_bpb is not None]
        final_validation_bpb = float(val_points[-1] if val_points else curve[-1].train_bpb)
        best_validation_bpb = float(min(val_points) if val_points else final_validation_bpb)
        mean_step_time_ms = sum(step_times_ms) / max(1, len(step_times_ms))
        tokens_per_sec = (self.eval_config.token_budget / (sum(step_times_ms) / 1000.0)) if step_times_ms else 0.0
        stability_penalty = float((nan_failures + inf_failures) * 10 + grad_norm_spikes)

        metrics = TrialMetrics(
            final_validation_bpb=final_validation_bpb,
            best_validation_bpb=best_validation_bpb,
            train_curve_auc=_curve_auc(curve, "train_bpb"),
            validation_curve_auc=_curve_auc(curve, "val_bpb"),
            mean_step_time_ms=mean_step_time_ms,
            tokens_per_sec=tokens_per_sec,
            nan_failures=nan_failures,
            inf_failures=inf_failures,
            grad_norm_spikes=grad_norm_spikes,
            max_grad_norm=optimizer.last_step_stats.max_grad_norm,
            mean_update_param_ratio=(sum(update_ratios) / len(update_ratios)) if update_ratios else 0.0,
            max_update_param_ratio=max_ratio,
            memory_overhead_bytes=optimizer.estimated_state_bytes,
            stability_penalty=stability_penalty,
        )
        return EvaluationOutcome(
            candidate_id=candidate_id,
            spec_name=spec.name,
            seed=seed,
            valid=failure_type is None,
            metrics=metrics,
            curve=curve,
            telemetry=telemetry,
            failure_type=failure_type,
            diagnostics={"spec": spec.to_dict(), "token_budget": self.eval_config.token_budget},
        )


def compare_baseline_candidate(
    *,
    backend: ToyNanoChatBackend,
    baseline_spec: MatrixOptimizerSpec,
    candidate_spec: MatrixOptimizerSpec,
    evaluation_config: EvaluationConfig,
    comparison_config: ComparisonConfig | None = None,
) -> dict[str, object]:
    comparison_config = comparison_config or ComparisonConfig()
    baseline = backend.evaluate(baseline_spec, seed=evaluation_config.seed, candidate_id=comparison_config.baseline_label)
    candidate = backend.evaluate(candidate_spec, seed=evaluation_config.seed, candidate_id=comparison_config.candidate_label)

    baseline_metrics = asdict(baseline.metrics) if baseline.metrics else {}
    candidate_metrics = asdict(candidate.metrics) if candidate.metrics else {}

    deltas = {}
    if baseline.metrics and candidate.metrics:
        deltas = {
            "validation_bpb_improvement": baseline.metrics.final_validation_bpb - candidate.metrics.final_validation_bpb,
            "step_time_ratio": candidate.metrics.mean_step_time_ms / max(baseline.metrics.mean_step_time_ms, 1e-8),
            "memory_ratio": candidate.metrics.memory_overhead_bytes / max(baseline.metrics.memory_overhead_bytes, 1),
            "tokens_per_sec_ratio": candidate.metrics.tokens_per_sec / max(baseline.metrics.tokens_per_sec, 1e-8),
        }

    return {
        "seed": evaluation_config.seed,
        "token_budget": evaluation_config.token_budget,
        "baseline_spec": baseline_spec.to_dict(),
        "candidate_spec": candidate_spec.to_dict(),
        "baseline": {
            "valid": baseline.valid,
            "failure_type": baseline.failure_type,
            "metrics": baseline_metrics,
            "curve": [asdict(point) for point in baseline.curve],
        },
        "candidate": {
            "valid": candidate.valid,
            "failure_type": candidate.failure_type,
            "metrics": candidate_metrics,
            "curve": [asdict(point) for point in candidate.curve],
        },
        "deltas": deltas,
        "notes": comparison_config.notes,
    }


def write_metrics_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
