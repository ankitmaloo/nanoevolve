from __future__ import annotations

import torch

from adamopt.optim_search.candidate_optimizer import ToyNanoChatModel, build_candidate_optimizer, build_nanochat_param_groups
from adamopt.optim_search.spec import MatrixOptimizerSpec


def test_baseline_split_matches_expected_grouping() -> None:
    model = ToyNanoChatModel(vocab_size=64, model_dim=32, hidden_dim=64, layers=2)
    groups = build_nanochat_param_groups(model)
    kinds = [group["kind"] for group in groups]
    assert kinds.count("adamw") >= 5
    assert kinds.count("matrix_candidate") >= 1


def test_candidate_optimizer_step_keeps_parameters_finite() -> None:
    torch.manual_seed(7)
    model = ToyNanoChatModel(vocab_size=64, model_dim=32, hidden_dim=64, layers=2)
    optimizer = build_candidate_optimizer(model, MatrixOptimizerSpec.baseline_nanochat())

    x = torch.randint(0, 64, (4, 12))
    y = torch.randint(0, 64, (4, 12))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.set_step_context(loss_value=float(loss.detach().item()), step=1, total_steps=1)
    optimizer.step()

    for param in model.parameters():
        assert torch.isfinite(param).all()
    assert optimizer.estimated_state_bytes > 0
    assert optimizer.last_step_stats.groups_touched >= 1


def test_trust_ratio_variant_instantiates() -> None:
    model = ToyNanoChatModel(vocab_size=64, model_dim=32, hidden_dim=64, layers=2)
    spec = MatrixOptimizerSpec.trust_ratio_variant()
    optimizer = build_candidate_optimizer(model, spec)
    assert optimizer.spec.trust_ratio.mode == "layerwise"


def test_stateful_annealing_variant_tracks_training_state_and_gate() -> None:
    torch.manual_seed(13)
    model = ToyNanoChatModel(vocab_size=64, model_dim=32, hidden_dim=64, layers=2)
    spec = MatrixOptimizerSpec.stateful_annealing_variant()
    optimizer = build_candidate_optimizer(model, spec)

    x = torch.randint(0, 64, (4, 12))
    y = torch.randint(0, 64, (4, 12))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.set_step_context(loss_value=float(loss.detach().item()), step=3, total_steps=10)
    optimizer.step()

    assert optimizer.spec.stateful_control.enabled is True
    assert optimizer.training_signals.loss_ema > 0.0
    assert optimizer.training_signals.step_fraction > 0.0
    assert 0.0 <= optimizer.last_step_stats.mean_gate <= 1.0
    assert optimizer.last_step_stats.groups_touched >= 1


def test_stateful_spec_round_trip_preserves_gate_coefficients() -> None:
    spec = MatrixOptimizerSpec.stateful_annealing_variant()
    restored = MatrixOptimizerSpec.from_dict(spec.to_dict())

    assert restored.stateful_control.enabled is True
    assert restored.stateful_control.gate.coefficients.grad_alignment_ema == spec.stateful_control.gate.coefficients.grad_alignment_ema
    assert restored.stateful_control.actuators.orthogonal_mix.conservative == spec.stateful_control.actuators.orthogonal_mix.conservative
