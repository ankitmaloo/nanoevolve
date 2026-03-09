from __future__ import annotations

from adamopt.optim_search.config import EvaluationConfig
from adamopt.optim_search.eval_candidate import ToyNanoChatBackend
from adamopt.optim_search.spec import MatrixOptimizerSpec


def test_same_seed_reproduces_curve_and_quality_metrics() -> None:
    config = EvaluationConfig(seed=11, steps=12, eval_every=4, batch_size=4, seq_len=12, model_dim=24, hidden_dim=48, layers=2)
    backend = ToyNanoChatBackend(config)
    spec = MatrixOptimizerSpec.baseline_nanochat()

    run_a = backend.evaluate(spec, seed=config.seed, candidate_id="a")
    run_b = backend.evaluate(spec, seed=config.seed, candidate_id="b")

    assert run_a.valid == run_b.valid
    assert run_a.failure_type == run_b.failure_type
    assert [point.train_bpb for point in run_a.curve] == [point.train_bpb for point in run_b.curve]
    assert [point.val_bpb for point in run_a.curve] == [point.val_bpb for point in run_b.curve]
    assert run_a.metrics is not None and run_b.metrics is not None
    assert run_a.metrics.final_validation_bpb == run_b.metrics.final_validation_bpb
    assert run_a.metrics.best_validation_bpb == run_b.metrics.best_validation_bpb


def test_large_update_multiplier_is_still_reported_cleanly() -> None:
    config = EvaluationConfig(seed=5, steps=8, eval_every=4, batch_size=4, seq_len=10)
    backend = ToyNanoChatBackend(config)
    spec = MatrixOptimizerSpec.baseline_nanochat().from_dict(
        {
            **MatrixOptimizerSpec.baseline_nanochat().to_dict(),
            "name": "aggressive",
            "update_multiplier": 3.0,
        }
    )
    run = backend.evaluate(spec, seed=config.seed, candidate_id="aggressive")
    assert run.metrics is not None or run.failure_type is not None


def test_stateful_annealing_variant_is_reproducible_for_same_seed() -> None:
    config = EvaluationConfig(seed=17, steps=12, eval_every=4, batch_size=4, seq_len=12, model_dim=24, hidden_dim=48, layers=2)
    backend = ToyNanoChatBackend(config)
    spec = MatrixOptimizerSpec.stateful_annealing_variant()

    run_a = backend.evaluate(spec, seed=config.seed, candidate_id="stateful_a")
    run_b = backend.evaluate(spec, seed=config.seed, candidate_id="stateful_b")

    assert run_a.valid == run_b.valid
    assert run_a.metrics is not None and run_b.metrics is not None
    assert run_a.metrics.final_validation_bpb == run_b.metrics.final_validation_bpb
    assert run_a.metrics.best_validation_bpb == run_b.metrics.best_validation_bpb
    assert [point.train_bpb for point in run_a.curve] == [point.train_bpb for point in run_b.curve]
