from __future__ import annotations

from mvp.population_db import PopulationDB
from mvp.types import Candidate


def _candidate(
    cid: str,
    aggregate: float,
    active: bool = True,
    descriptor: str = "len:1|spread:1",
    failures: list[str] | None = None,
) -> Candidate:
    return Candidate(
        id=cid,
        generation=0,
        parent_id=None,
        source="def x():\n    return 1\n",
        aggregate_score=aggregate,
        metrics={"aggregate_score": aggregate, "placed_jobs_ratio": aggregate},
        failure_reasons=failures or [],
        stage_results=[],
        active=active,
        meta={"descriptor": descriptor},
    )


def test_feedback_bundle_includes_failure_reasons() -> None:
    db = PopulationDB(seed=1)
    db.add(_candidate("c1", 0.5, failures=["bad diff format"]))
    db.add(_candidate("c2", 0.7, failures=["runtime error"]))

    feedback = db.build_feedback_bundle()

    assert "aggregate_score" in feedback.best_metrics
    assert any("runtime" in reason for reason in feedback.weak_failure_reasons)


def test_prune_survivors_keeps_top_and_diverse() -> None:
    db = PopulationDB(seed=1)
    db.add(_candidate("top", 0.9, descriptor="len:1|spread:1"))
    db.add(_candidate("second", 0.8, descriptor="len:1|spread:1"))
    db.add(_candidate("diverse", 0.7, descriptor="len:9|spread:5"))

    survivors, dropped = db.prune_survivors(top_k=2, diversity_slots=1)

    assert "top" in survivors
    assert "second" in survivors
    assert "diverse" in survivors
    assert dropped == []
