"""Population management with diversity-aware selection and pruning."""
from __future__ import annotations

import random
from collections import Counter

from enigma.types import Candidate, FeedbackBundle


class PopulationDB:
    """Manages the population of candidate programs across evolution loops."""

    def __init__(self, seed: int = 42, maximize: bool = True) -> None:
        self._rng = random.Random(seed)
        self._candidates: list[Candidate] = []
        self._by_id: dict[str, Candidate] = {}
        self._maximize = maximize
        self._counter = 0

    @property
    def candidates(self) -> list[Candidate]:
        return self._candidates

    def next_id(self) -> str:
        self._counter += 1
        return f"cand_{self._counter:04d}"

    def add(self, candidate: Candidate) -> None:
        self._candidates.append(candidate)
        self._by_id[candidate.id] = candidate

    def get(self, candidate_id: str) -> Candidate:
        return self._by_id[candidate_id]

    def active_candidates(self) -> list[Candidate]:
        return [c for c in self._candidates if c.active]

    def best(self) -> Candidate:
        active = self.active_candidates()
        if not active:
            raise ValueError("No active candidates.")
        return max(active, key=lambda c: c.aggregate_score) if self._maximize \
            else min(active, key=lambda c: c.aggregate_score)

    def score_key(self, c: Candidate) -> float:
        return c.aggregate_score if self._maximize else -c.aggregate_score

    def sample_parent(self, metric_key: str = "aggregate_score") -> Candidate:
        active = self.active_candidates()
        if not active:
            raise ValueError("No active candidates to sample from.")

        values = [float(c.metrics.get(metric_key, c.aggregate_score)) for c in active]
        if not self._maximize:
            # For minimization, invert so lower = higher weight
            max_v = max(values) + 1e-6
            values = [max_v - v for v in values]

        min_v = min(values)
        weights = [v - min_v + 1e-6 for v in values]

        if sum(weights) <= 0:
            return self._rng.choice(active)

        return self._rng.choices(active, weights=weights, k=1)[0]

    def sample_inspirations(self, k: int, exclude_id: str | None = None) -> list[Candidate]:
        active = [c for c in self.active_candidates() if c.id != exclude_id]
        active.sort(key=lambda c: self.score_key(c), reverse=True)
        return active[:k]

    def build_feedback_bundle(self) -> FeedbackBundle:
        active = self.active_candidates()
        if active:
            best = max(active, key=lambda c: self.score_key(c))
            best_metrics = dict(best.metrics)
        else:
            best_metrics = {}

        weak_reasons: list[str] = []
        for candidate in reversed(self._candidates[-10:]):
            weak_reasons.extend(candidate.failure_reasons)
        weak_reasons = list(dict.fromkeys(weak_reasons))[:5]

        dropped_notes: list[str] = []
        for candidate in self._candidates[-10:]:
            note = candidate.meta.get("drop_reason")
            if note:
                dropped_notes.append(f"{candidate.id}: {note}")
        dropped_notes = dropped_notes[:5]

        return FeedbackBundle(
            best_metrics=best_metrics,
            weak_failure_reasons=weak_reasons,
            dropped_notes=dropped_notes,
        )

    def prune_survivors(
        self,
        top_k: int,
        diversity_slots: int = 1,
    ) -> tuple[list[str], list[str]]:
        active = self.active_candidates()
        if not active:
            return [], []

        active_sorted = sorted(active, key=lambda c: self.score_key(c), reverse=True)
        keep = active_sorted[:top_k]
        kept_ids = {c.id for c in keep}

        # Diversity: keep candidates with unique descriptors
        if diversity_slots > 0:
            descriptor_counts = Counter(c.meta.get("descriptor") for c in keep)
            rest = [c for c in active_sorted if c.id not in kept_ids]
            chosen_diverse: list[Candidate] = []
            for candidate in rest:
                descriptor = candidate.meta.get("descriptor")
                if descriptor_counts[descriptor] == 0:
                    chosen_diverse.append(candidate)
                    descriptor_counts[descriptor] += 1
                if len(chosen_diverse) >= diversity_slots:
                    break
            keep.extend(chosen_diverse)
            kept_ids = {c.id for c in keep}

        dropped_ids: list[str] = []
        for candidate in active:
            if candidate.id in kept_ids:
                candidate.active = True
            else:
                candidate.active = False
                candidate.meta["drop_reason"] = "Not in top-k or diversity slot"
                dropped_ids.append(candidate.id)

        return [c.id for c in keep], dropped_ids

    def candidates_for_loop(self, loop: int) -> list[Candidate]:
        return [c for c in self._candidates if c.loop == loop]

    def candidates_for_hypothesis(self, hypothesis_id: str) -> list[Candidate]:
        return [c for c in self._candidates if c.hypothesis_id == hypothesis_id]
