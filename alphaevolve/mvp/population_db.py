from __future__ import annotations

import random
from collections import Counter

from mvp.types import Candidate, FeedbackBundle


class PopulationDB:
    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self._candidates: list[Candidate] = []
        self._by_id: dict[str, Candidate] = {}

    @property
    def candidates(self) -> list[Candidate]:
        return self._candidates

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
            raise ValueError("No active candidates in population.")
        return max(active, key=lambda c: c.aggregate_score)

    def sample_parent(self, metric_key: str = "aggregate_score") -> Candidate:
        active = self.active_candidates()
        if not active:
            raise ValueError("No active candidates to sample from.")

        values = [float(c.metrics.get(metric_key, c.aggregate_score)) for c in active]
        min_value = min(values)
        # Shift weights to keep them positive and preserve relative quality.
        weights = [v - min_value + 1e-6 for v in values]

        if sum(weights) <= 0:
            return self._rng.choice(active)

        return self._rng.choices(active, weights=weights, k=1)[0]

    def sample_inspirations(self, k: int, exclude_id: str | None = None) -> list[Candidate]:
        active = [c for c in self.active_candidates() if c.id != exclude_id]
        active.sort(key=lambda c: c.aggregate_score, reverse=True)
        return active[:k]

    def build_feedback_bundle(self) -> FeedbackBundle:
        active = self.active_candidates()
        if active:
            best = max(active, key=lambda c: c.aggregate_score)
            best_metrics = dict(best.metrics)
        else:
            best_metrics = {}
        # Fallback: derive weak signals from failure reasons among latest candidates.
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

    def prune_survivors(self, top_k: int, diversity_slots: int = 1) -> tuple[list[str], list[str]]:
        active = self.active_candidates()
        if not active:
            return [], []

        active_sorted = sorted(active, key=lambda c: c.aggregate_score, reverse=True)
        keep = active_sorted[:top_k]
        kept_ids = {c.id for c in keep}

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
                candidate.meta["drop_reason"] = "Not in top-k survivors and not selected for diversity slot"
                dropped_ids.append(candidate.id)

        survivor_ids = [c.id for c in keep]
        return survivor_ids, dropped_ids
