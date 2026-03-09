from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class PromptCandidate:
    id: str
    text: str
    uses: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.uses == 0:
            return 0.0
        return self.reward_sum / self.uses


class PromptEvolver:
    """Simple prompt co-evolution layer for mutation instructions."""

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self._counter = 0
        self._bank = [
            "Prioritize validity first, then optimize score.",
            "Prefer compact edits affecting only EVOLVE blocks.",
            "When uncertain, preserve prior strengths and only fix observed failures.",
            "Bias toward improving robustness across different scenarios.",
            "Avoid brittle constants that only help one case.",
            "Prefer changes that reduce pathological edge-case behavior.",
        ]
        seeds = [
            "Be conservative: target small, valid, high-confidence edits.",
            "Be exploratory: propose one bold but coherent scoring change.",
            "Focus on balancing efficiency and solution quality trade-offs.",
            "Use previous failures as hard constraints to avoid repeating mistakes.",
        ]
        self._prompts: dict[str, PromptCandidate] = {}
        for s in seeds:
            self._add_prompt(s)

    def _add_prompt(self, text: str) -> str:
        self._counter += 1
        pid = f"prompt_{self._counter:03d}"
        self._prompts[pid] = PromptCandidate(id=pid, text=text)
        return pid

    def sample(self) -> PromptCandidate:
        prompts = list(self._prompts.values())
        weights: list[float] = []
        for p in prompts:
            if p.uses == 0:
                weights.append(1.5)
            else:
                # Shift mean reward to positive and keep exploration alive.
                weights.append(max(0.05, p.mean_reward + 1.0))
        return self._rng.choices(prompts, weights=weights, k=1)[0]

    def update(self, prompt_id: str, reward: float) -> None:
        p = self._prompts[prompt_id]
        p.uses += 1
        p.reward_sum += reward

    def evolve_population(self, max_prompts: int = 12) -> None:
        if len(self._prompts) >= max_prompts:
            return

        ranked = sorted(self._prompts.values(), key=lambda p: p.mean_reward, reverse=True)
        if len(ranked) < 2:
            return

        a = ranked[0]
        b = ranked[1]
        a_parts = [s.strip() for s in a.text.split(".") if s.strip()]
        b_parts = [s.strip() for s in b.text.split(".") if s.strip()]

        child_parts: list[str] = []
        child_parts.extend(a_parts[:1])
        child_parts.extend(b_parts[:1])
        child_parts.append(self._rng.choice(self._bank))

        child_text = ". ".join(child_parts).strip()
        if not child_text.endswith("."):
            child_text += "."

        self._add_prompt(child_text)

    def top_prompts(self, k: int = 3) -> list[dict[str, object]]:
        ranked = sorted(self._prompts.values(), key=lambda p: p.mean_reward, reverse=True)
        result: list[dict[str, object]] = []
        for p in ranked[:k]:
            result.append({
                "id": p.id,
                "mean_reward": p.mean_reward,
                "uses": p.uses,
                "text": p.text,
            })
        return result
