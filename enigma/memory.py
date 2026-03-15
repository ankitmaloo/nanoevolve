"""Persistent search memory — the brain of the evolution engine.

Maintains structured state across loops: variables, hypotheses, gaps,
portfolio, negative knowledge, mutation ledger, and loop log.
All state is serializable to JSON for cold-start handoff to another LLM.
"""
from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from enigma.types import (
    AttackSurface,
    ExperimentRole,
    GapEntry,
    Hypothesis,
    HypothesisStatus,
    LoopRecord,
    MutationNeighborhood,
    NegativeKnowledge,
    PortfolioSlot,
    SurfaceStatus,
)


def _frozen_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Reconstruct a frozen dataclass, ignoring unknown keys."""
    valid_keys = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


class SearchMemory:
    """Structured, persistent search state for the Enigma doctrine loop.

    Manages all the 'md files' from the playbook as structured data:
    - variables (task snapshot, constraints, bottlenecks)
    - attack surfaces
    - hypotheses
    - gaps
    - portfolio
    - negative knowledge
    - mutation ledger
    - loop log
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Task-level variables (free-form dict, mirrors variables.md)
        self.variables: dict[str, Any] = {}

        # Attack surfaces
        self.surfaces: dict[str, AttackSurface] = {}

        # Hypotheses
        self.hypotheses: dict[str, Hypothesis] = {}

        # Gap entries
        self.gaps: dict[str, GapEntry] = {}

        # Active portfolio
        self.portfolio: list[PortfolioSlot] = []

        # Negative knowledge
        self.negative_knowledge: dict[str, NegativeKnowledge] = {}

        # Mutation ledger
        self.mutation_ledger: dict[str, MutationNeighborhood] = {}

        # Loop log
        self.loop_log: list[LoopRecord] = []

        # Current loop counter
        self.current_loop: int = 0

        # Load existing state if present
        self._load()

    # ── Persistence ─────────────────────────────────────────────

    def save(self) -> None:
        """Persist all memory to disk."""
        state = {
            "version": 1,
            "current_loop": self.current_loop,
            "saved_at": datetime.now().isoformat(),
            "variables": self.variables,
            "surfaces": {k: asdict(v) for k, v in self.surfaces.items()},
            "hypotheses": {k: asdict(v) for k, v in self.hypotheses.items()},
            "gaps": {k: asdict(v) for k, v in self.gaps.items()},
            "portfolio": [asdict(s) for s in self.portfolio],
            "negative_knowledge": {k: asdict(v) for k, v in self.negative_knowledge.items()},
            "mutation_ledger": {k: asdict(v) for k, v in self.mutation_ledger.items()},
            "loop_log": [asdict(r) for r in self.loop_log],
        }
        path = self.state_dir / "search_memory.json"
        path.write_text(json.dumps(state, indent=2, default=str))

    def _load(self) -> None:
        """Load state from disk if it exists."""
        path = self.state_dir / "search_memory.json"
        if not path.exists():
            return

        state = json.loads(path.read_text())
        self.current_loop = state.get("current_loop", 0)
        self.variables = state.get("variables", {})

        for k, v in state.get("surfaces", {}).items():
            v["status"] = SurfaceStatus(v["status"])
            self.surfaces[k] = _frozen_from_dict(AttackSurface, v)

        for k, v in state.get("hypotheses", {}).items():
            v["status"] = HypothesisStatus(v["status"])
            self.hypotheses[k] = _frozen_from_dict(Hypothesis, v)

        for k, v in state.get("gaps", {}).items():
            self.gaps[k] = _frozen_from_dict(GapEntry, v)

        self.portfolio = []
        for s in state.get("portfolio", []):
            s["role"] = ExperimentRole(s["role"])
            self.portfolio.append(_frozen_from_dict(PortfolioSlot, s))

        for k, v in state.get("negative_knowledge", {}).items():
            self.negative_knowledge[k] = _frozen_from_dict(NegativeKnowledge, v)

        for k, v in state.get("mutation_ledger", {}).items():
            self.mutation_ledger[k] = _frozen_from_dict(MutationNeighborhood, v)

        self.loop_log = []
        for r in state.get("loop_log", []):
            self.loop_log.append(LoopRecord(**r))

    # ── Query helpers ───────────────────────────────────────────

    def active_hypotheses(self) -> list[Hypothesis]:
        return [
            h for h in self.hypotheses.values()
            if h.status in (HypothesisStatus.SHORTLISTED, HypothesisStatus.ACTIVE)
        ]

    def killed_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses.values() if h.status == HypothesisStatus.KILLED]

    def unexplored_surfaces(self) -> list[AttackSurface]:
        return [s for s in self.surfaces.values() if s.status == SurfaceStatus.UNEXPLORED]

    def active_neighborhoods(self) -> list[MutationNeighborhood]:
        return [n for n in self.mutation_ledger.values() if not n.retired]

    def retired_neighborhoods(self) -> list[MutationNeighborhood]:
        return [n for n in self.mutation_ledger.values() if n.retired]

    def high_urgency_gaps(self) -> list[GapEntry]:
        return [g for g in self.gaps.values() if g.urgency == "high"]

    def negative_knowledge_for_family(self, family: str) -> list[NegativeKnowledge]:
        return [
            nk for nk in self.negative_knowledge.values()
            if nk.move_family == family
        ]

    def last_loop(self) -> LoopRecord | None:
        return self.loop_log[-1] if self.loop_log else None

    # ── Mutation helpers ────────────────────────────────────────

    def next_hypothesis_id(self) -> str:
        existing = [int(h_id[1:]) for h_id in self.hypotheses if h_id[0] == "H" and h_id[1:].isdigit()]
        n = max(existing, default=0) + 1
        return f"H{n:02d}"

    def next_surface_id(self) -> str:
        existing = [int(s_id[1:]) for s_id in self.surfaces if s_id[0] == "S" and s_id[1:].isdigit()]
        n = max(existing, default=0) + 1
        return f"S{n:02d}"

    def next_gap_id(self) -> str:
        existing = [int(g_id[1:]) for g_id in self.gaps if g_id[0] == "G" and g_id[1:].isdigit()]
        n = max(existing, default=0) + 1
        return f"G{n:02d}"

    def next_nk_id(self) -> str:
        existing_ids = [
            int(nk_id[2:]) for nk_id in self.negative_knowledge
            if nk_id[:2] == "NK" and nk_id[2:].isdigit()
        ]
        n = max(existing_ids, default=0) + 1
        return f"NK{n:02d}"

    def next_neighborhood_id(self) -> str:
        existing = [int(m_id[1:]) for m_id in self.mutation_ledger if m_id[0] == "M" and m_id[1:].isdigit()]
        n = max(existing, default=0) + 1
        return f"M{n:02d}"

    # ── Overlap detection ───────────────────────────────────────

    def check_neighborhood_overlap(
        self,
        target_region: str,
        mechanism: str,
        bottleneck: str,
        benchmark_slice: str,
    ) -> list[MutationNeighborhood]:
        """Find active neighborhoods that overlap with the given parameters."""
        overlaps = []
        for n in self.active_neighborhoods():
            matches = 0
            if n.target_region == target_region:
                matches += 1
            if n.bottleneck_attacked == bottleneck:
                matches += 1
            if n.operator_family == mechanism:
                matches += 1
            if n.benchmark_slice == benchmark_slice:
                matches += 1
            if matches >= 3:
                overlaps.append(n)
        return overlaps

    # ── Export for LLM context ──────────────────────────────────

    def export_for_prompt(self, max_tokens: int = 4000) -> str:
        """Export search memory as a compact text block for LLM prompts."""
        sections: list[str] = []

        # Negative knowledge first — most important to avoid repeating mistakes
        if self.negative_knowledge:
            nk_lines = ["## Negative Knowledge (do NOT repeat these)"]
            for nk in self.negative_knowledge.values():
                nk_lines.append(
                    f"- {nk.id}: {nk.move_family} — {nk.observed_failure}. "
                    f"Cause: {nk.likely_cause}. "
                    f"Do not repeat unless: {nk.do_not_repeat_unless}"
                )
            sections.append("\n".join(nk_lines))

        # Active hypotheses
        active = self.active_hypotheses()
        if active:
            h_lines = ["## Active Hypotheses"]
            for h in active:
                h_lines.append(
                    f"- {h.id} [{h.status.value}]: {h.title} "
                    f"(family={h.family}, bottleneck={h.bottleneck_attacked})"
                )
            sections.append("\n".join(h_lines))

        # Current portfolio
        if self.portfolio:
            p_lines = ["## Current Portfolio"]
            for s in self.portfolio:
                p_lines.append(
                    f"- {s.id} [{s.role.value}]: hypothesis={s.hypothesis_id}, "
                    f"region={s.target_region}, result={s.result}"
                )
            sections.append("\n".join(p_lines))

        # Active neighborhoods
        active_n = self.active_neighborhoods()
        if active_n:
            n_lines = ["## Active Mutation Neighborhoods"]
            for n in active_n:
                n_lines.append(
                    f"- {n.id}: region={n.target_region}, "
                    f"operator={n.operator_family}, outcome={n.outcome}"
                )
            sections.append("\n".join(n_lines))

        # High-urgency gaps
        urgent = self.high_urgency_gaps()
        if urgent:
            g_lines = ["## High-Urgency Gaps"]
            for g in urgent:
                g_lines.append(f"- {g.id}: {g.uncovered_area} — {g.why_it_matters}")
            sections.append("\n".join(g_lines))

        # Last loop summary
        last = self.last_loop()
        if last:
            sections.append(
                f"## Last Loop ({last.loop})\n"
                f"Best: {last.best_candidate}, Score: {last.score_movement}\n"
                f"Learned: {last.strongest_evidence}\n"
                f"Next focus: {last.next_loop_focus}"
            )

        text = "\n\n".join(sections)

        # Rough token estimate: ~4 chars per token
        if len(text) > max_tokens * 4:
            text = text[: max_tokens * 4] + "\n... (truncated)"

        return text
