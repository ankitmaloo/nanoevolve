from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


@dataclass
class SlotState:
    slot: int
    parent_id: str = "-"
    child_id: str = "-"
    metric_key: str = "-"
    status: str = "idle"
    model: str = "-"
    score: str = "-"
    valid: str = "-"
    note: str = "-"


@dataclass
class GenerationState:
    generation: int
    status: str = "pending"
    evaluated: int = 0
    valid: int = 0
    invalid: int = 0
    best_candidate_id: str = "-"
    best_score: float = float("-inf")


class EvolutionLiveTUI:
    """Lightweight live dashboard for the AlphaEvolve loop."""

    def __init__(
        self,
        *,
        mode: str,
        model_name: str,
        total_generations: int,
        parallel_candidates: int,
    ) -> None:
        self.mode = mode
        self.model_name = model_name
        self.total_generations = total_generations
        self.parallel_candidates = parallel_candidates

        self.console = Console()
        self.is_interactive = bool(self.console.is_terminal and not getattr(self.console, "is_dumb_terminal", False))
        self.live: Live | None = None

        self.current_generation = 0
        self.total_candidates_seen = 0
        self.valid_candidates = 0
        self.invalid_candidates = 0
        self.best_candidate_id = "-"
        self.best_score = float("-inf")
        self.recent_logs: list[str] = []
        self.slot_states = {slot: SlotState(slot=slot) for slot in range(parallel_candidates)}
        self.generation_states = {
            generation: GenerationState(generation=generation)
            for generation in range(0, total_generations + 1)
        }
        self.generation_states[0].status = "waiting_seed"

    def start(self) -> None:
        if self.is_interactive:
            self.live = Live(self._render(), console=self.console, refresh_per_second=8, transient=False)
            self.live.start()
        else:
            self.console.print(
                f"[tui] mode={self.mode} model={self.model_name} gens={self.total_generations} "
                f"parallel={self.parallel_candidates}",
                markup=False,
            )

    def stop(self) -> None:
        if self.live is not None:
            self.live.update(self._render(), refresh=True)
            self.live.stop()
            self.live = None

    def handle_event(self, event: dict[str, object]) -> None:
        event_type = str(event.get("event_type", ""))

        if event_type == "run_start":
            self._log(f"run start: mode={event.get('mode')} model={event.get('model_name')}")

        elif event_type == "seed_evaluated":
            seed_candidate_id = str(event.get("candidate_id", "-"))
            seed_valid = bool(event.get("valid", False))
            seed_score = float(event.get("aggregate_score", -1.0))
            self._register_candidate(
                candidate_id=seed_candidate_id,
                valid=seed_valid,
                aggregate_score=seed_score,
            )
            g0 = self.generation_states[0]
            g0.status = "done"
            g0.evaluated = 1
            g0.valid = 1 if seed_valid else 0
            g0.invalid = 0 if seed_valid else 1
            g0.best_candidate_id = seed_candidate_id
            g0.best_score = seed_score
            self._log(
                f"seed {event.get('candidate_id')} score={float(event.get('aggregate_score', -1.0)):.4f}"
            )

        elif event_type == "generation_start":
            self.current_generation = int(event.get("generation", 0))
            if self.current_generation in self.generation_states:
                self.generation_states[self.current_generation].status = "running"
            for slot_state in self.slot_states.values():
                slot_state.status = "idle"
                slot_state.note = "-"
            self._log(f"generation {self.current_generation} started")

        elif event_type == "slot_queued":
            slot = int(event.get("slot", 0))
            state = self.slot_states[slot]
            state.parent_id = str(event.get("parent_id", "-"))
            state.child_id = str(event.get("child_id", "-"))
            state.metric_key = str(event.get("metric_key", "-"))
            state.status = "queued"
            state.model = "-"
            state.score = "-"
            state.valid = "-"
            state.note = "-"

        elif event_type == "slot_mutation_start":
            slot = int(event.get("slot", 0))
            self.slot_states[slot].status = "mutating"

        elif event_type == "slot_mutation_done":
            slot = int(event.get("slot", 0))
            state = self.slot_states[slot]
            state.status = "mutated"
            state.model = str(event.get("mutation_model", "-"))

        elif event_type == "slot_mutation_error":
            slot = int(event.get("slot", 0))
            state = self.slot_states[slot]
            state.status = "mutation_error"
            state.note = self._truncate(str(event.get("error", "")), 80)

        elif event_type == "slot_evaluated":
            slot = int(event.get("slot", 0))
            score = float(event.get("aggregate_score", -1.0))
            valid = bool(event.get("valid", False))
            candidate_id = str(event.get("candidate_id", "-"))
            state = self.slot_states[slot]
            state.status = "evaluated"
            state.score = f"{score:.4f}"
            state.valid = "yes" if valid else "no"
            state.model = str(event.get("mutation_model", state.model))
            state.note = self._truncate(str(event.get("error", "")), 80) if not valid else "-"
            self._register_candidate(candidate_id=candidate_id, valid=valid, aggregate_score=score)
            generation = int(event.get("generation", self.current_generation))
            gen_state = self.generation_states.get(generation)
            if gen_state is not None:
                gen_state.evaluated += 1
                if valid:
                    gen_state.valid += 1
                else:
                    gen_state.invalid += 1
                if score > gen_state.best_score:
                    gen_state.best_score = score
                    gen_state.best_candidate_id = candidate_id

        elif event_type == "generation_end":
            gen = int(event.get("generation", 0))
            best = float(event.get("best_aggregate_score", -1.0))
            gen_state = self.generation_states.get(gen)
            if gen_state is not None:
                gen_state.status = "done"
            self._log(f"generation {gen} end: best={best:.4f}")

        elif event_type == "run_end":
            self._log(
                "run end: "
                f"best={event.get('best_candidate_id')} score={float(event.get('best_aggregate_score', -1.0)):.4f}"
            )

        if self.live is not None:
            self.live.update(self._render(), refresh=True)
        elif event_type in {"generation_start", "slot_evaluated", "run_end"}:
            self.console.print(
                "[tui] "
                f"gen={self.current_generation}/{self.total_generations} "
                f"seen={self.total_candidates_seen} valid={self.valid_candidates} "
                f"best={self.best_candidate_id}:{self.best_score:.4f}",
                markup=False,
            )

    def _register_candidate(self, candidate_id: str, valid: bool, aggregate_score: float) -> None:
        self.total_candidates_seen += 1
        if valid:
            self.valid_candidates += 1
        else:
            self.invalid_candidates += 1
        if aggregate_score > self.best_score:
            self.best_score = aggregate_score
            self.best_candidate_id = candidate_id

    def _log(self, text: str) -> None:
        self.recent_logs.append(text)
        self.recent_logs = self.recent_logs[-8:]

    def _render(self) -> Group:
        summary = Table.grid(expand=True)
        summary.add_column()
        summary.add_column()
        summary.add_column()
        summary.add_column()
        summary.add_row(
            f"mode: [bold]{self.mode}[/bold]",
            f"model: [bold]{self.model_name}[/bold]",
            f"generation: [bold]{self.current_generation}/{self.total_generations}[/bold]",
            f"best: [bold]{self.best_candidate_id} ({self.best_score:.4f})[/bold]",
        )
        summary.add_row(
            f"candidates seen: {self.total_candidates_seen}",
            f"valid: {self.valid_candidates}",
            f"invalid: {self.invalid_candidates}",
            f"parallel slots: {self.parallel_candidates}",
        )

        slot_table = Table(title="Slot Status", expand=True)
        slot_table.add_column("slot", width=4)
        slot_table.add_column("parent")
        slot_table.add_column("child")
        slot_table.add_column("metric")
        slot_table.add_column("status")
        slot_table.add_column("model")
        slot_table.add_column("score", justify="right")
        slot_table.add_column("valid", justify="center")
        slot_table.add_column("note")
        for slot in sorted(self.slot_states):
            state = self.slot_states[slot]
            slot_table.add_row(
                str(slot),
                state.parent_id,
                state.child_id,
                state.metric_key,
                state.status,
                state.model,
                state.score,
                state.valid,
                state.note,
            )

        generation_table = Table(title="Generation History", expand=True)
        generation_table.add_column("gen", width=5)
        generation_table.add_column("status")
        generation_table.add_column("eval")
        generation_table.add_column("valid")
        generation_table.add_column("invalid")
        generation_table.add_column("best candidate")
        generation_table.add_column("best score", justify="right")
        for generation in range(0, self.total_generations + 1):
            state = self.generation_states[generation]
            best_score_str = "-" if state.best_score == float("-inf") else f"{state.best_score:.4f}"
            generation_table.add_row(
                "seed" if generation == 0 else str(generation),
                state.status,
                str(state.evaluated),
                str(state.valid),
                str(state.invalid),
                state.best_candidate_id,
                best_score_str,
            )

        logs = "\n".join(self.recent_logs) if self.recent_logs else "no events yet"
        return Group(
            Panel(summary, title="AlphaEvolve Live"),
            slot_table,
            generation_table,
            Panel(logs, title="Recent Events"),
        )

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text or "-"
        return text[: limit - 3] + "..."
