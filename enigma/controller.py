"""The Enigma Controller — orchestrates the full doctrine loop.

This is the Bombe machine: it runs the disciplined evolutionary search
cycle from the Enigma Playbook. Each loop:

1.  Map variables (or update from prior loop)
2.  Map attack surfaces
3.  Generate 10 hypotheses
4.  Prune weak/redundant ones
5.  Find gaps
6.  Select top-5 portfolio with scout/exploit/wildcard roles
7.  Register mutation neighborhoods
8.  Implement one candidate per slot
9.  Evaluate candidates
10. Run evaluator skepticism
11. Record postmortem and causal learning
12. Update negative knowledge and search memory
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from enigma.config import ProviderConfig, SearchConfig, TaskConfig
from enigma.context import generate_surface_mapping_prompt, generate_task_context, generate_variables_prompt
from enigma.evaluator import Evaluator
from enigma.memory import SearchMemory
from enigma.mutator import LLMMutator, MutationError, apply_search_replace_blocks
from enigma.population import PopulationDB
from enigma.prompts import (
    build_candidate_implementer_prompt,
    build_evaluator_skeptic_prompt,
    build_gap_finder_prompt,
    build_hypothesis_generator_prompt,
    build_hypothesis_pruner_prompt,
    build_portfolio_selector_prompt,
    build_postmortem_prompt,
)
from enigma.types import (
    AttackSurface,
    Candidate,
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

logger = logging.getLogger("enigma")


class EnigmaController:
    """Orchestrates the full Enigma doctrine loop for any target program."""

    def __init__(
        self,
        task_dir: Path,
        search_config: SearchConfig,
        provider_config: ProviderConfig,
        run_dir: Path | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.task_dir = task_dir.resolve()
        self.task_config = TaskConfig.load(task_dir)
        self.search_config = search_config
        self.provider_config = provider_config
        self.event_callback = event_callback

        # Set up run directory
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.task_dir / "runs" / f"enigma_{timestamp}"
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.diffs_dir = self.run_dir / "diffs"
        self.evals_dir = self.run_dir / "evaluations"
        self.prompts_dir = self.run_dir / "prompts"
        for d in [self.diffs_dir, self.evals_dir, self.prompts_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Core components
        self.memory = SearchMemory(self.run_dir / "memory")
        self.population = PopulationDB(
            seed=search_config.seed,
            maximize=self.task_config.maximize,
        )
        self.evaluator = Evaluator(self.task_config, self.task_dir)
        self.mutator = LLMMutator(provider_config)

        # Events log
        self.events_path = self.run_dir / "events.jsonl"

    # ── Public API ──────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Run the full evolution search."""
        self._emit("run_start", config=asdict(self.search_config))

        # Load and evaluate seed program
        seed_path = self.task_config.resolve_seed_path(self.task_dir)
        seed_source = seed_path.read_text()
        (self.run_dir / f"seed_program{seed_path.suffix}").write_text(seed_source)

        seed_eval = self.evaluator.evaluate(seed_source)
        seed_candidate = Candidate(
            id="cand_0000",
            generation=0,
            loop=0,
            parent_id=None,
            source=seed_source,
            aggregate_score=seed_eval.aggregate_score,
            metrics=seed_eval.metrics,
            failure_reasons=seed_eval.failure_reasons,
            stage_results=seed_eval.stage_results,
            active=seed_eval.valid,
            lineage={"mutation": "seed"},
        )
        self.population.add(seed_candidate)
        self._emit("seed_evaluated", score=seed_eval.aggregate_score, valid=seed_eval.valid)

        # Generate dynamic context
        code_context = generate_task_context(self.task_config, self.task_dir)
        (self.run_dir / "code_context.md").write_text(code_context)

        # Run doctrine loops
        for loop_num in range(1, self.search_config.max_loops + 1):
            self.memory.current_loop = loop_num
            self._emit("loop_start", loop=loop_num)

            try:
                self._run_doctrine_loop(loop_num, code_context, seed_source)
            except Exception as e:
                logger.exception(f"Loop {loop_num} failed: {e}")
                self._emit("loop_error", loop=loop_num, error=str(e))
                # Continue to next loop — resilience over fragility
                continue

            self.memory.save()
            self._emit("loop_end", loop=loop_num)

        # Final summary
        best = self.population.best()
        (self.run_dir / f"best_program{seed_path.suffix}").write_text(best.source)

        summary = {
            "task": self.task_config.name,
            "loops_completed": self.memory.current_loop,
            "total_candidates": len(self.population.candidates),
            "best_candidate_id": best.id,
            "best_score": best.aggregate_score,
            "best_metrics": best.metrics,
            "hypotheses_tested": len(self.memory.hypotheses),
            "negative_knowledge_entries": len(self.memory.negative_knowledge),
            "run_dir": str(self.run_dir),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self._emit("run_end", **summary)
        return summary

    # ── Doctrine Loop ───────────────────────────────────────────

    def _run_doctrine_loop(
        self,
        loop_num: int,
        code_context: str,
        seed_source: str,
    ) -> None:
        """Execute one full doctrine loop (steps 1-11 from the playbook)."""

        # Step 1: Map variables (first loop) or update (subsequent loops)
        if loop_num == 1 or not self.memory.variables:
            self._step_map_variables(code_context)

        variables_json = json.dumps(self.memory.variables, indent=2, default=str)

        # Step 2: Map attack surfaces (first loop or if coverage is thin)
        if loop_num == 1 or len(self.memory.unexplored_surfaces()) < 2:
            self._step_map_surfaces(code_context, variables_json)

        surfaces_json = json.dumps(
            [asdict(s) for s in self.memory.surfaces.values()],
            indent=2, default=str,
        )

        # Step 3: Generate hypotheses
        hypotheses = self._step_generate_hypotheses(
            code_context, variables_json, surfaces_json,
        )

        # Step 4: Prune hypotheses
        self._step_prune_hypotheses(variables_json)

        hypotheses_json = json.dumps(
            [asdict(h) for h in self.memory.active_hypotheses()],
            indent=2, default=str,
        )

        # Step 5: Find gaps
        self._step_find_gaps(hypotheses_json, variables_json, surfaces_json)

        # Step 6: Select portfolio
        portfolio = self._step_select_portfolio(hypotheses_json, variables_json)

        # Steps 7-8: Register neighborhoods and implement candidates
        candidates = self._step_implement_and_evaluate(
            portfolio, code_context, variables_json,
        )

        # Step 9: Evaluate skeptically
        self._step_evaluate_skeptically(portfolio, candidates, variables_json)

        # Steps 10-11: Postmortem and update memory
        self._step_postmortem(portfolio, candidates, variables_json)

        # Prune population
        self.population.prune_survivors(
            top_k=self.search_config.survivor_top_k,
            diversity_slots=self.search_config.diversity_slots,
        )

    # ── Step implementations ────────────────────────────────────

    def _step_map_variables(self, code_context: str) -> None:
        """Step 1: Map the search space variables."""
        self._emit("stage", stage="map_variables")
        prompt = generate_variables_prompt(self.task_config, self.task_dir, code_context)
        self._save_prompt("variables", prompt)

        try:
            result = self.mutator.call_llm_json(prompt, stage="map_variables")
            self.memory.variables = result
        except MutationError as e:
            logger.warning(f"Variable mapping failed: {e}")

    def _step_map_surfaces(self, code_context: str, variables_json: str) -> None:
        """Step 2: Map attack surfaces."""
        self._emit("stage", stage="map_surfaces")
        prompt = generate_surface_mapping_prompt(code_context, variables_json)
        self._save_prompt("surfaces", prompt)

        try:
            result = self.mutator.call_llm_json(prompt, stage="map_surfaces")
            surfaces = result if isinstance(result, list) else result.get("surfaces", [])
            for s_data in surfaces:
                sid = s_data.get("id", self.memory.next_surface_id())
                surface = AttackSurface(
                    id=sid,
                    region=s_data.get("region", ""),
                    bottleneck=s_data.get("bottleneck", ""),
                    change_class=s_data.get("change_class", ""),
                    mechanism=s_data.get("mechanism", ""),
                    leverage=int(s_data.get("leverage", 3)),
                    plausibility=int(s_data.get("plausibility", 3)),
                    observability=int(s_data.get("observability", 3)),
                    implementation_cost=int(s_data.get("implementation_cost", 3)),
                    overlap_risk=int(s_data.get("overlap_risk", 2)),
                    status=SurfaceStatus.UNEXPLORED,
                    notes=s_data.get("notes", ""),
                )
                self.memory.surfaces[sid] = surface
        except MutationError as e:
            logger.warning(f"Surface mapping failed: {e}")

    def _step_generate_hypotheses(
        self,
        code_context: str,
        variables_json: str,
        surfaces_json: str,
    ) -> list[Hypothesis]:
        """Step 3: Generate hypotheses."""
        self._emit("stage", stage="generate_hypotheses")

        nk_text = self._negative_knowledge_text()
        ml_text = self._mutation_ledger_text()

        prompt = build_hypothesis_generator_prompt(
            code_context, variables_json, surfaces_json,
            nk_text, ml_text,
            num_hypotheses=self.search_config.hypotheses_per_loop,
        )
        self._save_prompt("hypotheses", prompt)

        try:
            result = self.mutator.call_llm_json(prompt, stage="generate_hypotheses")
            hypotheses_data = result if isinstance(result, list) else result.get("hypotheses", [])

            new_hypotheses = []
            for h_data in hypotheses_data:
                hid = h_data.get("id", self.memory.next_hypothesis_id())
                # Don't overwrite existing hypotheses
                if hid in self.memory.hypotheses:
                    hid = self.memory.next_hypothesis_id()

                hypothesis = Hypothesis(
                    id=hid,
                    title=h_data.get("title", ""),
                    status=HypothesisStatus.PROPOSED,
                    family=h_data.get("family", ""),
                    bottleneck_attacked=h_data.get("bottleneck_attacked", ""),
                    mechanism=h_data.get("mechanism", ""),
                    code_change_class=h_data.get("code_change_class", ""),
                    expected_win=h_data.get("expected_win", ""),
                    main_risk=h_data.get("main_risk", ""),
                    evidence_needed=h_data.get("evidence_needed", ""),
                    disproof_signal=h_data.get("disproof_signal", ""),
                    cheapest_test=h_data.get("cheapest_test", ""),
                    applies_when=h_data.get("applies_when", ""),
                    avoid_when=h_data.get("avoid_when", ""),
                    upside=int(h_data.get("upside", 3)),
                    feasibility=int(h_data.get("feasibility", 3)),
                    distinctness=int(h_data.get("distinctness", 3)),
                    information_gain=int(h_data.get("information_gain", 3)),
                    transferability=int(h_data.get("transferability", 3)),
                    notes=h_data.get("notes", ""),
                )
                self.memory.hypotheses[hid] = hypothesis
                new_hypotheses.append(hypothesis)

            return new_hypotheses
        except MutationError as e:
            logger.warning(f"Hypothesis generation failed: {e}")
            return []

    def _step_prune_hypotheses(self, variables_json: str) -> None:
        """Step 4: Prune weak/redundant hypotheses."""
        self._emit("stage", stage="prune_hypotheses")

        hypotheses_json = json.dumps(
            [asdict(h) for h in self.memory.hypotheses.values()
             if h.status == HypothesisStatus.PROPOSED],
            indent=2, default=str,
        )
        nk_text = self._negative_knowledge_text()
        ml_text = self._mutation_ledger_text()

        prompt = build_hypothesis_pruner_prompt(
            hypotheses_json, variables_json, nk_text, ml_text,
        )
        self._save_prompt("prune", prompt)

        try:
            result = self.mutator.call_llm_json(prompt, stage="prune_hypotheses")

            for kept in result.get("kept", []):
                hid = kept.get("id")
                if hid in self.memory.hypotheses:
                    old = self.memory.hypotheses[hid]
                    from dataclasses import replace
                    self.memory.hypotheses[hid] = replace(
                        old, status=HypothesisStatus.SHORTLISTED,
                    )

            for killed in result.get("killed", []):
                hid = killed.get("id")
                if hid in self.memory.hypotheses:
                    old = self.memory.hypotheses[hid]
                    from dataclasses import replace
                    self.memory.hypotheses[hid] = replace(
                        old,
                        status=HypothesisStatus.KILLED,
                        notes=f"Killed: {killed.get('reason', 'no reason given')}",
                    )
        except MutationError as e:
            logger.warning(f"Hypothesis pruning failed: {e}")
            # Shortlist all proposed as fallback
            from dataclasses import replace
            for hid, h in self.memory.hypotheses.items():
                if h.status == HypothesisStatus.PROPOSED:
                    self.memory.hypotheses[hid] = replace(h, status=HypothesisStatus.SHORTLISTED)

    def _step_find_gaps(
        self,
        hypotheses_json: str,
        variables_json: str,
        surfaces_json: str,
    ) -> None:
        """Step 5: Find gaps in coverage."""
        self._emit("stage", stage="find_gaps")

        prompt = build_gap_finder_prompt(hypotheses_json, variables_json, surfaces_json)
        self._save_prompt("gaps", prompt)

        try:
            result = self.mutator.call_llm_json(prompt, stage="find_gaps")

            for g_data in result.get("gaps", []):
                gid = g_data.get("id", self.memory.next_gap_id())
                gap = GapEntry(
                    id=gid,
                    uncovered_area=g_data.get("uncovered_area", ""),
                    why_it_matters=g_data.get("why_it_matters", ""),
                    current_portfolio_miss=g_data.get("current_portfolio_miss", ""),
                    candidate_hypothesis=g_data.get("candidate_hypothesis", ""),
                    expected_information_gain=g_data.get("expected_information_gain", ""),
                    missing_evidence=g_data.get("missing_evidence", ""),
                    urgency=g_data.get("urgency", "medium"),
                )
                self.memory.gaps[gid] = gap

            # Add gap-filling hypotheses
            for h_data in result.get("gap_filling_hypotheses", []):
                hid = self.memory.next_hypothesis_id()
                hypothesis = Hypothesis(
                    id=hid,
                    title=h_data.get("title", ""),
                    status=HypothesisStatus.SHORTLISTED,
                    family=h_data.get("family", ""),
                    bottleneck_attacked=h_data.get("bottleneck_attacked", ""),
                    mechanism=h_data.get("mechanism", ""),
                    code_change_class=h_data.get("code_change_class", ""),
                    expected_win=h_data.get("expected_win", ""),
                    main_risk=h_data.get("main_risk", ""),
                    evidence_needed=h_data.get("evidence_needed", ""),
                    disproof_signal=h_data.get("disproof_signal", ""),
                    cheapest_test=h_data.get("cheapest_test", ""),
                    upside=int(h_data.get("upside", 3)),
                    feasibility=int(h_data.get("feasibility", 3)),
                    distinctness=int(h_data.get("distinctness", 4)),
                    information_gain=int(h_data.get("information_gain", 4)),
                    transferability=int(h_data.get("transferability", 3)),
                    notes=f"Gap-filling hypothesis for: {h_data.get('bottleneck_attacked', '')}",
                )
                self.memory.hypotheses[hid] = hypothesis
        except MutationError as e:
            logger.warning(f"Gap finding failed: {e}")

    def _step_select_portfolio(
        self,
        hypotheses_json: str,
        variables_json: str,
    ) -> list[PortfolioSlot]:
        """Step 6: Select the execution portfolio."""
        self._emit("stage", stage="select_portfolio")

        gaps_json = json.dumps(
            [asdict(g) for g in self.memory.gaps.values()],
            indent=2, default=str,
        )
        nk_text = self._negative_knowledge_text()
        ml_text = self._mutation_ledger_text()

        prompt = build_portfolio_selector_prompt(
            hypotheses_json, gaps_json, nk_text, ml_text,
            portfolio_size=self.search_config.portfolio_size,
        )
        self._save_prompt("portfolio", prompt)

        try:
            result = self.mutator.call_llm_json(prompt, stage="select_portfolio")

            portfolio: list[PortfolioSlot] = []
            for p_data in result.get("portfolio", []):
                slot = PortfolioSlot(
                    id=p_data.get("id", f"P{len(portfolio) + 1}"),
                    role=ExperimentRole(p_data.get("role", "scout")),
                    hypothesis_id=p_data.get("hypothesis_id", ""),
                    family=p_data.get("family", ""),
                    target_region=p_data.get("target_region", ""),
                    operator_family=p_data.get("operator_family", ""),
                    why_selected=p_data.get("why_selected", ""),
                    expected_signal=p_data.get("expected_signal", ""),
                    acceptance_test=p_data.get("acceptance_test", ""),
                    kill_condition=p_data.get("kill_condition", ""),
                    overlap_check=p_data.get("overlap_check", ""),
                )
                portfolio.append(slot)

                # Mark hypothesis as active
                hid = slot.hypothesis_id
                if hid in self.memory.hypotheses:
                    from dataclasses import replace
                    self.memory.hypotheses[hid] = replace(
                        self.memory.hypotheses[hid],
                        status=HypothesisStatus.ACTIVE,
                    )

            self.memory.portfolio = portfolio
            return portfolio

        except MutationError as e:
            logger.warning(f"Portfolio selection failed: {e}")
            # Fallback: pick top shortlisted by upside
            active = sorted(
                self.memory.active_hypotheses(),
                key=lambda h: h.upside + h.information_gain,
                reverse=True,
            )[:self.search_config.portfolio_size]

            portfolio = []
            for i, h in enumerate(active):
                role = ExperimentRole.SCOUT if i < 2 else (
                    ExperimentRole.WILDCARD if i == len(active) - 1
                    else ExperimentRole.EXPLOIT
                )
                slot = PortfolioSlot(
                    id=f"P{i + 1}",
                    role=role,
                    hypothesis_id=h.id,
                    family=h.family,
                    target_region="",
                    operator_family=h.code_change_class,
                    why_selected=f"Top by upside+info_gain (fallback): {h.title}",
                    expected_signal=h.expected_win,
                    acceptance_test=h.evidence_needed,
                    kill_condition=h.disproof_signal,
                    overlap_check="not checked (fallback)",
                )
                portfolio.append(slot)

            self.memory.portfolio = portfolio
            return portfolio

    def _step_implement_and_evaluate(
        self,
        portfolio: list[PortfolioSlot],
        code_context: str,
        variables_json: str,
    ) -> list[Candidate]:
        """Steps 7-8: Implement one candidate per portfolio slot and evaluate."""
        self._emit("stage", stage="implement_and_evaluate")
        candidates: list[Candidate] = []

        best_parent = self.population.best()
        nk_text = self._negative_knowledge_text()
        memory_text = self.memory.export_for_prompt(
            max_tokens=self.search_config.max_context_tokens,
        )

        for slot in portfolio:
            self._emit("slot_start", slot=slot.id, hypothesis=slot.hypothesis_id)

            # Register mutation neighborhood
            n_id = self.memory.next_neighborhood_id()
            neighborhood = MutationNeighborhood(
                id=n_id,
                loop=self.memory.current_loop,
                hypothesis_id=slot.hypothesis_id,
                target_region=slot.target_region,
                operator_family=slot.operator_family,
                bottleneck_attacked=slot.family,
                benchmark_slice="all",
                parent_candidate=best_parent.id,
                overlap_status=slot.overlap_check,
                why_allowed=slot.why_selected,
            )
            self.memory.mutation_ledger[n_id] = neighborhood

            # Get hypothesis details
            hypothesis = self.memory.hypotheses.get(slot.hypothesis_id)
            hypothesis_json = json.dumps(asdict(hypothesis), indent=2, default=str) if hypothesis else "{}"
            slot_json = json.dumps(asdict(slot), indent=2, default=str)

            # Build implementation prompt
            prompt = build_candidate_implementer_prompt(
                slot_json=slot_json,
                hypothesis_json=hypothesis_json,
                parent_source=best_parent.source,
                code_context=code_context,
                negative_knowledge_text=nk_text,
                search_memory_text=memory_text,
            )
            self._save_prompt(f"implement_{slot.id}", prompt)

            # Generate diff
            candidate_id = self.population.next_id()
            try:
                proposal = self.mutator.generate_diff(prompt, stage="implement")
                diff_path = self.diffs_dir / f"loop_{self.memory.current_loop:03d}_{slot.id}.diff"
                diff_path.write_text(proposal.raw_diff)

                # Apply diff
                child_source, apply_stats = apply_search_replace_blocks(
                    best_parent.source, proposal.raw_diff,
                )

                # Evaluate
                evaluation = self.evaluator.evaluate(child_source)

                child = Candidate(
                    id=candidate_id,
                    generation=len(self.population.candidates),
                    loop=self.memory.current_loop,
                    parent_id=best_parent.id,
                    source=child_source,
                    aggregate_score=evaluation.aggregate_score,
                    metrics=evaluation.metrics,
                    failure_reasons=evaluation.failure_reasons,
                    stage_results=evaluation.stage_results,
                    hypothesis_id=slot.hypothesis_id,
                    portfolio_slot=slot.id,
                    active=evaluation.valid,
                    lineage={
                        "mutation_model": proposal.model,
                        "parent_id": best_parent.id,
                        "hypothesis": slot.hypothesis_id,
                        "slot": slot.id,
                        "apply_stats": apply_stats,
                    },
                )
                self.population.add(child)
                candidates.append(child)

                # Save evaluation
                eval_path = self.evals_dir / f"loop_{self.memory.current_loop:03d}_{slot.id}.json"
                eval_path.write_text(json.dumps({
                    "candidate_id": candidate_id,
                    "slot": slot.id,
                    "hypothesis": slot.hypothesis_id,
                    "valid": evaluation.valid,
                    "aggregate_score": evaluation.aggregate_score,
                    "metrics": evaluation.metrics,
                    "failure_reasons": evaluation.failure_reasons,
                }, indent=2))

                self._emit(
                    "slot_evaluated",
                    slot=slot.id,
                    candidate=candidate_id,
                    valid=evaluation.valid,
                    score=evaluation.aggregate_score,
                )

            except MutationError as e:
                logger.warning(f"Slot {slot.id} mutation failed: {e}")
                child = Candidate(
                    id=candidate_id,
                    generation=len(self.population.candidates),
                    loop=self.memory.current_loop,
                    parent_id=best_parent.id,
                    source=best_parent.source,
                    aggregate_score=-1.0,
                    metrics={},
                    failure_reasons=[f"Mutation failed: {e}"],
                    stage_results=[],
                    hypothesis_id=slot.hypothesis_id,
                    portfolio_slot=slot.id,
                    active=False,
                    lineage={"mutation_model": "failed", "error": str(e)},
                )
                self.population.add(child)
                candidates.append(child)
                self._emit("slot_failed", slot=slot.id, error=str(e))

        return candidates

    def _step_evaluate_skeptically(
        self,
        portfolio: list[PortfolioSlot],
        candidates: list[Candidate],
        variables_json: str,
    ) -> None:
        """Step 9: Run evaluator skepticism."""
        self._emit("stage", stage="evaluate_skeptically")

        portfolio_json = json.dumps([asdict(s) for s in portfolio], indent=2, default=str)
        results_json = json.dumps([
            {
                "candidate_id": c.id,
                "slot": c.portfolio_slot,
                "hypothesis": c.hypothesis_id,
                "valid": c.active,
                "score": c.aggregate_score,
                "metrics": c.metrics,
                "failures": c.failure_reasons,
            }
            for c in candidates
        ], indent=2)

        feedback = self.population.build_feedback_bundle()
        loop_context = f"Loop {self.memory.current_loop}, baseline score: {feedback.best_metrics}"

        prompt = build_evaluator_skeptic_prompt(
            portfolio_json, results_json, variables_json, loop_context,
        )
        self._save_prompt("skeptic", prompt)

        try:
            result = self.mutator.call_llm_json(prompt, stage="evaluate_skeptically")

            # Process assessments — kill misleading candidates
            for assessment in result.get("assessments", []):
                cid = assessment.get("candidate_id")
                verdict = assessment.get("verdict", "tentative")
                action = assessment.get("recommended_action", "keep")

                if verdict == "misleading" and action == "kill":
                    for c in candidates:
                        if c.id == cid:
                            c.active = False
                            c.meta["skeptic_verdict"] = verdict
                            c.meta["skeptic_concerns"] = assessment.get("concerns", [])

            self._emit("skeptic_done", warnings=result.get("fake_win_warnings", []))
        except MutationError as e:
            logger.warning(f"Skeptic stage failed: {e}")

    def _step_postmortem(
        self,
        portfolio: list[PortfolioSlot],
        candidates: list[Candidate],
        variables_json: str,
    ) -> None:
        """Steps 10-11: Postmortem and update durable memory."""
        self._emit("stage", stage="postmortem")

        portfolio_json = json.dumps([asdict(s) for s in portfolio], indent=2, default=str)
        results_json = json.dumps([
            {
                "candidate_id": c.id,
                "valid": c.active,
                "score": c.aggregate_score,
                "metrics": c.metrics,
                "hypothesis": c.hypothesis_id,
            }
            for c in candidates
        ], indent=2)
        hypotheses_json = json.dumps(
            [asdict(h) for h in self.memory.hypotheses.values()
             if h.status in (HypothesisStatus.ACTIVE, HypothesisStatus.TESTED)],
            indent=2, default=str,
        )
        nk_text = self._negative_knowledge_text()
        ml_text = self._mutation_ledger_text()

        prompt = build_postmortem_prompt(
            portfolio_json, results_json, hypotheses_json, nk_text, ml_text,
        )
        self._save_prompt("postmortem", prompt)

        try:
            result = self.mutator.call_llm_json(prompt, stage="postmortem")

            # Record loop
            loop_record = LoopRecord(
                loop=self.memory.current_loop,
                date=datetime.now().isoformat(),
                target=self.task_config.name,
                parent_baseline=self.population.best().id if self.population.active_candidates() else "none",
                baseline_metrics=self.population.build_feedback_bundle().best_metrics,
                active_portfolio=[s.id for s in portfolio],
                overlap_conflicts=[],
                best_candidate=result.get("best_candidate", candidates[0].id if candidates else "none"),
                worst_candidate=result.get("worst_candidate", ""),
                score_movement=result.get("score_movement", "unknown"),
                holdout_movement=result.get("holdout_movement", "unknown"),
                stability_movement=result.get("stability_movement", "unknown"),
                what_improved=result.get("what_improved", ""),
                what_regressed=result.get("what_regressed", ""),
                strongest_evidence=result.get("strongest_evidence", ""),
                likely_causal_explanation=result.get("likely_causal_explanation", ""),
                hypotheses_promoted=result.get("hypotheses_promoted", []),
                hypotheses_killed=result.get("hypotheses_killed", []),
                child_hypotheses=result.get("child_hypotheses", []),
                negative_knowledge_added=result.get("negative_knowledge", []),
                neighborhoods_retired=result.get("neighborhoods_to_retire", []),
                neighborhoods_reopened=result.get("neighborhoods_to_reopen", []),
                next_loop_focus=result.get("next_loop_focus", ""),
            )
            self.memory.loop_log.append(loop_record)

            # Update hypothesis statuses
            from dataclasses import replace as dc_replace
            for hid in result.get("hypotheses_promoted", []):
                if hid in self.memory.hypotheses:
                    self.memory.hypotheses[hid] = dc_replace(
                        self.memory.hypotheses[hid],
                        status=HypothesisStatus.PROMOTED,
                    )
            for hid in result.get("hypotheses_killed", []):
                if hid in self.memory.hypotheses:
                    self.memory.hypotheses[hid] = dc_replace(
                        self.memory.hypotheses[hid],
                        status=HypothesisStatus.KILLED,
                    )

            # Add negative knowledge
            for nk_text_item in result.get("negative_knowledge", []):
                if isinstance(nk_text_item, str):
                    nk_id = self.memory.next_nk_id()
                    nk = NegativeKnowledge(
                        id=nk_id,
                        move_family="postmortem",
                        observed_failure=nk_text_item,
                        likely_cause="See postmortem",
                        evidence=f"Loop {self.memory.current_loop}",
                        confidence="medium",
                        do_not_repeat_unless="new evidence changes assumptions",
                        revisit_trigger="new code or new hypothesis",
                        source_loops=[self.memory.current_loop],
                    )
                    self.memory.negative_knowledge[nk_id] = nk
                elif isinstance(nk_text_item, dict):
                    nk_id = nk_text_item.get("id", self.memory.next_nk_id())
                    nk = NegativeKnowledge(
                        id=nk_id,
                        move_family=nk_text_item.get("move_family", "postmortem"),
                        observed_failure=nk_text_item.get("observed_failure", ""),
                        likely_cause=nk_text_item.get("likely_cause", ""),
                        evidence=nk_text_item.get("evidence", f"Loop {self.memory.current_loop}"),
                        confidence=nk_text_item.get("confidence", "medium"),
                        do_not_repeat_unless=nk_text_item.get("do_not_repeat_unless", ""),
                        revisit_trigger=nk_text_item.get("revisit_trigger", ""),
                        source_loops=[self.memory.current_loop],
                    )
                    self.memory.negative_knowledge[nk_id] = nk

            # Retire neighborhoods
            from dataclasses import replace as dc_replace2
            for n_id in result.get("neighborhoods_to_retire", []):
                if n_id in self.memory.mutation_ledger:
                    old = self.memory.mutation_ledger[n_id]
                    self.memory.mutation_ledger[n_id] = dc_replace2(old, retired=True)

            self._emit("postmortem_done", next_focus=result.get("next_loop_focus", ""))

        except MutationError as e:
            logger.warning(f"Postmortem failed: {e}")
            # Still record a minimal loop entry
            self.memory.loop_log.append(LoopRecord(
                loop=self.memory.current_loop,
                date=datetime.now().isoformat(),
                target=self.task_config.name,
                parent_baseline="",
                baseline_metrics={},
                active_portfolio=[s.id for s in portfolio],
                overlap_conflicts=[],
                best_candidate=candidates[0].id if candidates else "none",
                worst_candidate="",
                score_movement="postmortem failed",
                holdout_movement="",
                stability_movement="",
                what_improved="",
                what_regressed="",
                strongest_evidence="",
                likely_causal_explanation="",
                hypotheses_promoted=[],
                hypotheses_killed=[],
                child_hypotheses=[],
                negative_knowledge_added=[],
                neighborhoods_retired=[],
                neighborhoods_reopened=[],
                next_loop_focus="retry postmortem",
            ))

    # ── Helpers ─────────────────────────────────────────────────

    def _negative_knowledge_text(self) -> str:
        if not self.memory.negative_knowledge:
            return ""
        lines = []
        for nk in self.memory.negative_knowledge.values():
            lines.append(
                f"- {nk.id} [{nk.move_family}]: {nk.observed_failure}. "
                f"Cause: {nk.likely_cause}. "
                f"Don't repeat unless: {nk.do_not_repeat_unless}"
            )
        return "\n".join(lines)

    def _mutation_ledger_text(self) -> str:
        if not self.memory.mutation_ledger:
            return ""
        lines = []
        for n in self.memory.mutation_ledger.values():
            status = "RETIRED" if n.retired else "active"
            lines.append(
                f"- {n.id} [{status}]: region={n.target_region}, "
                f"operator={n.operator_family}, outcome={n.outcome}"
            )
        return "\n".join(lines)

    def _save_prompt(self, name: str, prompt: str) -> None:
        path = self.prompts_dir / f"loop_{self.memory.current_loop:03d}_{name}.txt"
        path.write_text(prompt)

    def _emit(self, event_type: str, **payload: Any) -> None:
        event = {"event_type": event_type, "ts": datetime.now().isoformat()}
        event.update(payload)

        # Append to events log
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception:
                pass
