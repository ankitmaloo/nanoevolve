from __future__ import annotations

import asyncio
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from mvp.config import RunConfig
from mvp.diff_engine import DiffFormatError, SearchBlockNotFoundError, apply_search_replace_blocks
from mvp.evaluator import Evaluator
from mvp.evolve_blocks import assert_has_evolve_blocks
from mvp.mutator_gemini import GeminiMutator
from mvp.mutator_mock import MockMutator
from mvp.population_db import PopulationDB
from mvp.prompt_evolver import PromptEvolver
from mvp.types import Candidate, EvaluationResult, GenerationRecord, StageResult


class EvolutionController:
    def __init__(
        self,
        base_dir: Path,
        config: RunConfig,
        event_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.config = config
        self.event_callback = event_callback
        self.rng = random.Random(config.seed)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.run_name or f"{config.mode}_{timestamp}"
        self.run_dir = (base_dir / "runs" / run_name).resolve()
        self.prompts_dir = self.run_dir / "prompts"
        self.diffs_dir = self.run_dir / "diffs"
        self.evals_dir = self.run_dir / "evaluations"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.diffs_dir.mkdir(parents=True, exist_ok=True)
        self.evals_dir.mkdir(parents=True, exist_ok=True)

        self.db = PopulationDB(seed=config.seed)
        self.prompt_evolver = PromptEvolver(seed=config.seed)
        self.evaluator = Evaluator()
        self.parallel_candidates = max(1, int(config.parallel_candidates))
        self.llm_concurrency = max(1, int(config.llm_concurrency))
        self.metric_keys = [
            "aggregate_score",
            "placed_jobs_ratio",
            "balance_score",
            "fragmentation_score",
            "solved_ratio",
            "path_quality",
            "expansion_efficiency",
        ]
        self.events_path = self.run_dir / "events.jsonl"

        if config.mode == "mock":
            self.mutator = MockMutator(config.resolve_mock_diff_path(base_dir))
        elif config.mode == "gemini":
            self.mutator = GeminiMutator(model_name=config.model_name)
        elif config.mode == "openai":
            from mvp.mutator_openai import OpenAIMutator

            self.mutator = OpenAIMutator(
                model_name=config.model_name,
                fast_model_name=config.openai_fast_model_name,
                slow_every=config.openai_slow_every,
                request_timeout_s=config.openai_request_timeout_s,
                max_retries=config.openai_max_retries,
                max_output_tokens=config.openai_max_output_tokens,
            )
        else:
            raise ValueError(f"Unsupported mode: {config.mode}")

    def run(self) -> dict[str, object]:
        self._emit_runtime_event(
            "run_start",
            mode=self.config.mode,
            model_name=self.config.model_name,
            generations=self.config.generations,
            parallel_candidates=self.parallel_candidates,
            llm_concurrency=self.llm_concurrency,
            run_dir=str(self.run_dir),
        )
        seed_source = self.config.resolve_seed_program_path(self.base_dir).read_text()
        assert_has_evolve_blocks(seed_source)
        (self.run_dir / "seed_program.py").write_text(seed_source)
        (self.run_dir / "config.json").write_text(json.dumps(asdict(self.config), indent=2))

        seed_eval = self.evaluator.evaluate(seed_source)
        seed_candidate = Candidate(
            id="cand_0000",
            generation=0,
            parent_id=None,
            source=seed_source,
            aggregate_score=seed_eval.aggregate_score,
            metrics=seed_eval.metrics,
            failure_reasons=seed_eval.failure_reasons,
            stage_results=seed_eval.stage_results,
            active=seed_eval.valid,
            lineage={"mutation": "seed"},
            meta={"descriptor": self._compute_descriptor(seed_source, seed_eval.metrics)},
        )
        self.db.add(seed_candidate)

        event = {
            "generation": 0,
            "candidate_id": seed_candidate.id,
            "aggregate_score": seed_candidate.aggregate_score,
            "valid": seed_eval.valid,
            "mode": self.config.mode,
        }
        self._append_event(event)
        self._emit_runtime_event(
            "seed_evaluated",
            candidate_id=seed_candidate.id,
            aggregate_score=seed_candidate.aggregate_score,
            valid=seed_eval.valid,
        )

        candidate_counter = 1

        for generation in range(1, self.config.generations + 1):
            self._emit_runtime_event(
                "generation_start",
                generation=generation,
                active_population=len(self.db.active_candidates()),
            )
            batch_specs: list[dict[str, object]] = []
            for slot in range(self.parallel_candidates):
                metric_key = self.rng.choice(self.metric_keys)
                parent = self.db.sample_parent(metric_key=metric_key)
                inspirations = self.db.sample_inspirations(self.config.inspirations_k, exclude_id=parent.id)
                feedback = self.db.build_feedback_bundle()
                prompt_candidate = self.prompt_evolver.sample()

                prompt = self._build_prompt(parent, inspirations, feedback, prompt_candidate.text)
                prompt_file = self.prompts_dir / f"gen_{generation:04d}_slot_{slot:02d}.txt"
                prompt_file.write_text(prompt)

                child_id = f"cand_{candidate_counter:04d}"
                candidate_counter += 1
                diff_file = self.diffs_dir / f"gen_{generation:04d}_slot_{slot:02d}.diff"
                eval_file = self.evals_dir / f"gen_{generation:04d}_slot_{slot:02d}.json"

                batch_specs.append(
                    {
                        "slot": slot,
                        "generation": generation,
                        "metric_key": metric_key,
                        "parent": parent,
                        "inspirations": inspirations,
                        "feedback": feedback,
                        "prompt_candidate": prompt_candidate,
                        "prompt": prompt,
                        "prompt_file": prompt_file,
                        "child_id": child_id,
                        "diff_file": diff_file,
                        "eval_file": eval_file,
                    }
                )
                self._emit_runtime_event(
                    "slot_queued",
                    generation=generation,
                    slot=slot,
                    parent_id=parent.id,
                    child_id=child_id,
                    metric_key=metric_key,
                    prompt_id=prompt_candidate.id,
                )

            mutate_results = asyncio.run(self._mutate_batch(batch_specs))

            for spec, mutate_result in zip(batch_specs, mutate_results):
                parent = spec["parent"]
                inspirations = spec["inspirations"]
                prompt_candidate = spec["prompt_candidate"]
                metric_key = spec["metric_key"]
                child_id = spec["child_id"]
                diff_file = spec["diff_file"]
                eval_file = spec["eval_file"]
                prompt_file = spec["prompt_file"]
                generation = spec["generation"]
                slot = int(spec["slot"])

                candidate_valid = False
                prompt_reward = -1.0
                survivor_ids: list[str] = [c.id for c in self.db.active_candidates()]
                dropped_ids: list[str] = []

                if mutate_result["error"] is not None:
                    self._emit_runtime_event(
                        "slot_mutation_error",
                        generation=generation,
                        slot=slot,
                        parent_id=parent.id,
                        child_id=child_id,
                        error=str(mutate_result["error"]),
                    )
                    evaluation = EvaluationResult(
                        valid=False,
                        aggregate_score=-1.0,
                        metrics={},
                        failure_reasons=[f"Controller generation failure: {mutate_result['error']}"],
                        stage_results=[StageResult(name="controller", passed=False, message=str(mutate_result["error"]))],
                    )
                    child_candidate = Candidate(
                        id=child_id,
                        generation=generation,
                        parent_id=parent.id,
                        source=parent.source,
                        aggregate_score=-1.0,
                        metrics={},
                        failure_reasons=evaluation.failure_reasons,
                        stage_results=evaluation.stage_results,
                        active=False,
                        lineage={"mutation_model": "controller_error", "parent_id": parent.id},
                        meta={"descriptor": self._compute_descriptor(parent.source, {})},
                    )
                    self.db.add(child_candidate)
                    eval_file.write_text(
                        json.dumps(
                            {
                                "candidate_id": child_id,
                                "slot": slot,
                                "valid": False,
                                "aggregate_score": -1.0,
                                "failure_reasons": evaluation.failure_reasons,
                                "stage_results": [asdict(s) for s in evaluation.stage_results],
                                "mutation_model": "controller_error",
                            },
                            indent=2,
                        )
                    )
                    self._emit_runtime_event(
                        "slot_evaluated",
                        generation=generation,
                        slot=slot,
                        candidate_id=child_id,
                        valid=False,
                        aggregate_score=-1.0,
                        mutation_model="controller_error",
                        error=str(mutate_result["error"]),
                    )
                else:
                    try:
                        proposal = mutate_result["proposal"]
                        self._emit_runtime_event(
                            "slot_mutation_done",
                            generation=generation,
                            slot=slot,
                            parent_id=parent.id,
                            child_id=child_id,
                            mutation_model=proposal.model,
                        )
                        diff_file.write_text(proposal.raw_diff)

                        child_source, apply_stats = apply_search_replace_blocks(parent.source, proposal.raw_diff)
                        assert_has_evolve_blocks(child_source)

                        evaluation = self.evaluator.evaluate(child_source)
                        candidate_valid = evaluation.valid
                        prompt_reward = evaluation.aggregate_score if evaluation.valid else -1.0

                        child_candidate = Candidate(
                            id=child_id,
                            generation=generation,
                            parent_id=parent.id,
                            source=child_source,
                            aggregate_score=evaluation.aggregate_score,
                            metrics=evaluation.metrics,
                            failure_reasons=evaluation.failure_reasons,
                            stage_results=evaluation.stage_results,
                            active=evaluation.valid,
                            lineage={
                                "mutation_model": proposal.model,
                                "parent_id": parent.id,
                                "inspiration_ids": [c.id for c in inspirations],
                            },
                            meta={
                                "descriptor": self._compute_descriptor(child_source, evaluation.metrics),
                                "apply_stats": apply_stats,
                                "proposal_metadata": proposal.metadata,
                                "prompt_id": prompt_candidate.id,
                            },
                        )
                        self.db.add(child_candidate)

                        if evaluation.valid:
                            survivor_ids, dropped_ids = self.db.prune_survivors(
                                top_k=self.config.survivor_top_k,
                                diversity_slots=self.config.diversity_slots,
                            )
                        eval_file.write_text(
                            json.dumps(
                                {
                                    "candidate_id": child_id,
                                    "slot": slot,
                                    "valid": evaluation.valid,
                                    "aggregate_score": evaluation.aggregate_score,
                                    "metrics": evaluation.metrics,
                                    "failure_reasons": evaluation.failure_reasons,
                                    "stage_results": [asdict(s) for s in evaluation.stage_results],
                                    "diagnostics": evaluation.diagnostics,
                                    "mutation_model": proposal.model,
                                    "proposal_metadata": proposal.metadata,
                                },
                                indent=2,
                            )
                        )
                        self._emit_runtime_event(
                            "slot_evaluated",
                            generation=generation,
                            slot=slot,
                            candidate_id=child_id,
                            valid=evaluation.valid,
                            aggregate_score=evaluation.aggregate_score,
                            mutation_model=proposal.model,
                        )

                    except (DiffFormatError, SearchBlockNotFoundError) as exc:
                        diff_file.write_text(diff_file.read_text() if diff_file.exists() else "")
                        evaluation = EvaluationResult(
                            valid=False,
                            aggregate_score=-1.0,
                            metrics={},
                            failure_reasons=[f"Diff apply failed: {exc}"],
                            stage_results=[StageResult(name="diff_apply", passed=False, message=str(exc))],
                        )
                        child_candidate = Candidate(
                            id=child_id,
                            generation=generation,
                            parent_id=parent.id,
                            source=parent.source,
                            aggregate_score=evaluation.aggregate_score,
                            metrics=evaluation.metrics,
                            failure_reasons=evaluation.failure_reasons,
                            stage_results=evaluation.stage_results,
                            active=False,
                            lineage={"mutation_model": "failed_diff", "parent_id": parent.id},
                            meta={"descriptor": self._compute_descriptor(parent.source, evaluation.metrics)},
                        )
                        self.db.add(child_candidate)
                        prompt_reward = -1.0
                        eval_file.write_text(
                            json.dumps(
                                {
                                    "candidate_id": child_id,
                                    "slot": slot,
                                    "valid": False,
                                    "aggregate_score": -1.0,
                                    "failure_reasons": evaluation.failure_reasons,
                                    "stage_results": [asdict(s) for s in evaluation.stage_results],
                                    "mutation_model": "failed_diff",
                                },
                                indent=2,
                            )
                        )
                        self._emit_runtime_event(
                            "slot_evaluated",
                            generation=generation,
                            slot=slot,
                            candidate_id=child_id,
                            valid=False,
                            aggregate_score=-1.0,
                            mutation_model="failed_diff",
                            error=str(exc),
                        )

                self.prompt_evolver.update(prompt_candidate.id, prompt_reward)
                if generation % 5 == 0:
                    self.prompt_evolver.evolve_population()

                record = GenerationRecord(
                    generation=generation,
                    parent_id=parent.id,
                    child_id=child_id,
                    metric_key=metric_key,
                    prompt_id=prompt_candidate.id,
                    prompt_reward=prompt_reward,
                    candidate_valid=candidate_valid,
                    survivor_ids=survivor_ids,
                    dropped_ids=dropped_ids,
                    prompt_file=str(prompt_file.relative_to(self.run_dir)),
                    diff_file=str(diff_file.relative_to(self.run_dir)),
                    evaluation_file=str(eval_file.relative_to(self.run_dir)),
                )

                event_payload = asdict(record)
                event_payload["slot"] = slot
                event_payload["mutation_model"] = child_candidate.lineage.get("mutation_model")
                event_payload["proposal_metadata"] = child_candidate.meta.get("proposal_metadata", {})
                self._append_event(event_payload)

            gen_best = self.db.best()
            self._emit_runtime_event(
                "generation_end",
                generation=generation,
                active_population=len(self.db.active_candidates()),
                best_candidate_id=gen_best.id,
                best_aggregate_score=gen_best.aggregate_score,
            )

        best = self.db.best()
        (self.run_dir / "best_program.py").write_text(best.source)

        summary = {
            "mode": self.config.mode,
            "model_name": self.config.model_name,
            "generations": self.config.generations,
            "seed_program_path": self.config.seed_program_path,
            "total_candidates": len(self.db.candidates),
            "active_candidates": [c.id for c in self.db.active_candidates()],
            "best_candidate_id": best.id,
            "best_aggregate_score": best.aggregate_score,
            "top_prompts": self.prompt_evolver.top_prompts(),
            "run_dir": str(self.run_dir),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self._emit_runtime_event(
            "run_end",
            best_candidate_id=best.id,
            best_aggregate_score=best.aggregate_score,
            total_candidates=len(self.db.candidates),
            run_dir=str(self.run_dir),
        )
        return summary

    async def _mutate_batch(self, batch_specs: list[dict[str, object]]) -> list[dict[str, object]]:
        sem = asyncio.Semaphore(self.llm_concurrency)

        async def _run_one(spec: dict[str, object]) -> dict[str, object]:
            async with sem:
                parent = spec["parent"]
                child_id = str(spec["child_id"])
                generation = int(spec["generation"])
                slot = int(spec["slot"])
                self._emit_runtime_event(
                    "slot_mutation_start",
                    generation=generation,
                    slot=slot,
                    parent_id=parent.id,
                    child_id=child_id,
                )
                try:
                    proposal = await self.mutator.mutate(
                        spec["parent"],
                        spec["inspirations"],
                        spec["feedback"],
                        spec["prompt"],
                    )
                    return {"proposal": proposal, "error": None}
                except Exception as exc:
                    return {"proposal": None, "error": str(exc)}

        tasks = [_run_one(spec) for spec in batch_specs]
        return await asyncio.gather(*tasks)

    def _compute_descriptor(self, source: str, metrics: dict[str, float]) -> str:
        source_bucket = len(source) // 200
        if metrics:
            values = list(metrics.values())
            spread = max(values) - min(values)
        else:
            spread = 0.0
        spread_bucket = int(spread * 10)
        return f"len:{source_bucket}|spread:{spread_bucket}"

    def _build_prompt(
        self,
        parent: Candidate,
        inspirations: list[Candidate],
        feedback,
        prompt_strategy_text: str,
    ) -> str:
        inspiration_section = "\n\n".join(
            [
                f"- {c.id}: aggregate={c.aggregate_score:.4f}, metrics={json.dumps(c.metrics)}"
                for c in inspirations
            ]
        )
        if not inspiration_section:
            inspiration_section = "- none"

        failure_section = "\n".join([f"- {r}" for r in feedback.weak_failure_reasons]) or "- none"
        dropped_section = "\n".join([f"- {r}" for r in feedback.dropped_notes]) or "- none"
        astar_guidance = ""
        if "def priority_score(" in parent.source and "def tie_break_priority(" in parent.source:
            astar_guidance = """
- Do not return a no-op mutation.
- Change at least one numeric coefficient in EVOLVE blocks.
- If current score plateaus, search nearby coefficients:
  - slightly lower turn/crowding penalties,
  - slightly increase progress weight,
  - or refine tie-break coefficients.
"""

        return f"""Act as an expert software developer improving an evolvable heuristic program.

Goal: propose a code mutation that improves target metrics while keeping program validity.

Prompt strategy (co-evolved):
{prompt_strategy_text}

Best metrics seen so far:
{json.dumps(feedback.best_metrics, indent=2)}

Recent weak-candidate failure reasons:
{failure_section}

Recent drop notes:
{dropped_section}

Inspiration candidates:
{inspiration_section}

Current parent candidate: {parent.id}
Parent metrics: {json.dumps(parent.metrics)}

# Current program
{parent.source}

# SEARCH/REPLACE block rules
Every block must use this exact format:
<<<<<<< SEARCH
<original code to match>
=======
<replacement code>
>>>>>>> REPLACE

Rules:
- SEARCH must exactly match existing code.
- Propose coherent edits only.
- Preserve EVOLVE-BLOCK markers.
- Prefer small targeted changes.
- Fix weaknesses seen in recent failures.
{astar_guidance}

ONLY EVER RETURN CODE IN SEARCH/REPLACE BLOCKS.
"""

    def _append_event(self, payload: dict[str, object]) -> None:
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _emit_runtime_event(self, event_type: str, **payload: object) -> None:
        if self.event_callback is None:
            return
        event: dict[str, object] = {"event_type": event_type, "ts": datetime.now().isoformat()}
        event.update(payload)
        try:
            self.event_callback(event)
        except Exception:
            # Keep evolution loop robust if UI callback fails.
            return
