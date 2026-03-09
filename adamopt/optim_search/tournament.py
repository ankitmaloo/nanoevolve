from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from .archive import SearchArchive
from .config import EvaluationConfig, SearchConfig
from .eval_candidate import ToyNanoChatBackend
from .mutations import mutate_spec
from .score import analyze_win_hierarchy, composite_score, pareto_frontier
from .spec import MatrixOptimizerSpec
from .types import CandidateRecord, PromotionResult, TournamentSummary


class OptimizerTournament:
    def __init__(
        self,
        *,
        root_dir: Path,
        search_config: SearchConfig,
        evaluation_config: EvaluationConfig,
        backend: ToyNanoChatBackend | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.search_config = search_config
        self.evaluation_config = evaluation_config
        self.backend = backend or ToyNanoChatBackend(evaluation_config)
        self.archive = SearchArchive()
        self.rng = random.Random(search_config.seed)
        self.run_dir = search_config.resolve_run_dir(root_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _baseline_record(self) -> CandidateRecord:
        spec = MatrixOptimizerSpec.baseline_nanochat()
        outcome = self.backend.evaluate(spec, seed=self.evaluation_config.seed, candidate_id="cand_0000")
        record = CandidateRecord(
            id="cand_0000",
            generation=0,
            parent_id=None,
            spec=spec.to_dict(),
            score=0.0,
            pareto=True,
            promoted=True,
            status="baseline",
            primary_outcome=outcome,
            win_assessment=analyze_win_hierarchy(outcome, outcome, self.search_config),
            lineage={"mutation": "seed"},
        )
        self.archive.add(record)
        return record

    def _promotion_compare(self, candidate: CandidateRecord, baseline: CandidateRecord) -> PromotionResult:
        candidate_spec = MatrixOptimizerSpec.from_dict(candidate.spec)
        baseline_spec = MatrixOptimizerSpec.from_dict(baseline.spec)
        candidate_runs = []
        baseline_runs = []
        for seed in self.search_config.promotion_seeds:
            cfg = EvaluationConfig(**{**asdict(self.evaluation_config), "seed": seed})
            backend = ToyNanoChatBackend(cfg)
            baseline_runs.append(backend.evaluate(baseline_spec, seed=seed, candidate_id=f"{baseline.id}_seed_{seed}"))
            candidate_runs.append(backend.evaluate(candidate_spec, seed=seed, candidate_id=f"{candidate.id}_seed_{seed}"))

        candidate_metrics = [run.metrics for run in candidate_runs if run.metrics is not None]
        baseline_metrics = [run.metrics for run in baseline_runs if run.metrics is not None]
        mean_candidate_bpb = sum(item.final_validation_bpb for item in candidate_metrics) / len(candidate_metrics)
        mean_baseline_bpb = sum(item.final_validation_bpb for item in baseline_metrics) / len(baseline_metrics)
        mean_candidate_step = sum(item.mean_step_time_ms for item in candidate_metrics) / len(candidate_metrics)
        mean_baseline_step = sum(item.mean_step_time_ms for item in baseline_metrics) / len(baseline_metrics)
        mean_candidate_tps = sum(item.tokens_per_sec for item in candidate_metrics) / len(candidate_metrics)
        mean_baseline_tps = sum(item.tokens_per_sec for item in baseline_metrics) / len(baseline_metrics)
        mean_candidate_memory = sum(item.memory_overhead_bytes for item in candidate_metrics) / len(candidate_metrics)
        mean_baseline_memory = sum(item.memory_overhead_bytes for item in baseline_metrics) / len(baseline_metrics)
        mean_candidate_stability = sum(item.stability_penalty for item in candidate_metrics) / len(candidate_metrics)
        mean_baseline_stability = sum(item.stability_penalty for item in baseline_metrics) / len(baseline_metrics)
        mean_candidate_spikes = sum(item.grad_norm_spikes for item in candidate_metrics) / len(candidate_metrics)
        mean_baseline_spikes = sum(item.grad_norm_spikes for item in baseline_metrics) / len(baseline_metrics)

        improvement = mean_baseline_bpb - mean_candidate_bpb
        speed_ratio = mean_candidate_step / max(mean_baseline_step, 1e-8)
        tokens_per_sec_ratio = mean_candidate_tps / max(mean_baseline_tps, 1e-8)
        memory_ratio = mean_candidate_memory / max(mean_baseline_memory, 1.0)
        per_seed_wins = 0
        per_seed_assessments = []
        time_to_target_ratios: list[float] = []
        for candidate_run, baseline_run in zip(candidate_runs, baseline_runs):
            assessment = analyze_win_hierarchy(
                candidate_run,
                baseline_run,
                self.search_config,
                allow_multi_seed_hierarchy=False,
            )
            per_seed_assessments.append(assessment)
            if assessment.winner:
                per_seed_wins += 1
            if assessment.wallclock and assessment.wallclock.value is not None:
                time_to_target_ratios.append(assessment.wallclock.value)

        seed_win_rate = per_seed_wins / max(len(candidate_runs), 1)
        aggregate_outcome = candidate.primary_outcome
        baseline_outcome = baseline.primary_outcome
        assessment = analyze_win_hierarchy(
            aggregate_outcome,
            baseline_outcome,
            self.search_config,
            seed_win_rate=seed_win_rate,
            allow_multi_seed_hierarchy=True,
        )
        notes = list(assessment.notes)
        if not all(run.valid for run in candidate_runs):
            notes.append("stability_failure")
        if seed_win_rate < self.search_config.min_seed_win_rate:
            notes.append("seed_win_rate_below_threshold")
        winner = assessment.winner and all(run.valid for run in candidate_runs)

        return PromotionResult(
            candidate_id=candidate.id,
            winner=winner,
            mean_final_validation_bpb=mean_candidate_bpb,
            mean_step_time_ms=mean_candidate_step,
            mean_tokens_per_sec=mean_candidate_tps,
            mean_memory_overhead_bytes=mean_candidate_memory,
            mean_stability_penalty=mean_candidate_stability,
            mean_grad_norm_spikes=mean_candidate_spikes,
            baseline_mean_final_validation_bpb=mean_baseline_bpb,
            baseline_mean_step_time_ms=mean_baseline_step,
            baseline_mean_tokens_per_sec=mean_baseline_tps,
            baseline_mean_memory_overhead_bytes=mean_baseline_memory,
            baseline_mean_stability_penalty=mean_baseline_stability,
            baseline_mean_grad_norm_spikes=mean_baseline_spikes,
            improvement_bpb=improvement,
            speed_ratio=speed_ratio,
            tokens_per_sec_ratio=tokens_per_sec_ratio,
            memory_ratio=memory_ratio,
            time_to_target_ratio=(sum(time_to_target_ratios) / len(time_to_target_ratios)) if time_to_target_ratios else None,
            seed_win_rate=seed_win_rate,
            win_assessment=assessment,
            notes=notes,
        )

    def run(self) -> TournamentSummary:
        baseline = self._baseline_record()
        next_candidate_number = 1

        for generation in range(1, self.search_config.generations + 1):
            parents = sorted(self.archive.survivors(), key=lambda item: item.score, reverse=True)[: self.search_config.survivor_top_k]
            if not parents:
                parents = [baseline]

            generation_records: list[CandidateRecord] = []
            for slot in range(self.search_config.candidates_per_generation):
                parent = parents[slot % len(parents)]
                spec, lineage = mutate_spec(MatrixOptimizerSpec.from_dict(parent.spec), self.rng)
                candidate_id = f"cand_{next_candidate_number:04d}"
                next_candidate_number += 1
                outcome = self.backend.evaluate(spec, seed=self.evaluation_config.seed, candidate_id=candidate_id)
                record = CandidateRecord(
                    id=candidate_id,
                    generation=generation,
                    parent_id=parent.id,
                    spec=spec.to_dict(),
                    score=composite_score(outcome, baseline.primary_outcome, self.search_config),
                    pareto=False,
                    promoted=False,
                    status="dead" if not outcome.valid else "survivor",
                    primary_outcome=outcome,
                    win_assessment=analyze_win_hierarchy(outcome, baseline.primary_outcome, self.search_config),
                    lineage=lineage,
                    failure_type=outcome.failure_type,
                )
                generation_records.append(record)
                self.archive.add(record)

            valid_records = [record for record in generation_records if record.primary_outcome.valid]
            frontier = pareto_frontier(valid_records, baseline.primary_outcome) if valid_records else []
            frontier_ids = {record.id for record in frontier}
            for record in generation_records:
                record.pareto = record.id in frontier_ids

            promoted_pool = sorted(frontier or valid_records, key=lambda item: item.score, reverse=True)[: self.search_config.promotion_top_k]
            survivor_ids = {baseline.id}
            for record in promoted_pool:
                record.promoted = True
                promotion = self._promotion_compare(record, baseline)
                record.promotion_result = promotion
                record.win_assessment = promotion.win_assessment
                record.status = "winner" if promotion.winner else "survivor"
                survivor_ids.add(record.id)

            survivor_ids.update(
                record.id
                for record in sorted(valid_records, key=lambda item: item.score, reverse=True)[: self.search_config.survivor_top_k]
            )
            self.archive.prune(survivor_ids)

            generation_payload = {
                "generation": generation,
                "frontier_ids": sorted(frontier_ids),
                "promoted_ids": [record.id for record in promoted_pool],
                "survivor_ids": sorted(survivor_ids),
            }
            (self.run_dir / f"generation_{generation:04d}.json").write_text(json.dumps(generation_payload, indent=2))

        self.archive.persist(self.run_dir)
        best = self.archive.best()
        winners = [record.id for record in self.archive.records if record.status == "winner"]
        pareto_ids = [record.id for record in self.archive.records if record.pareto]
        summary = TournamentSummary(
            run_dir=str(self.run_dir),
            baseline_candidate_id=baseline.id,
            best_candidate_id=best.id,
            total_candidates=len(self.archive.records),
            winners=winners,
            pareto_frontier=pareto_ids,
        )
        (self.run_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))
        return summary
