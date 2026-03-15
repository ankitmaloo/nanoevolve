"""Integration test: Enigma doctrine loop driving real adamopt DSL mutations.

No LLM calls. We manually walk through every doctrine step, generate
5 real MatrixOptimizerSpec mutations, evaluate them through the toy
backend, score them, and verify the full pipeline end-to-end.

This proves:
1. Search memory round-trips correctly
2. The doctrine stages produce coherent state
3. Real adamopt mutations + evaluations work through Enigma's pipeline
4. Scoring, win hierarchy, and candidate tracking all function
"""
from __future__ import annotations

import json
import random
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from enigma.memory import SearchMemory
from enigma.population import PopulationDB
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
    StageResult,
    EvaluationResult,
)

# Import adamopt directly
from optim_search.spec import MatrixOptimizerSpec
from optim_search.mutations import mutate_spec
from optim_search.eval_candidate import ToyNanoChatBackend
from optim_search.config import EvaluationConfig, SearchConfig as AdamoptSearchConfig
from optim_search.score import composite_score, analyze_win_hierarchy


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def eval_config() -> EvaluationConfig:
    """Small, fast evaluation config for testing."""
    return EvaluationConfig(
        seed=42,
        steps=16,
        eval_every=8,
        batch_size=4,
        seq_len=16,
        vocab_size=32,
        train_eval_batches=1,
        val_eval_batches=2,
        model_dim=16,
        hidden_dim=32,
        layers=1,
        device="cpu",
    )


@pytest.fixture
def backend(eval_config: EvaluationConfig) -> ToyNanoChatBackend:
    return ToyNanoChatBackend(eval_config)


@pytest.fixture
def baseline_spec() -> MatrixOptimizerSpec:
    return MatrixOptimizerSpec.baseline_nanochat()


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    d = tmp_path / "enigma_state"
    d.mkdir()
    return d


# ── Step 1: Map Variables ───────────────────────────────────────

def test_step1_map_variables(state_dir: Path) -> None:
    """Manually populate variables — the search space mapping."""
    memory = SearchMemory(state_dir)
    memory.current_loop = 1

    memory.variables = {
        "task_snapshot": {
            "target": "NanoChat MuonAdamW optimizer",
            "primary_metric": "final_validation_bpb (lower is better)",
            "secondary_metrics": [
                "best_validation_bpb", "train_curve_auc",
                "tokens_per_sec", "stability_penalty",
            ],
            "baseline_score": None,  # will be filled after eval
            "hardware": "CPU (toy), GPU A100 (real)",
            "framework": "PyTorch",
            "dominant_workload_shape": "matrix-only 2D params (attention, MLP projections)",
            "hard_constraints": [
                "matrix_only=True (only 2D params use evolved optimizer)",
                "gate coefficients ∈ [-8, +8]",
                "momentum ∈ [0, 1)",
                "ns_steps ∈ [1, 7]",
                "beta2 ∈ [0.85, 0.99]",
            ],
        },
        "immutable_constraints": {
            "semantic_invariants": [
                "Embeddings, layer norms, scalars stay on AdamW",
                "All specs must pass MatrixOptimizerSpec.validate()",
            ],
            "numerical_tolerances": ["no NaN/Inf in parameters or loss"],
        },
        "controllable_variables": {
            "momentum_placement": ["pre_orthogonal", "post_orthogonal"],
            "trust_ratio": ["none", "layerwise (clamp adjustable)"],
            "clip_policy": ["none", "update_rms", "global_norm"],
            "decay_mode": ["cautious", "decoupled", "none"],
            "ns_steps": "1-7 (Polar Express iterations)",
            "second_moment_beta2": "0.85-0.99",
            "update_multiplier": "0.1+ (scaled by ×0.85-1.15)",
            "stateful_control": "full gate + actuator system",
            "gate_coefficients": "6 sensors × continuous weights",
            "actuator_ranges": "5 actuators × aggressive/conservative bounds",
        },
        "known_bottlenecks": [
            {
                "bottleneck": "orthogonalization cost (Polar Express ns_steps)",
                "evidence": "each step = matrix mult chain",
                "confidence": "high",
            },
            {
                "bottleneck": "training stability at higher learning rates",
                "evidence": "NaN/Inf failures in aggressive configs",
                "confidence": "high",
            },
        ],
        "fake_win_risks": [
            "Winning only on one seed (seed-specific overfitting)",
            "Winning on train but not validation",
            "Speed win from trivially weaker optimizer (less work = faster but worse)",
        ],
    }

    memory.save()
    reloaded = SearchMemory(state_dir)
    assert reloaded.variables["task_snapshot"]["target"] == "NanoChat MuonAdamW optimizer"


# ── Step 2: Map Attack Surfaces ─────────────────────────────────

def test_step2_map_surfaces(state_dir: Path) -> None:
    """Manually define attack surfaces for the optimizer search."""
    memory = SearchMemory(state_dir)
    memory.current_loop = 1

    surfaces = {
        "S01": AttackSurface(
            id="S01",
            region="momentum_pipeline",
            bottleneck="momentum placement affects orthogonalization quality",
            change_class="structural",
            mechanism="Toggle pre/post orthogonal momentum to change gradient processing order",
            leverage=3, plausibility=4, observability=4,
            implementation_cost=1, overlap_risk=1,
            notes="Binary toggle — cheap scout",
        ),
        "S02": AttackSurface(
            id="S02",
            region="trust_ratio_system",
            bottleneck="layer-wise update scale mismatch",
            change_class="policy",
            mechanism="Add layerwise trust ratio to normalize updates across layers",
            leverage=4, plausibility=4, observability=4,
            implementation_cost=2, overlap_risk=2,
            notes="Adaptive per-layer scaling",
        ),
        "S03": AttackSurface(
            id="S03",
            region="gradient_clipping",
            bottleneck="gradient spikes destabilizing training",
            change_class="policy",
            mechanism="Enable update RMS or global norm clipping",
            leverage=3, plausibility=5, observability=5,
            implementation_cost=1, overlap_risk=1,
            notes="Stability focused",
        ),
        "S04": AttackSurface(
            id="S04",
            region="stateful_gate_system",
            bottleneck="static optimizer ignoring training phase dynamics",
            change_class="state",
            mechanism="Enable stateful control with loss/gradient sensors driving adaptive behavior",
            leverage=5, plausibility=3, observability=3,
            implementation_cost=4, overlap_risk=2,
            notes="The big bet — training-phase-aware optimization",
        ),
        "S05": AttackSurface(
            id="S05",
            region="orthogonalization_depth",
            bottleneck="ns_steps trades off approximation quality vs compute cost",
            change_class="schedule",
            mechanism="Adjust ns_steps to find optimal quality/speed tradeoff",
            leverage=4, plausibility=4, observability=5,
            implementation_cost=1, overlap_risk=1,
            notes="Direct compute/quality knob",
        ),
        "S06": AttackSurface(
            id="S06",
            region="second_moment_estimator",
            bottleneck="beta2 controls variance of gradient signal",
            change_class="policy",
            mechanism="Adjust beta2 to change gradient noise filtering",
            leverage=3, plausibility=4, observability=3,
            implementation_cost=1, overlap_risk=2,
        ),
        "S07": AttackSurface(
            id="S07",
            region="decay_policy",
            bottleneck="weight decay mode affects generalization",
            change_class="policy",
            mechanism="Switch between cautious/decoupled/none decay modes",
            leverage=3, plausibility=4, observability=4,
            implementation_cost=1, overlap_risk=1,
        ),
    }

    memory.surfaces = surfaces
    memory.save()
    reloaded = SearchMemory(state_dir)
    assert len(reloaded.surfaces) == 7
    assert reloaded.surfaces["S04"].leverage == 5


# ── Step 3: Generate Hypotheses ─────────────────────────────────

def test_step3_generate_hypotheses(state_dir: Path) -> None:
    """Manually generate 10 hypotheses from the attack surfaces."""
    memory = SearchMemory(state_dir)
    memory.current_loop = 1

    hypotheses = {
        "H01": Hypothesis(
            id="H01",
            title="Post-orthogonal momentum improves gradient signal",
            status=HypothesisStatus.PROPOSED,
            family="momentum",
            bottleneck_attacked="momentum placement order",
            mechanism="Move momentum to post-orthogonal so orthogonalization sees raw gradients",
            code_change_class="structural",
            expected_win="Cleaner orthogonalized updates → better validation BPB",
            main_risk="May destabilize if momentum was dampening noise",
            evidence_needed="Validation BPB improves with stable training",
            disproof_signal="No BPB change or stability degrades",
            cheapest_test="Toggle momentum_placement to post_orthogonal",
            upside=3, feasibility=5, distinctness=4, information_gain=4, transferability=3,
        ),
        "H02": Hypothesis(
            id="H02",
            title="Layerwise trust ratio normalizes cross-layer update scale",
            status=HypothesisStatus.PROPOSED,
            family="trust_ratio",
            bottleneck_attacked="layer-wise update magnitude mismatch",
            mechanism="Enable layerwise trust ratio with tight clamp [0.5, 1.5]",
            code_change_class="policy",
            expected_win="More uniform per-layer optimization → smoother loss descent",
            main_risk="Clamp too tight → suppresses useful variance",
            evidence_needed="Loss curve smoother, val BPB improves",
            disproof_signal="No improvement or regression on val BPB",
            cheapest_test="Enable trust ratio with default clamps",
            upside=4, feasibility=5, distinctness=5, information_gain=4, transferability=4,
        ),
        "H03": Hypothesis(
            id="H03",
            title="Update RMS clipping prevents gradient explosions",
            status=HypothesisStatus.PROPOSED,
            family="clipping",
            bottleneck_attacked="gradient spike instability",
            mechanism="Enable update_rms clipping at threshold=1.0",
            code_change_class="policy",
            expected_win="Fewer grad spikes → stable training → better final BPB",
            main_risk="Clipping too aggressively slows learning",
            evidence_needed="Stability penalty drops, BPB does not regress",
            disproof_signal="BPB regresses or no stability improvement",
            cheapest_test="Toggle clip to update_rms",
            upside=3, feasibility=5, distinctness=4, information_gain=3, transferability=4,
        ),
        "H04": Hypothesis(
            id="H04",
            title="Stateful gate adapts optimizer to training phase",
            status=HypothesisStatus.PROPOSED,
            family="stateful_control",
            bottleneck_attacked="static optimizer ignores training dynamics",
            mechanism="Enable full stateful control: loss/gradient sensors → gate → actuators",
            code_change_class="state",
            expected_win="Phase-aware updates: aggressive early, conservative late",
            main_risk="Gate may track noise, adding overhead without benefit",
            evidence_needed="Distinct gate behavior across training, BPB improves",
            disproof_signal="Gate stays flat or BPB doesn't improve",
            cheapest_test="Enable stateful control with default config",
            upside=5, feasibility=4, distinctness=5, information_gain=5, transferability=5,
        ),
        "H05": Hypothesis(
            id="H05",
            title="Fewer Polar Express steps trade quality for speed",
            status=HypothesisStatus.PROPOSED,
            family="orthogonalization",
            bottleneck_attacked="orthogonalization compute cost",
            mechanism="Reduce ns_steps from 5 to 3 — Polar Express converges fast",
            code_change_class="schedule",
            expected_win="Faster steps with minimal quality loss → better time-to-target",
            main_risk="Approximation degrades enough to hurt final BPB",
            evidence_needed="Step time drops >10% with <0.01 BPB regression",
            disproof_signal="BPB degrades more than speed gained",
            cheapest_test="Set ns_steps=3",
            upside=4, feasibility=5, distinctness=4, information_gain=4, transferability=3,
        ),
        "H06": Hypothesis(
            id="H06",
            title="Higher beta2 smooths second moment for better signal",
            status=HypothesisStatus.PROPOSED,
            family="second_moment",
            bottleneck_attacked="gradient noise in second moment estimate",
            mechanism="Raise beta2 from 0.95 to 0.97 for smoother RMS",
            code_change_class="policy",
            expected_win="Less noisy RMS → better update scaling",
            main_risk="Too smooth → slow to adapt to distribution shifts",
            evidence_needed="Validation BPB improves",
            disproof_signal="No change or regression",
            cheapest_test="Set beta2=0.97",
            upside=3, feasibility=5, distinctness=3, information_gain=3, transferability=3,
        ),
        "H07": Hypothesis(
            id="H07",
            title="Decoupled weight decay improves generalization",
            status=HypothesisStatus.PROPOSED,
            family="decay",
            bottleneck_attacked="cautious decay may be too conservative",
            mechanism="Switch to decoupled decay — standard AdamW-style",
            code_change_class="policy",
            expected_win="Better regularization → lower val BPB",
            main_risk="May increase overfitting if cautious was correct",
            evidence_needed="Val BPB improves without train BPB regression",
            disproof_signal="Train-val gap increases",
            cheapest_test="Toggle decay to decoupled",
            upside=3, feasibility=5, distinctness=3, information_gain=3, transferability=4,
        ),
        "H08": Hypothesis(
            id="H08",
            title="Combined trust ratio + clipping stabilizes aggressive training",
            status=HypothesisStatus.PROPOSED,
            family="compound",
            bottleneck_attacked="interaction between layer normalization and gradient control",
            mechanism="Enable both trust ratio and update_rms clipping",
            code_change_class="policy",
            expected_win="Compound stability → enables more aggressive learning",
            main_risk="Over-regularization kills learning rate",
            evidence_needed="Both stability and BPB improve",
            disproof_signal="BPB regresses below either-alone baseline",
            cheapest_test="Enable both trust_ratio=layerwise and clip=update_rms",
            upside=4, feasibility=4, distinctness=4, information_gain=4, transferability=3,
        ),
        "H09": Hypothesis(
            id="H09",
            title="Gate bias shift changes default aggressiveness",
            status=HypothesisStatus.PROPOSED,
            family="stateful_control",
            bottleneck_attacked="default gate output may be wrong for training start",
            mechanism="Adjust gate bias to start more aggressive (bias=+0.5)",
            code_change_class="state",
            expected_win="Better initial learning phase → lower final BPB",
            main_risk="Too aggressive early → instability",
            evidence_needed="Early loss descent faster, final BPB competitive",
            disproof_signal="NaN/Inf failures or stability regression",
            cheapest_test="Enable stateful with bias=+0.5",
            upside=3, feasibility=4, distinctness=3, information_gain=3, transferability=3,
        ),
        "H10": Hypothesis(
            id="H10",
            title="Scaled update multiplier amplifies effective learning rate",
            status=HypothesisStatus.PROPOSED,
            family="update_scaling",
            bottleneck_attacked="default update magnitude may be suboptimal",
            mechanism="Scale update_multiplier to 1.15× for more aggressive updates",
            code_change_class="policy",
            expected_win="Faster convergence at acceptable risk",
            main_risk="Overshoot → instability",
            evidence_needed="BPB improves without stability regression",
            disproof_signal="Grad spikes increase or NaN failures",
            cheapest_test="Set update_multiplier=1.15",
            upside=3, feasibility=5, distinctness=2, information_gain=2, transferability=3,
        ),
    }

    memory.hypotheses = hypotheses
    memory.save()
    reloaded = SearchMemory(state_dir)
    assert len(reloaded.hypotheses) == 10


# ── Step 4: Prune Hypotheses ────────────────────────────────────

def test_step4_prune_hypotheses(state_dir: Path) -> None:
    """Prune: kill weak, shortlist strong."""
    memory = SearchMemory(state_dir)
    # Load from step 3
    test_step3_generate_hypotheses(state_dir)
    memory = SearchMemory(state_dir)

    # Kill: H10 is pure knob tuning, H09 overlaps H04 (both stateful_control)
    memory.hypotheses["H10"] = replace(
        memory.hypotheses["H10"],
        status=HypothesisStatus.KILLED,
        notes="Killed: pure scalar retune with no mechanism",
    )
    memory.hypotheses["H09"] = replace(
        memory.hypotheses["H09"],
        status=HypothesisStatus.KILLED,
        notes="Killed: overlaps H04 (same stateful_control surface)",
    )

    # Shortlist the rest
    for hid in ["H01", "H02", "H03", "H04", "H05", "H06", "H07", "H08"]:
        memory.hypotheses[hid] = replace(
            memory.hypotheses[hid],
            status=HypothesisStatus.SHORTLISTED,
        )

    memory.save()
    reloaded = SearchMemory(state_dir)
    assert len(reloaded.active_hypotheses()) == 8
    assert len(reloaded.killed_hypotheses()) == 2


# ── Step 5: Find Gaps ──────────────────────────────────────────

def test_step5_find_gaps(state_dir: Path) -> None:
    """Check coverage gaps."""
    memory = SearchMemory(state_dir)
    test_step4_prune_hypotheses(state_dir)
    memory = SearchMemory(state_dir)

    memory.gaps["G01"] = GapEntry(
        id="G01",
        uncovered_area="No hypothesis tests schedule invention over training phase",
        why_it_matters="Optimizer behavior should change from early exploration to late convergence",
        current_portfolio_miss="All current hypotheses are static configs",
        candidate_hypothesis="H04 partially covers this via stateful gate",
        expected_information_gain="high — reveals if phase-awareness matters",
        missing_evidence="Gate telemetry across training phases",
        urgency="medium",
    )

    memory.save()
    assert len(memory.gaps) == 1


# ── Step 6: Select Portfolio ────────────────────────────────────

def test_step6_select_portfolio(state_dir: Path) -> None:
    """Select top 5 with explicit roles."""
    memory = SearchMemory(state_dir)
    test_step5_find_gaps(state_dir)
    memory = SearchMemory(state_dir)

    portfolio = [
        PortfolioSlot(
            id="P1", role=ExperimentRole.SCOUT,
            hypothesis_id="H01",
            family="momentum", target_region="momentum_pipeline",
            operator_family="structural",
            why_selected="Cheap binary toggle, broad information on momentum order",
            expected_signal="BPB change from momentum placement swap",
            acceptance_test="Val BPB improves ≥0.005",
            kill_condition="No change or stability regression",
            overlap_check="unique — only momentum surface",
        ),
        PortfolioSlot(
            id="P2", role=ExperimentRole.SCOUT,
            hypothesis_id="H02",
            family="trust_ratio", target_region="trust_ratio_system",
            operator_family="policy",
            why_selected="Tests if cross-layer normalization helps",
            expected_signal="Smoother loss curve, better val BPB",
            acceptance_test="Val BPB improves, no stability regression",
            kill_condition="No improvement after full run",
            overlap_check="unique — only trust ratio surface",
        ),
        PortfolioSlot(
            id="P3", role=ExperimentRole.EXPLOIT,
            hypothesis_id="H04",
            family="stateful_control", target_region="stateful_gate_system",
            operator_family="state",
            why_selected="Highest upside — phase-aware optimization",
            expected_signal="Gate varies across training, BPB improves",
            acceptance_test="Val BPB improves AND gate telemetry shows phase tracking",
            kill_condition="Gate stays flat or BPB regresses",
            overlap_check="unique — only stateful surface",
        ),
        PortfolioSlot(
            id="P4", role=ExperimentRole.EXPLOIT,
            hypothesis_id="H05",
            family="orthogonalization", target_region="orthogonalization_depth",
            operator_family="schedule",
            why_selected="Direct compute/quality tradeoff — tests if 5 steps is overkill",
            expected_signal="Faster step time with minimal BPB cost",
            acceptance_test="Step time drops >10%, BPB regression <0.01",
            kill_condition="BPB drops more than speed gained",
            overlap_check="unique — only orthogonalization surface",
        ),
        PortfolioSlot(
            id="P5", role=ExperimentRole.WILDCARD,
            hypothesis_id="H08",
            family="compound", target_region="trust_ratio+clipping",
            operator_family="policy",
            why_selected="High-risk compound test — do stability features compound?",
            expected_signal="Better stability AND quality than either alone",
            acceptance_test="Beats both H02 and H03 individually",
            kill_condition="Worse than single-feature variants",
            overlap_check="partial overlap with P2 (trust ratio) — justified as compound test",
        ),
    ]

    memory.portfolio = portfolio
    # Mark selected hypotheses as active
    for slot in portfolio:
        hid = slot.hypothesis_id
        if hid in memory.hypotheses:
            memory.hypotheses[hid] = replace(
                memory.hypotheses[hid], status=HypothesisStatus.ACTIVE,
            )

    memory.save()
    reloaded = SearchMemory(state_dir)
    assert len(reloaded.portfolio) == 5
    roles = [s.role for s in reloaded.portfolio]
    assert ExperimentRole.SCOUT in roles
    assert ExperimentRole.EXPLOIT in roles
    assert ExperimentRole.WILDCARD in roles


# ── Steps 7-8: Generate Real DSL Mutations & Evaluate ───────────

def _make_mutation_for_slot(
    slot: PortfolioSlot,
    baseline: MatrixOptimizerSpec,
) -> tuple[MatrixOptimizerSpec, dict[str, object]]:
    """Generate a concrete MatrixOptimizerSpec mutation for each portfolio slot.

    This is the critical bridge: Enigma doctrine → real adamopt DSL mutation.
    No LLM needed — the doctrine tells us WHAT to mutate, and the adamopt
    spec system tells us HOW.
    """
    if slot.hypothesis_id == "H01":
        # Toggle momentum placement
        spec = replace(
            baseline,
            name="enigma_H01_post_momentum",
            momentum_placement="post_orthogonal",
            metadata={"enigma_hypothesis": "H01", "enigma_slot": slot.id},
        )
        lineage = {"mutation": "toggle_momentum_placement", "value": "post_orthogonal"}
        return spec, lineage

    elif slot.hypothesis_id == "H02":
        # Enable layerwise trust ratio
        from optim_search.spec import TrustRatioConfig
        spec = replace(
            baseline,
            name="enigma_H02_trust_ratio",
            trust_ratio=TrustRatioConfig(mode="layerwise", clamp_min=0.5, clamp_max=1.5),
            metadata={"enigma_hypothesis": "H02", "enigma_slot": slot.id},
        )
        lineage = {"mutation": "enable_trust_ratio", "mode": "layerwise"}
        return spec, lineage

    elif slot.hypothesis_id == "H04":
        # Enable stateful control (the big one)
        spec = MatrixOptimizerSpec.stateful_annealing_variant()
        spec = replace(
            spec,
            name="enigma_H04_stateful_gate",
            metadata={"enigma_hypothesis": "H04", "enigma_slot": slot.id},
        )
        lineage = {"mutation": "enable_stateful_control", "full_config": True}
        return spec, lineage

    elif slot.hypothesis_id == "H05":
        # Reduce ns_steps from 5 to 3
        spec = replace(
            baseline,
            name="enigma_H05_ns3",
            ns_steps=3,
            metadata={"enigma_hypothesis": "H05", "enigma_slot": slot.id},
        )
        lineage = {"mutation": "adjust_ns_steps", "ns_steps": 3}
        return spec, lineage

    elif slot.hypothesis_id == "H08":
        # Compound: trust ratio + clipping
        from optim_search.spec import TrustRatioConfig, ClipConfig
        spec = replace(
            baseline,
            name="enigma_H08_compound_stability",
            trust_ratio=TrustRatioConfig(mode="layerwise", clamp_min=0.5, clamp_max=1.5),
            clip=ClipConfig(mode="update_rms", threshold=1.0),
            metadata={"enigma_hypothesis": "H08", "enigma_slot": slot.id},
        )
        lineage = {"mutation": "compound_trust_ratio_clipping"}
        return spec, lineage

    else:
        raise ValueError(f"Unknown hypothesis: {slot.hypothesis_id}")


class TestFullDoctrineLoop:
    """End-to-end test: manual doctrine steps → real mutations → real evaluation."""

    def test_full_pipeline(
        self,
        state_dir: Path,
        backend: ToyNanoChatBackend,
        baseline_spec: MatrixOptimizerSpec,
        eval_config: EvaluationConfig,
    ) -> None:
        """Walk through all doctrine steps and verify pipeline integrity."""

        # ── Setup: populate search memory (steps 1-6) ───────────
        memory = SearchMemory(state_dir)
        memory.current_loop = 1
        population = PopulationDB(seed=42, maximize=False)  # lower BPB = better

        # Evaluate baseline
        print("\n=== Evaluating baseline ===")
        baseline_outcome = backend.evaluate(
            baseline_spec, seed=eval_config.seed, candidate_id="baseline",
        )
        assert baseline_outcome.valid, f"Baseline failed: {baseline_outcome.failure_type}"
        print(f"  Baseline val BPB: {baseline_outcome.metrics.final_validation_bpb:.4f}")
        print(f"  Baseline step time: {baseline_outcome.metrics.mean_step_time_ms:.2f}ms")

        # Add baseline to population
        baseline_cand = Candidate(
            id="cand_0000", generation=0, loop=0,
            parent_id=None, source=baseline_spec.to_json(),
            aggregate_score=baseline_outcome.metrics.final_validation_bpb,
            metrics={
                "final_validation_bpb": baseline_outcome.metrics.final_validation_bpb,
                "best_validation_bpb": baseline_outcome.metrics.best_validation_bpb,
                "tokens_per_sec": baseline_outcome.metrics.tokens_per_sec,
                "stability_penalty": baseline_outcome.metrics.stability_penalty,
                "mean_step_time_ms": baseline_outcome.metrics.mean_step_time_ms,
            },
            failure_reasons=[], stage_results=[],
            active=True, lineage={"mutation": "baseline"},
        )
        population.add(baseline_cand)

        # Populate surfaces
        test_step2_map_surfaces(state_dir)
        memory = SearchMemory(state_dir)

        # Populate hypotheses (steps 3-4)
        test_step4_prune_hypotheses(state_dir)
        memory = SearchMemory(state_dir)

        # Populate gaps (step 5)
        test_step5_find_gaps(state_dir)
        memory = SearchMemory(state_dir)

        # Build portfolio (step 6)
        test_step6_select_portfolio(state_dir)
        memory = SearchMemory(state_dir)

        assert len(memory.portfolio) == 5

        # ── Step 7-8: Generate mutations and evaluate ───────────
        print("\n=== Generating and evaluating 5 mutations ===")
        results: list[dict[str, object]] = []

        for slot in memory.portfolio:
            # Register mutation neighborhood (step 7)
            n_id = memory.next_neighborhood_id()
            neighborhood = MutationNeighborhood(
                id=n_id, loop=1,
                hypothesis_id=slot.hypothesis_id,
                target_region=slot.target_region,
                operator_family=slot.operator_family,
                bottleneck_attacked=slot.family,
                benchmark_slice="all",
                parent_candidate="cand_0000",
                overlap_status=slot.overlap_check,
                why_allowed=slot.why_selected,
            )
            memory.mutation_ledger[n_id] = neighborhood

            # Generate concrete DSL mutation (step 8)
            mutated_spec, lineage = _make_mutation_for_slot(slot, baseline_spec)

            # Validate the spec (DSL bounds check)
            mutated_spec.validate()

            # Round-trip test (critical for reproducibility)
            spec_dict = mutated_spec.to_dict()
            roundtripped = MatrixOptimizerSpec.from_dict(spec_dict)
            assert roundtripped.stable_id() == mutated_spec.stable_id(), \
                f"Round-trip failed for {slot.id}"

            # Evaluate through toy backend
            outcome = backend.evaluate(
                mutated_spec, seed=eval_config.seed,
                candidate_id=f"enigma_{slot.id}",
            )

            # Score against baseline
            score = composite_score(outcome, baseline_outcome)
            win = analyze_win_hierarchy(outcome, baseline_outcome)

            # Record in population
            cand_id = population.next_id()
            cand = Candidate(
                id=cand_id,
                generation=1, loop=1,
                parent_id="cand_0000",
                source=mutated_spec.to_json(),
                aggregate_score=outcome.metrics.final_validation_bpb if outcome.valid else 999.0,
                metrics={
                    "final_validation_bpb": outcome.metrics.final_validation_bpb if outcome.valid else 999.0,
                    "composite_score": score,
                    "hierarchy_level": win.hierarchy_level,
                    "winner": win.winner,
                },
                failure_reasons=[outcome.failure_type] if outcome.failure_type else [],
                stage_results=[],
                hypothesis_id=slot.hypothesis_id,
                portfolio_slot=slot.id,
                active=outcome.valid,
                lineage=lineage,
            )
            population.add(cand)

            result = {
                "slot": slot.id,
                "hypothesis": slot.hypothesis_id,
                "role": slot.role.value,
                "spec_name": mutated_spec.name,
                "valid": outcome.valid,
                "val_bpb": outcome.metrics.final_validation_bpb if outcome.valid else None,
                "composite_score": score,
                "winner": win.winner,
                "hierarchy_level": win.hierarchy_level,
                "dominant_axes": win.dominant_axes,
                "step_time_ms": outcome.metrics.mean_step_time_ms if outcome.valid else None,
                "stability_penalty": outcome.metrics.stability_penalty if outcome.valid else None,
            }
            results.append(result)

            valid_str = "OK" if outcome.valid else "FAIL"
            bpb_str = f"{outcome.metrics.final_validation_bpb:.4f}" if outcome.valid else "N/A"
            print(f"  [{valid_str}] {slot.id} ({slot.role.value}): "
                  f"{mutated_spec.name} → BPB={bpb_str}, "
                  f"score={score:.2f}, win={win.winner}")

        # ── Verify results ──────────────────────────────────────
        print(f"\n=== Results Summary ===")

        # All 5 mutations should produce valid specs
        assert all(r["valid"] for r in results), \
            f"Some mutations produced invalid results: {[r for r in results if not r['valid']]}"

        # All 5 should have real BPB values (not NaN/Inf)
        for r in results:
            assert r["val_bpb"] is not None
            assert r["val_bpb"] > 0  # BPB should be positive

        # Population should have 6 candidates (1 baseline + 5 mutations)
        assert len(population.candidates) == 6

        # All mutations should have different specs (no accidental duplicates)
        spec_ids = set()
        for r in results:
            spec_json = [c for c in population.candidates if c.portfolio_slot == r["slot"]][0].source
            spec = MatrixOptimizerSpec.from_dict(json.loads(spec_json))
            spec_ids.add(spec.stable_id())
        assert len(spec_ids) == 5, "All 5 mutations should produce unique specs"

        # ── Step 9-11: Evaluate skeptically & postmortem ────────
        # Find best candidate
        valid_results = [r for r in results if r["valid"]]
        best = min(valid_results, key=lambda r: r["val_bpb"])
        worst = max(valid_results, key=lambda r: r["val_bpb"])

        baseline_bpb = baseline_outcome.metrics.final_validation_bpb
        print(f"  Baseline BPB:    {baseline_bpb:.4f}")
        print(f"  Best candidate:  {best['slot']} ({best['hypothesis']}) "
              f"BPB={best['val_bpb']:.4f} Δ={baseline_bpb - best['val_bpb']:.4f}")
        print(f"  Worst candidate: {worst['slot']} ({worst['hypothesis']}) "
              f"BPB={worst['val_bpb']:.4f} Δ={baseline_bpb - worst['val_bpb']:.4f}")

        # Determine promoted and killed hypotheses
        promoted = [r["hypothesis"] for r in results
                    if r["composite_score"] > 0]
        killed = [r["hypothesis"] for r in results
                  if r["composite_score"] < -5]

        # Record postmortem in memory
        loop_record = LoopRecord(
            loop=1,
            date="2026-03-15",
            target="NanoChat MuonAdamW",
            parent_baseline="cand_0000",
            baseline_metrics={"final_validation_bpb": baseline_bpb},
            active_portfolio=[s.id for s in memory.portfolio],
            overlap_conflicts=[],
            best_candidate=best["slot"],
            worst_candidate=worst["slot"],
            score_movement=f"Best Δ={baseline_bpb - best['val_bpb']:.4f}",
            holdout_movement="N/A (toy backend)",
            stability_movement="all stable",
            what_improved=f"{best['slot']} ({best['hypothesis']}): BPB improved",
            what_regressed=f"{worst['slot']}: worst relative performance" if worst["composite_score"] < 0 else "nothing",
            strongest_evidence=f"{best['hypothesis']} scored {best['composite_score']:.2f}",
            likely_causal_explanation=f"{best['hypothesis']} mechanism worked on toy backend",
            hypotheses_promoted=promoted,
            hypotheses_killed=killed,
            child_hypotheses=[],
            negative_knowledge_added=[],
            neighborhoods_retired=[],
            neighborhoods_reopened=[],
            next_loop_focus="Validate on real NanoChat GPU runs",
        )
        memory.loop_log.append(loop_record)
        memory.save()

        # ── Verify memory persistence ───────────────────────────
        reloaded = SearchMemory(state_dir)
        assert reloaded.current_loop == 1
        assert len(reloaded.portfolio) == 5
        assert len(reloaded.mutation_ledger) == 5
        assert len(reloaded.loop_log) == 1
        assert reloaded.loop_log[0].best_candidate == best["slot"]

        export = reloaded.export_for_prompt()
        assert "Active Hypotheses" in export or "Negative Knowledge" in export or "Portfolio" in export

        print(f"\n=== Pipeline Verification Complete ===")
        print(f"  Memory persisted: {state_dir / 'search_memory.json'}")
        print(f"  Portfolio: {len(reloaded.portfolio)} slots")
        print(f"  Neighborhoods: {len(reloaded.mutation_ledger)} registered")
        print(f"  Loop log: {len(reloaded.loop_log)} entries")
        print(f"  All specs validated, round-tripped, and evaluated.")


class TestMutationSpecIntegrity:
    """Verify that all 5 mutation specs are valid and non-degenerate."""

    def test_all_mutations_validate(self, baseline_spec: MatrixOptimizerSpec) -> None:
        """Every mutation must pass MatrixOptimizerSpec.validate()."""
        test_step6_select_portfolio(Path(tempfile.mkdtemp()))
        memory = SearchMemory(Path(tempfile.mkdtemp()))
        test_step6_select_portfolio(memory.state_dir)
        memory = SearchMemory(memory.state_dir)

        for slot in memory.portfolio:
            spec, lineage = _make_mutation_for_slot(slot, baseline_spec)
            spec.validate()  # Raises on invalid
            # Round-trip
            rt = MatrixOptimizerSpec.from_dict(spec.to_dict())
            assert rt.stable_id() == spec.stable_id()

    def test_mutations_differ_from_baseline(self, baseline_spec: MatrixOptimizerSpec) -> None:
        """Each mutation should differ from the baseline in at least one way."""
        memory = SearchMemory(Path(tempfile.mkdtemp()))
        test_step6_select_portfolio(memory.state_dir)
        memory = SearchMemory(memory.state_dir)

        baseline_id = baseline_spec.stable_id()
        for slot in memory.portfolio:
            spec, _ = _make_mutation_for_slot(slot, baseline_spec)
            assert spec.stable_id() != baseline_id, \
                f"Mutation {slot.id} produced identical spec to baseline"

    def test_adamopt_mutate_spec_integration(self, baseline_spec: MatrixOptimizerSpec) -> None:
        """Verify adamopt's own mutate_spec works with our baseline."""
        rng = random.Random(42)
        seen_names: set[str] = set()
        for _ in range(20):
            mutated, lineage = mutate_spec(baseline_spec, rng)
            mutated.validate()
            assert "mutation" in lineage
            assert "parent_spec" in lineage
            seen_names.add(lineage["mutation"])

        # Should have exercised multiple mutation operators
        assert len(seen_names) >= 3, f"Only saw mutations: {seen_names}"
