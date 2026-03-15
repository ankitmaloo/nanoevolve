"""Tests for Enigma core components."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from enigma.config import SearchConfig, TaskConfig
from enigma.context import analyze_code_structure, detect_language, generate_task_context, identify_hot_regions
from enigma.memory import SearchMemory
from enigma.mutator import apply_search_replace_blocks, extract_search_replace_blocks, parse_json_response
from enigma.population import PopulationDB
from enigma.types import (
    AttackSurface,
    Candidate,
    ExperimentRole,
    GapEntry,
    Hypothesis,
    HypothesisStatus,
    MutationNeighborhood,
    NegativeKnowledge,
    PortfolioSlot,
    SurfaceStatus,
)


# ── Language detection ──────────────────────────────────────────

class TestDetectLanguage:
    def test_python(self) -> None:
        assert detect_language(Path("foo.py")) == "python"

    def test_cuda(self) -> None:
        assert detect_language(Path("kernel.cu")) == "cuda"

    def test_rust(self) -> None:
        assert detect_language(Path("main.rs")) == "rust"

    def test_cpp(self) -> None:
        assert detect_language(Path("solver.cpp")) == "cpp"

    def test_unknown(self) -> None:
        assert detect_language(Path("file.xyz")) == "unknown"


# ── Code analysis ───────────────────────────────────────────────

class TestCodeAnalysis:
    def test_python_analysis(self) -> None:
        source = """
import math

# A comment
class Solver:
    def solve(self, x):
        # EVOLVE-BLOCK-START
        for i in range(10):
            x = math.sqrt(x)
        # EVOLVE-BLOCK-END
        return x

def helper():
    while True:
        break
"""
        analysis = analyze_code_structure(source, "python")
        assert analysis["total_lines"] > 0
        assert len(analysis["functions"]) == 2  # solve, helper
        assert len(analysis["classes"]) == 1  # Solver
        assert len(analysis["evolve_blocks"]) == 1
        assert len(analysis["loops"]) >= 2  # for, while
        assert any("import math" in i for i in analysis["imports"])

    def test_c_analysis(self) -> None:
        source = """
#include <stdio.h>

__global__ void my_kernel(float* out, int N) {
    // EVOLVE-BLOCK-START
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < N; i++) {
        out[i] = 0.0f;
    }
    // EVOLVE-BLOCK-END
}
"""
        analysis = analyze_code_structure(source, "cuda")
        assert len(analysis["functions"]) >= 1
        assert len(analysis["evolve_blocks"]) == 1
        assert "my_kernel" in analysis.get("cuda_kernels", [])

    def test_hot_regions(self) -> None:
        source = """
for i in range(100):
    for j in range(100):
        for k in range(100):
            x += 1
"""
        analysis = analyze_code_structure(source, "python")
        hot = identify_hot_regions(source, analysis)
        assert len(hot) >= 1
        assert any(h["severity"] == "high" for h in hot)


# ── Search/Replace blocks ──────────────────────────────────────

class TestSearchReplace:
    def test_extract_blocks(self) -> None:
        text = """Some text
<<<<<<< SEARCH
old code
=======
new code
>>>>>>> REPLACE
more text"""
        blocks = extract_search_replace_blocks(text)
        assert len(blocks) == 1
        assert blocks[0] == ("old code", "new code")

    def test_extract_multiple_blocks(self) -> None:
        text = """
<<<<<<< SEARCH
first old
=======
first new
>>>>>>> REPLACE

<<<<<<< SEARCH
second old
=======
second new
>>>>>>> REPLACE
"""
        blocks = extract_search_replace_blocks(text)
        assert len(blocks) == 2

    def test_apply_blocks(self) -> None:
        source = "hello world\nfoo bar\nbaz qux"
        diff = """
<<<<<<< SEARCH
foo bar
=======
foo replaced
>>>>>>> REPLACE
"""
        result, stats = apply_search_replace_blocks(source, diff)
        assert "foo replaced" in result
        assert "foo bar" not in result
        assert stats["applied"] == 1

    def test_apply_fails_on_no_match(self) -> None:
        source = "hello world"
        diff = """
<<<<<<< SEARCH
nonexistent
=======
replaced
>>>>>>> REPLACE
"""
        from enigma.mutator import MutationError
        with pytest.raises(MutationError):
            apply_search_replace_blocks(source, diff)


# ── JSON parsing ────────────────────────────────────────────────

class TestJsonParsing:
    def test_plain_json(self) -> None:
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced(self) -> None:
        result = parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_json_with_preamble(self) -> None:
        result = parse_json_response('Here is the JSON:\n{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_array(self) -> None:
        result = parse_json_response('[1, 2, 3]')
        assert result == [1, 2, 3]


# ── Population DB ──────────────────────────────────────────────

class TestPopulationDB:
    def _make_candidate(self, cid: str, score: float, active: bool = True) -> Candidate:
        return Candidate(
            id=cid, generation=0, loop=0, parent_id=None,
            source="x = 1", aggregate_score=score,
            metrics={"aggregate_score": score},
            failure_reasons=[], stage_results=[],
            active=active,
        )

    def test_best_maximize(self) -> None:
        db = PopulationDB(seed=42, maximize=True)
        db.add(self._make_candidate("a", 1.0))
        db.add(self._make_candidate("b", 5.0))
        db.add(self._make_candidate("c", 3.0))
        assert db.best().id == "b"

    def test_best_minimize(self) -> None:
        db = PopulationDB(seed=42, maximize=False)
        db.add(self._make_candidate("a", 1.0))
        db.add(self._make_candidate("b", 5.0))
        db.add(self._make_candidate("c", 3.0))
        assert db.best().id == "a"

    def test_prune_survivors(self) -> None:
        db = PopulationDB(seed=42)
        for i in range(10):
            db.add(self._make_candidate(f"c{i}", float(i)))
        survivors, dropped = db.prune_survivors(top_k=3, diversity_slots=0)
        assert len(survivors) == 3
        assert len(dropped) == 7
        assert "c9" in survivors  # highest score

    def test_next_id(self) -> None:
        db = PopulationDB()
        assert db.next_id() == "cand_0001"
        assert db.next_id() == "cand_0002"

    def test_sample_parent(self) -> None:
        db = PopulationDB(seed=42)
        db.add(self._make_candidate("a", 10.0))
        db.add(self._make_candidate("b", 1.0))
        parent = db.sample_parent()
        assert parent.id in ("a", "b")


# ── Search Memory ──────────────────────────────────────────────

class TestSearchMemory:
    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)

            # Create and populate memory
            mem = SearchMemory(state_dir)
            mem.current_loop = 3
            mem.variables = {"task_snapshot": {"target": "test"}}
            mem.surfaces["S01"] = AttackSurface(
                id="S01", region="main_loop", bottleneck="memory",
                change_class="structural", mechanism="tiling",
                leverage=4, plausibility=3, observability=4,
                implementation_cost=2, overlap_risk=1,
            )
            mem.hypotheses["H01"] = Hypothesis(
                id="H01", title="Test hypothesis",
                status=HypothesisStatus.SHORTLISTED,
                family="memory", bottleneck_attacked="cache misses",
                mechanism="tile for L1", code_change_class="structural",
                expected_win="2x throughput", main_risk="register pressure",
                evidence_needed="benchmark improves",
                disproof_signal="no improvement",
                cheapest_test="tile one loop",
            )
            mem.negative_knowledge["NK01"] = NegativeKnowledge(
                id="NK01", move_family="memory",
                observed_failure="shared memory overflow",
                likely_cause="tile too large",
                evidence="loop 2 candidate 5",
                confidence="high",
                do_not_repeat_unless="smaller tile size used",
                revisit_trigger="new memory budget",
            )
            mem.save()

            # Reload
            mem2 = SearchMemory(state_dir)
            assert mem2.current_loop == 3
            assert mem2.variables["task_snapshot"]["target"] == "test"
            assert "S01" in mem2.surfaces
            assert mem2.surfaces["S01"].leverage == 4
            assert "H01" in mem2.hypotheses
            assert mem2.hypotheses["H01"].status == HypothesisStatus.SHORTLISTED
            assert "NK01" in mem2.negative_knowledge

    def test_overlap_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = SearchMemory(Path(tmpdir))
            mem.mutation_ledger["M01"] = MutationNeighborhood(
                id="M01", loop=1, hypothesis_id="H01",
                target_region="kernel", operator_family="structural",
                bottleneck_attacked="memory", benchmark_slice="all",
            )
            overlaps = mem.check_neighborhood_overlap(
                "kernel", "structural", "memory", "all",
            )
            assert len(overlaps) == 1

            no_overlaps = mem.check_neighborhood_overlap(
                "different_region", "policy", "latency", "subset",
            )
            assert len(no_overlaps) == 0

    def test_export_for_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = SearchMemory(Path(tmpdir))
            mem.negative_knowledge["NK01"] = NegativeKnowledge(
                id="NK01", move_family="test",
                observed_failure="failed", likely_cause="reason",
                evidence="evidence", confidence="high",
                do_not_repeat_unless="changed",
                revisit_trigger="trigger",
            )
            text = mem.export_for_prompt()
            assert "NK01" in text
            assert "Negative Knowledge" in text

    def test_next_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = SearchMemory(Path(tmpdir))
            assert mem.next_hypothesis_id() == "H01"
            mem.hypotheses["H01"] = Hypothesis(
                id="H01", title="t", status=HypothesisStatus.PROPOSED,
                family="f", bottleneck_attacked="b", mechanism="m",
                code_change_class="structural", expected_win="w",
                main_risk="r", evidence_needed="e",
                disproof_signal="d", cheapest_test="c",
            )
            assert mem.next_hypothesis_id() == "H02"


# ── TaskConfig ──────────────────────────────────────────────────

class TestTaskConfig:
    def test_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            (task_dir / "task.json").write_text(json.dumps({
                "name": "test_task",
                "description": "A test",
                "seed_file": "program.py",
                "mutable_files": ["program.py"],
                "eval_command": "python eval.py {candidate_file}",
                "metric_keys": ["score"],
                "primary_metric": "score",
            }))
            config = TaskConfig.load(task_dir)
            assert config.name == "test_task"
            assert config.primary_metric == "score"
            assert config.eval_mode == "command"

    def test_resolve_paths(self) -> None:
        config = TaskConfig(
            name="t", description="d", seed_file="prog.py",
            mutable_files=["prog.py"], context_files=["ctx.md"],
        )
        task_dir = Path("/tmp/test")
        assert config.resolve_seed_path(task_dir) == Path("/tmp/test/prog.py").resolve()
        assert len(config.resolve_context_paths(task_dir)) == 1


# ── Dynamic Context Generation ─────────────────────────────────

class TestContextGeneration:
    def test_generate_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            (task_dir / "task.json").write_text(json.dumps({
                "name": "test",
                "description": "Test program",
                "seed_file": "program.py",
                "mutable_files": ["program.py"],
                "eval_command": "python eval.py {candidate_file}",
            }))
            (task_dir / "program.py").write_text("""
def solve(x):
    # EVOLVE-BLOCK-START
    for i in range(100):
        x += 1
    # EVOLVE-BLOCK-END
    return x
""")
            config = TaskConfig.load(task_dir)
            context = generate_task_context(config, task_dir)
            assert "test" in context.lower()
            assert "Evolve Block" in context
            assert "solve" in context
