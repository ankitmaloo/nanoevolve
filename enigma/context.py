"""Dynamic context generation — analyzes target code to build rich context.

This is the key differentiator: instead of relying on humans to write
variables.md and attack surfaces, we auto-generate them from code analysis.
The LLM then refines this auto-generated context through the doctrine loop.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from enigma.config import TaskConfig


def detect_language(file_path: Path) -> str:
    """Detect programming language from file extension."""
    ext_map = {
        ".py": "python", ".pyx": "cython",
        ".cu": "cuda", ".cuh": "cuda",
        ".c": "c", ".h": "c",
        ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
        ".rs": "rust",
        ".go": "go",
        ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
        ".java": "java",
        ".swift": "swift",
        ".rb": "ruby",
        ".jl": "julia",
        ".lua": "lua",
        ".zig": "zig",
        ".sh": "bash", ".bash": "bash",
        ".sql": "sql",
    }
    return ext_map.get(file_path.suffix.lower(), "unknown")


def analyze_code_structure(source: str, language: str) -> dict[str, Any]:
    """Extract structural information from source code."""
    lines = source.splitlines()
    analysis: dict[str, Any] = {
        "total_lines": len(lines),
        "blank_lines": sum(1 for l in lines if not l.strip()),
        "comment_lines": 0,
        "functions": [],
        "classes": [],
        "loops": [],
        "evolve_blocks": [],
        "imports": [],
        "hot_regions": [],
    }

    # Detect evolve blocks
    in_block = False
    block_start = 0
    for i, line in enumerate(lines, 1):
        if "EVOLVE-BLOCK-START" in line:
            in_block = True
            block_start = i
        elif "EVOLVE-BLOCK-END" in line:
            if in_block:
                analysis["evolve_blocks"].append({
                    "start": block_start,
                    "end": i,
                    "size": i - block_start - 1,
                })
            in_block = False

    if language == "python":
        _analyze_python(source, lines, analysis)
    elif language in ("c", "cpp", "cuda"):
        _analyze_c_family(source, lines, analysis)
    elif language == "rust":
        _analyze_rust(source, lines, analysis)
    else:
        _analyze_generic(source, lines, analysis)

    return analysis


def _analyze_python(source: str, lines: list[str], analysis: dict[str, Any]) -> None:
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            analysis["comment_lines"] += 1
        if stripped.startswith("def "):
            name = stripped.split("(")[0].replace("def ", "")
            analysis["functions"].append({"name": name, "line": i})
        if stripped.startswith("class "):
            name = stripped.split("(")[0].split(":")[0].replace("class ", "")
            analysis["classes"].append({"name": name, "line": i})
        if stripped.startswith(("for ", "while ")):
            analysis["loops"].append({"line": i, "type": stripped.split()[0]})
        if stripped.startswith(("import ", "from ")):
            analysis["imports"].append(stripped)


def _analyze_c_family(source: str, lines: list[str], analysis: dict[str, Any]) -> None:
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("//"):
            analysis["comment_lines"] += 1
        if re.match(r"(?:(?:void|int|float|double|__global__|__device__|static|unsigned|long|char|short|size_t)\s+)+\w+\s*\(", stripped):
            name_match = re.search(r"\b(\w+)\s*\(", stripped)
            if name_match:
                analysis["functions"].append({"name": name_match.group(1), "line": i})
        if re.match(r"(for|while)\s*\(", stripped):
            analysis["loops"].append({"line": i, "type": stripped.split("(")[0].strip()})
        if stripped.startswith("#include"):
            analysis["imports"].append(stripped)

    # CUDA-specific: detect kernel launches, shared memory
    if "__global__" in source:
        analysis["cuda_kernels"] = []
        for m in re.finditer(r"__global__\s+void\s+(\w+)", source):
            analysis["cuda_kernels"].append(m.group(1))
    if "__shared__" in source:
        analysis["uses_shared_memory"] = True
    if "atomicAdd" in source or "atomicCAS" in source:
        analysis["uses_atomics"] = True


def _analyze_rust(source: str, lines: list[str], analysis: dict[str, Any]) -> None:
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("//"):
            analysis["comment_lines"] += 1
        if stripped.startswith("fn "):
            name = stripped.split("(")[0].replace("fn ", "").replace("pub ", "")
            analysis["functions"].append({"name": name.strip(), "line": i})
        if stripped.startswith(("struct ", "pub struct ")):
            name = stripped.split("{")[0].split("<")[0]
            name = name.replace("pub ", "").replace("struct ", "").strip()
            analysis["classes"].append({"name": name, "line": i})
        if stripped.startswith(("for ", "while ", "loop ")):
            analysis["loops"].append({"line": i, "type": stripped.split()[0]})
        if stripped.startswith("use "):
            analysis["imports"].append(stripped)


def _analyze_generic(source: str, lines: list[str], analysis: dict[str, Any]) -> None:
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith(("//", "#", "--", ";")):
            analysis["comment_lines"] += 1


def identify_hot_regions(source: str, analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Heuristically identify likely performance-critical regions."""
    hot = []
    lines = source.splitlines()

    # Nested loops are often hot
    indent_stack: list[int] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if re.match(r"(for|while)\b", stripped):
            # Check nesting
            while indent_stack and indent_stack[-1] >= indent:
                indent_stack.pop()
            indent_stack.append(indent)
            if len(indent_stack) >= 2:
                hot.append({
                    "line": i,
                    "reason": f"nested loop (depth {len(indent_stack)})",
                    "severity": "high" if len(indent_stack) >= 3 else "medium",
                })

    # Large evolve blocks
    for block in analysis.get("evolve_blocks", []):
        if block["size"] > 20:
            hot.append({
                "line": block["start"],
                "reason": f"large evolve block ({block['size']} lines)",
                "severity": "medium",
            })

    return hot


def generate_task_context(
    task_config: TaskConfig,
    task_dir: Path,
    *,
    include_structure: bool = True,
    include_hot_regions: bool = True,
    max_source_lines: int = 500,
) -> str:
    """Generate rich dynamic context for the LLM from code analysis.

    This replaces the need to manually write CONTEXT.md — it auto-generates
    a comprehensive context document that the doctrine prompts can reference.
    """
    sections: list[str] = []

    # Task overview
    sections.append(f"# Target: {task_config.name}")
    sections.append(f"{task_config.description}")
    sections.append(f"Primary metric: {task_config.primary_metric} "
                    f"({'higher' if task_config.maximize else 'lower'} is better)")
    sections.append(f"All metrics: {', '.join(task_config.metric_keys)}")

    if task_config.hardware:
        sections.append(f"Hardware: {task_config.hardware}")

    if task_config.hard_constraints:
        sections.append("## Hard Constraints")
        for c in task_config.hard_constraints:
            sections.append(f"- {c}")

    # Analyze each mutable file
    for mutable_file in task_config.mutable_files:
        file_path = task_dir / mutable_file
        if not file_path.exists():
            continue

        source = file_path.read_text()
        lang = task_config.language
        if lang == "auto":
            lang = detect_language(file_path)

        sections.append(f"\n## Mutable File: {mutable_file} ({lang})")

        if include_structure:
            analysis = analyze_code_structure(source, lang)
            sections.append(f"Lines: {analysis['total_lines']} "
                          f"(blank: {analysis['blank_lines']}, "
                          f"comments: {analysis['comment_lines']})")

            if analysis["functions"]:
                sections.append("### Functions")
                for f in analysis["functions"]:
                    sections.append(f"- `{f['name']}` (line {f['line']})")

            if analysis["classes"]:
                sections.append("### Classes/Structs")
                for c in analysis["classes"]:
                    sections.append(f"- `{c['name']}` (line {c['line']})")

            if analysis["evolve_blocks"]:
                sections.append("### Evolve Blocks")
                for b in analysis["evolve_blocks"]:
                    sections.append(f"- Lines {b['start']}-{b['end']} ({b['size']} lines)")

            if analysis.get("cuda_kernels"):
                sections.append(f"### CUDA Kernels: {', '.join(analysis['cuda_kernels'])}")

            if include_hot_regions:
                hot = identify_hot_regions(source, analysis)
                if hot:
                    sections.append("### Likely Hot Regions")
                    for h in hot:
                        sections.append(f"- Line {h['line']}: {h['reason']} [{h['severity']}]")

        # Include source (truncated if needed)
        source_lines = source.splitlines()
        if len(source_lines) > max_source_lines:
            sections.append(f"\n### Source (first {max_source_lines} lines)")
            sections.append("```" + lang)
            sections.append("\n".join(source_lines[:max_source_lines]))
            sections.append(f"... ({len(source_lines) - max_source_lines} more lines)")
            sections.append("```")
        else:
            sections.append("\n### Full Source")
            sections.append("```" + lang)
            sections.append(source)
            sections.append("```")

    # Include context files (read-only)
    for ctx_file in task_config.context_files:
        ctx_path = task_dir / ctx_file
        if ctx_path.exists():
            sections.append(f"\n## Context File: {ctx_file} (read-only)")
            content = ctx_path.read_text()
            if len(content.splitlines()) > 200:
                sections.append(content[:4000] + "\n... (truncated)")
            else:
                sections.append(content)

    # Benchmark slices
    if task_config.benchmark_slices:
        sections.append("\n## Benchmark Slices")
        for s in task_config.benchmark_slices:
            sections.append(f"- {s}")

    return "\n".join(sections)


def generate_variables_prompt(
    task_config: TaskConfig,
    task_dir: Path,
    code_context: str,
) -> str:
    """Build the prompt for the Variable Mapper stage.

    This asks the LLM to fill out the variables template from the playbook,
    using the auto-generated code context as input.
    """
    return f"""You are mapping the optimization search space for a target program.

{code_context}

Your job is to produce a structured JSON analysis with these sections:

1. "task_snapshot": {{
    "target": string,
    "primary_metric": string,
    "secondary_metrics": [string],
    "baseline_score": number or null,
    "hardware": string,
    "framework": string,
    "dominant_workload_shape": string,
    "hard_constraints": [string],
    "soft_preferences": [string]
}}

2. "immutable_constraints": {{
    "semantic_invariants": [string],
    "numerical_tolerances": [string],
    "api_constraints": [string],
    "memory_limits": [string],
    "output_format_constraints": [string]
}}

3. "controllable_variables": {{
    "algorithmic_structure": [string],
    "memory_data_movement": [string],
    "parallelism_scheduling": [string],
    "control_flow": [string],
    "numerical_strategy": [string],
    "policy_decision_rules": [string],
    "state_latent_variables": [string],
    "schedules_annealing": [string]
}}

4. "known_bottlenecks": [
    {{"bottleneck": string, "evidence": string, "confidence": "high"|"medium"|"low"}}
]

5. "unknowns_needing_evidence": [
    {{"unknown": string, "why_it_matters": string, "cheapest_resolution": string}}
]

6. "ignored_variables_worth_questioning": [
    {{"variable": string, "why_it_might_matter": string, "mechanism": string, "cheapest_test": string}}
]

7. "what_can_change": {{
    "files": [string],
    "regions": [string],
    "evolve_blocks": [string]
}}

8. "what_must_not_change": {{
    "externally_visible_behavior": [string],
    "interfaces": [string],
    "benchmarking_protocol": [string]
}}

9. "fake_win_risks": [string]

10. "evaluator_blind_spots": [string]

IMPORTANT: Do not propose code changes yet. Do not generate hypotheses yet.
Focus on thoroughly mapping the search space.

Return ONLY valid JSON."""


def generate_surface_mapping_prompt(
    code_context: str,
    variables_json: str,
) -> str:
    """Build the prompt for the Attack Surface Mapper stage."""
    return f"""Map the attack surfaces for this target before generating hypotheses.

{code_context}

## Current Variable Mapping
{variables_json}

Your job is to identify attack surfaces — places where a code change could plausibly move the objective.

For each surface, provide:
- id: "S01", "S02", etc.
- region: which code region
- bottleneck: what bottleneck or failure mode
- change_class: "structural", "policy", "state", or "schedule"
- mechanism: what specific change and why it should help
- leverage: 1-5 (how much upside if correct)
- plausibility: 1-5 (how believable is the mechanism)
- observability: 1-5 (can the evaluator detect the effect)
- implementation_cost: 1-5 (how hard is the first probe)
- overlap_risk: 1-5 (how likely is duplication with another surface)
- notes: any additional context

Surface classes to consider:
1. Structural: loop structure, decomposition, fusion/defusion, dataflow ordering, reuse
2. Memory: locality, staging, layout, cache behavior, recomputation vs storage
3. Parallelism: mapping, tiling, vectorization, occupancy, synchronization
4. Policy: scoring rules, thresholds, tie-breaks, dispatch, update rules
5. State: missing summaries, counters, confidence signals, phase markers
6. Schedule: phase-aware, size-aware, depth-aware, error-aware, annealing
7. Robustness: narrow wins, holdout regressions, correctness drift, variance

Pick 5-8 surfaces for first-pass probing. They should:
- span multiple code regions,
- span multiple change classes,
- include at least one adaptive-rule or latent-state surface,
- include at least one schedule surface if the problem has phases,
- include at least one high-risk, high-information surface.

Label each chosen surface as "scout", "exploit_candidate", or "wildcard".

Return ONLY valid JSON as a list of surface objects."""
