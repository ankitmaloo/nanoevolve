"""Stage-specific prompt builders for the Enigma doctrine loop.

Each function maps to one of the 11 prompt templates from the playbook,
but produces structured prompts that work with any target program.
Dynamic context generation fills in the target-specific details.
"""
from __future__ import annotations

import json
from typing import Any

from enigma.types import Candidate, FeedbackBundle


# ── Stage 3: Hypothesis Generator ───────────────────────────────

def build_hypothesis_generator_prompt(
    code_context: str,
    variables_json: str,
    surfaces_json: str,
    negative_knowledge_text: str,
    mutation_ledger_text: str,
    num_hypotheses: int = 10,
) -> str:
    return f"""Generate {num_hypotheses} distinct optimization hypotheses for this target.

{code_context}

## Variable Mapping
{variables_json}

## Attack Surfaces
{surfaces_json}

## Negative Knowledge (failures to avoid)
{negative_knowledge_text if negative_knowledge_text else "None yet."}

## Mutation Ledger (prior neighborhoods)
{mutation_ledger_text if mutation_ledger_text else "None yet."}

Rules:
- Each hypothesis must state mechanism, expected win, risk, and evidence needed.
- Each must state bottleneck attacked, cheapest test, and disproof signal.
- Prefer mechanism-level code changes over pure scalar retuning.
- Allow invention of new state variables or schedules if there is a causal story.
- Spread across different families (memory, scheduling, algorithmic, policy, state, schedule).
- Avoid repeating anything blocked by negative knowledge.
- Avoid producing hypotheses that occupy the same mutation neighborhood.
- Optimize for search coverage, not just plausibility.

For each hypothesis, return:
{{
    "id": "H01",
    "title": string,
    "status": "proposed",
    "family": string,
    "bottleneck_attacked": string,
    "mechanism": string,
    "code_change_class": "structural"|"policy"|"state"|"schedule",
    "expected_win": string,
    "main_risk": string,
    "evidence_needed": string,
    "disproof_signal": string,
    "cheapest_test": string,
    "applies_when": string,
    "avoid_when": string,
    "upside": 1-5,
    "feasibility": 1-5,
    "distinctness": 1-5,
    "information_gain": 1-5,
    "transferability": 1-5,
    "notes": string
}}

Return ONLY valid JSON as a list of hypothesis objects."""


# ── Stage 4: Hypothesis Pruner ──────────────────────────────────

def build_hypothesis_pruner_prompt(
    hypotheses_json: str,
    variables_json: str,
    negative_knowledge_text: str,
    mutation_ledger_text: str,
) -> str:
    return f"""Prune the initial hypothesis list.

## Current Hypotheses
{hypotheses_json}

## Variable Mapping
{variables_json}

## Negative Knowledge
{negative_knowledge_text if negative_knowledge_text else "None yet."}

## Mutation Ledger
{mutation_ledger_text if mutation_ledger_text else "None yet."}

Your job:
- Remove redundant hypotheses
- Kill vague hypotheses with no cheap falsification path
- Kill or merge hypotheses that occupy the same mutation neighborhood
- Mark strong survivors as "shortlisted"

Kill a hypothesis if:
- it has no clear bottleneck
- it has no cheap test
- it overlaps another stronger idea
- it is just a retune with no mechanism
- it cannot be falsified
- it ignores known negative knowledge

Return JSON:
{{
    "kept": [
        {{"id": "H01", "status": "shortlisted", "reason": string}},
        ...
    ],
    "killed": [
        {{"id": "H03", "reason": string}},
        ...
    ]
}}"""


# ── Stage 5: Gap Finder ────────────────────────────────────────

def build_gap_finder_prompt(
    hypotheses_json: str,
    variables_json: str,
    surfaces_json: str,
) -> str:
    return f"""Review the shortlisted hypotheses and find major search-space gaps.

## Shortlisted Hypotheses
{hypotheses_json}

## Variable Mapping
{variables_json}

## Attack Surfaces
{surfaces_json}

Find:
- Important bottlenecks with no live hypothesis
- Overconcentration in one family
- Missing high-information probes
- Missing robustness or holdout-oriented probes

Common gap categories:
- no hypothesis attacks total work
- no hypothesis attacks memory movement
- no hypothesis attacks scheduling or occupancy
- no hypothesis attacks control-flow divergence
- no hypothesis tests a risky but informative move
- no hypothesis checks evaluator brittleness
- no hypothesis addresses common-case specialization

Return JSON:
{{
    "gaps": [
        {{
            "id": "G01",
            "uncovered_area": string,
            "why_it_matters": string,
            "current_portfolio_miss": string,
            "candidate_hypothesis": string,
            "expected_information_gain": string,
            "missing_evidence": string,
            "urgency": "high"|"medium"|"low"
        }}
    ],
    "gap_filling_hypotheses": [
        // 2-3 new hypotheses in the same format as the generator
    ]
}}"""


# ── Stage 6: Portfolio Selector ─────────────────────────────────

def build_portfolio_selector_prompt(
    hypotheses_json: str,
    gaps_json: str,
    negative_knowledge_text: str,
    mutation_ledger_text: str,
    portfolio_size: int = 5,
) -> str:
    return f"""Select the top {portfolio_size} hypotheses for the next loop.

## Shortlisted Hypotheses
{hypotheses_json}

## Gaps
{gaps_json}

## Negative Knowledge
{negative_knowledge_text if negative_knowledge_text else "None yet."}

## Mutation Ledger
{mutation_ledger_text if mutation_ledger_text else "None yet."}

Selection criteria: upside, feasibility, distinctness, information gain, transferability.

Desired mix:
- 2 scout slots for broad new surfaces
- 2 exploit slots for strongest live surfaces
- 1 wildcard slot for a risky or ignored surface

If there is no strong winner yet, prefer 3 scouts, 1 exploit, 1 wildcard.

Each slot must:
- test one main mechanism
- name the exact evidence that would count as success
- name the exact evidence that would kill it
- pass an overlap check against every other slot

Return JSON:
{{
    "portfolio": [
        {{
            "id": "P1",
            "role": "scout"|"exploit"|"ablation"|"wildcard",
            "hypothesis_id": "H01",
            "family": string,
            "target_region": string,
            "operator_family": string,
            "why_selected": string,
            "expected_signal": string,
            "acceptance_test": string,
            "kill_condition": string,
            "overlap_check": string
        }}
    ],
    "rejected": [
        {{"hypothesis_id": "H05", "reason": string, "revive_condition": string}}
    ]
}}"""


# ── Stage 7: Candidate Implementer ──────────────────────────────

def build_candidate_implementer_prompt(
    slot_json: str,
    hypothesis_json: str,
    parent_source: str,
    code_context: str,
    negative_knowledge_text: str,
    search_memory_text: str,
) -> str:
    return f"""Implement exactly one candidate for the assigned portfolio slot.

## Portfolio Slot
{slot_json}

## Hypothesis
{hypothesis_json}

## Target Context
{code_context}

## Negative Knowledge
{negative_knowledge_text if negative_knowledge_text else "None yet."}

## Search Memory
{search_memory_text}

## Current Program (mutable)
{parent_source}

Rules:
- Tie the change directly to the named hypothesis.
- Preserve invariants and EVOLVE-BLOCK markers.
- Avoid move families already marked as failed unless the hypothesis states what is different.
- Prefer a coherent test of the mechanism over broad opportunistic edits.
- If a constant becomes adaptive, explain the governing signal.
- If you introduce a new state variable, explain what hidden regime it captures.
- Do not collide with another active slot's mutation neighborhood.
- Do not claim success from one narrow slice if holdouts regress.

At the end, state:
- what changed
- what hypothesis was tested
- what evidence should confirm or refute it

# SEARCH/REPLACE block rules
Every block must use this exact format:
<<<<<<< SEARCH
<original code to match exactly>
=======
<replacement code>
>>>>>>> REPLACE

ONLY EVER RETURN CODE IN SEARCH/REPLACE BLOCKS, followed by a brief explanation."""


# ── Stage 8: Evaluator Skeptic ──────────────────────────────────

def build_evaluator_skeptic_prompt(
    portfolio_json: str,
    results_json: str,
    variables_json: str,
    loop_context: str,
) -> str:
    return f"""Review the latest candidate results with skepticism.

## Portfolio
{portfolio_json}

## Results
{results_json}

## Variable Mapping
{variables_json}

## Loop Context
{loop_context}

Your job:
- Identify fake wins (overfitting to one slice, exploiting evaluator noise)
- Identify variance or evaluator noise that weakens conclusions
- State which results are trustworthy, tentative, or misleading
- Check if holdout or secondary metrics regressed

For each candidate, assess:
- Is the improvement real or noise?
- Does it generalize (holdouts, other slices)?
- Did variance increase?
- Did correctness degrade?
- Is it large enough to matter?

Return JSON:
{{
    "assessments": [
        {{
            "candidate_id": string,
            "verdict": "trustworthy"|"tentative"|"misleading",
            "evidence": string,
            "concerns": [string],
            "recommended_action": "promote"|"retest"|"kill"|"ablate"
        }}
    ],
    "overall_loop_quality": string,
    "fake_win_warnings": [string]
}}"""


# ── Stage 9: Postmortem Analyst ─────────────────────────────────

def build_postmortem_prompt(
    portfolio_json: str,
    results_json: str,
    hypotheses_json: str,
    negative_knowledge_text: str,
    mutation_ledger_text: str,
) -> str:
    return f"""Analyze this completed loop and extract durable learning.

## Portfolio & Results
{portfolio_json}

## Results Detail
{results_json}

## Hypotheses
{hypotheses_json}

## Negative Knowledge
{negative_knowledge_text if negative_knowledge_text else "None yet."}

## Mutation Ledger
{mutation_ledger_text if mutation_ledger_text else "None yet."}

Write a postmortem with:

1. "score_movement": what happened to the primary metric
2. "holdout_movement": what happened to holdouts/secondary metrics
3. "what_improved": specific gains with evidence
4. "what_regressed": specific regressions with evidence
5. "strongest_evidence": the most informative result this loop
6. "likely_causal_explanation": your best causal theory for the strongest result
7. "hypotheses_promoted": IDs that showed real promise
8. "hypotheses_killed": IDs that should be retired, with reasons
9. "child_hypotheses": new hypotheses spawned by results
10. "negative_knowledge": durable lessons (not one-off accidents)
11. "neighborhoods_to_retire": which mutation neighborhoods are exhausted
12. "neighborhoods_to_reopen": which retired ones should reopen, with reason
13. "next_loop_focus": what the next loop should prioritize

Focus on causal learning, not narrative summary.

Return ONLY valid JSON matching the structure above."""


# ── Stage 10: Cold Start Handoff ────────────────────────────────

def build_cold_start_prompt(search_memory_export: str, code_context: str) -> str:
    return f"""You are taking over an evolutionary optimization search effort.

## Search Memory
{search_memory_export}

## Target Context
{code_context}

Before making any code change, answer these questions:

1. What exactly is being optimized?
2. What metric decides survival?
3. What constitutes a real win vs a fake win?
4. What move families are currently alive?
5. What move families are currently disfavored?
6. Which mutation neighborhoods are occupied or retired?
7. What attack surfaces are still underexplored?
8. Should the next loop invent a new rule, state variable, or schedule?
9. What should the next loop test first and why?

Return JSON:
{{
    "situation_assessment": string,
    "real_win_definition": string,
    "fake_win_risks": [string],
    "alive_families": [string],
    "disfavored_families": [string],
    "underexplored_surfaces": [string],
    "recommended_next_action": string,
    "recommended_hypothesis_type": string,
    "confidence": "high"|"medium"|"low"
}}"""


# ── Stage 11: Search Conductor (Full Loop) ──────────────────────

def build_search_conductor_prompt(
    loop_number: int,
    search_memory_export: str,
    code_context: str,
    last_loop_summary: str | None = None,
) -> str:
    last_section = ""
    if last_loop_summary:
        last_section = f"\n## Last Loop Summary\n{last_loop_summary}"

    return f"""You are the search conductor for loop {loop_number}.

## Search Memory
{search_memory_export}

## Target Context
{code_context}
{last_section}

Run this cycle:
1. Check if variables need updating
2. Check attack surface coverage
3. Generate hypotheses for uncovered areas
4. Prune weak or redundant hypotheses
5. Identify gaps
6. Select top portfolio with explicit scout/exploit/wildcard roles
7. Verify no overlap in mutation neighborhoods

Your objective is to improve the search policy itself over time,
not just find one win.

Return JSON with the full loop plan:
{{
    "variables_update_needed": bool,
    "new_surfaces_to_explore": [string],
    "hypotheses_to_generate": [brief descriptions],
    "portfolio_mix": {{"scouts": int, "exploits": int, "wildcards": int}},
    "focus_areas": [string],
    "avoid_areas": [string],
    "loop_strategy": string
}}"""
