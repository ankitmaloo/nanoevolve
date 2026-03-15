# Implementer Spec

This file is the operating manual for an LLM that must execute the Enigma Playbook with minimal judgment.

Read this file before editing code.

## Mission
Your job is to improve a target program through evolutionary search over code-level mechanisms.

You are not here to:
- do generic brainstorming,
- tune a few constants and stop,
- chase one benchmark win without checking holdouts,
- make broad clever edits with unclear causal meaning.

You are here to:
- propose explicit hypotheses,
- implement one mechanism per candidate,
- avoid overlapping candidates,
- learn from results,
- update the search state so the next loop is better.

## Core Principle
Search mechanisms, not knobs.

Good targets for mutation:
- code structure,
- decision rules,
- state variables,
- schedules,
- adaptive behavior,
- decomposition,
- reuse,
- control flow,
- specialization.

Weak targets for mutation:
- changing constants with no theory,
- threshold twiddling with no evidence,
- cosmetic rewrites,
- multiple unrelated edits in one candidate.

## Read Order
Before touching code, read these files in this exact order:
1. `README.md`
2. `IMPLEMENTER_SPEC.md`
3. `EXPERIMENTATION_STRATEGY.md`
4. `variables.md`
5. `negative_knowledge.md`
6. `mutation_ledger.md`
7. `hypotheses.md`
8. `gaps.md`
9. `portfolio.md`
10. `loop_log.md`
11. `prompt_templates.md`

Then inspect:
- the mutable code,
- the evaluator,
- the task configuration,
- the latest run outputs if present.

If you have not done all of the above, do not edit code.

## Required Outputs
Each loop must update these files:
- `variables.md`
- `hypotheses.md`
- `gaps.md`
- `portfolio.md`
- `mutation_ledger.md`
- `loop_log.md`
- `negative_knowledge.md`

Each loop should also use `EXPERIMENTATION_STRATEGY.md` to check whether surface coverage is broad enough.

## Loop Algorithm
Run this exact procedure.

### Step 1: Map The Search Space
Fill `variables.md`.

You must identify:
- what metric decides survival,
- what counts as correctness,
- what fake wins look like,
- which code regions are mutable,
- which bottlenecks are likely,
- which variables humans may be ignoring,
- which parts of the space are untouched.

If you do not know a bottleneck, write it as an unknown. Do not pretend certainty.

### Step 2: Map Attack Surfaces
Use `EXPERIMENTATION_STRATEGY.md`.

You must identify:
- the main mutable regions,
- the main bottleneck classes,
- the plausible change classes per region,
- which surfaces are scouts,
- which surfaces are exploit candidates,
- where coverage is still weak.

Do not skip this step.

### Step 3: Generate 10 Hypotheses
Write 10 hypotheses in `hypotheses.md`.

Rules:
- Each hypothesis must attack a bottleneck.
- Each hypothesis must have a cheap falsification path.
- Each hypothesis must state its `code change class`.
- Prefer:
  - structural changes,
  - policy changes,
  - latent-state additions,
  - schedule or annealing invention.
- Deprioritize pure knob tuning.

### Step 4: Prune Aggressively
Kill weak hypotheses early.

Kill a hypothesis if:
- it has no clear bottleneck,
- it has no cheap test,
- it overlaps another stronger idea,
- it is just a retune with no mechanism,
- it cannot be falsified,
- it ignores known negative knowledge.

Shortlist only the strong survivors.

### Step 5: Run Gap Analysis
Use `gaps.md`.

Look for missing coverage:
- no memory move,
- no scheduling move,
- no algorithmic move,
- no state invention,
- no schedule invention,
- no risky high-information probe,
- no holdout-oriented probe.

Add 2 or 3 hypotheses only if they close a real gap.

### Step 6: Build A Non-Overlapping Portfolio
Fill `portfolio.md`.

Pick the top 5 hypotheses.

Do not allow two slots to overlap unless one is an intentional ablation pair.

For each slot, define:
- target region,
- operator family,
- experiment role: scout, exploit, or ablation,
- expected signal,
- acceptance test,
- kill condition,
- overlap check.

The default portfolio mix is:
- 2 scouts,
- 2 exploits,
- 1 wildcard.

If there are no credible winners yet, prefer:
- 3 scouts,
- 1 exploit,
- 1 wildcard.

### Step 7: Register Mutation Neighborhoods
Write each slot into `mutation_ledger.md`.

Each entry must state:
- target region,
- operator family,
- bottleneck attacked,
- benchmark slice,
- relation to prior neighborhoods,
- why it is allowed.

If it overlaps a prior neighborhood, explain why the revisit is justified.

### Step 8: Implement One Candidate Per Slot
One slot = one main mechanism.

Allowed:
- introducing a new local variable,
- adding a new rule,
- replacing a constant with an adaptive function,
- adding a schedule,
- introducing a new state tracker,
- restructuring one hot code path.

Not allowed unless explicitly justified:
- multiple unrelated mechanisms in one slot,
- changing code outside the declared target region,
- retuning constants instead of testing the mechanism,
- copying another slot with tiny differences.

### Step 9: Evaluate Skeptically
After evaluation, ask:
- Did the candidate improve the real metric?
- Did holdouts improve too?
- Did variance increase?
- Did correctness degrade?
- Is this only a narrow-case win?
- Is the result large enough to matter?

If the answer is unclear, mark the result as tentative.

### Step 10: Record Causal Learning
Update `loop_log.md`.

You must state:
- what got stronger,
- what got weaker,
- what failed and why,
- what new child hypotheses appeared,
- what neighborhoods should be retired,
- what can be retried only under changed assumptions.

### Step 11: Update Durable Memory
Update:
- `negative_knowledge.md`
- `mutation_ledger.md`

Write only durable lessons.
Do not fill these files with noise.

## Decision Rules
When uncertain, use these defaults.

### If you do not know where to start
Start with one hypothesis from each of these classes:
- memory or locality,
- scheduling or parallelism,
- algorithmic work reduction,
- adaptive rule or latent state,
- risky high-information probe.

Then make sure at least one of those is a scout on a new surface rather than a local optimization of an old surface.

### If two ideas seem equally good
Choose the one with:
1. lower overlap,
2. broader surface coverage,
3. clearer falsification,
4. higher information gain.

### If a result is small
Treat it as weak evidence unless:
- it repeats consistently,
- it improves holdouts,
- it supports a broader mechanism hypothesis.

### If a candidate only changes constants
Assume it is low value.
Only keep it if:
- it is an explicit baseline,
- it isolates a mechanism,
- it supports an ablation.

### If you want to revisit an old idea
You must state what changed:
- new parent code,
- new evidence,
- corrected assumption,
- different region,
- different operator family,
- different benchmark slice.

If nothing changed, do not revisit it.

### If coverage is narrow
Broaden before optimizing.

Coverage is narrow if:
- most live ideas hit one code region,
- most live ideas share one change class,
- there is no state or schedule invention in play,
- there is no scout slot,
- there is no holdout-focused slot.

## Overlap Rules
Two candidates overlap too much if they share all of:
- same target region,
- same bottleneck,
- same mechanism,
- same operator family,
- same benchmark slice focus.

If overlap is high:
- kill one,
- merge them,
- or explicitly mark one as an ablation.

Do not keep accidental duplicates.

## Standard For A Good Candidate
A good candidate:
- tests one clear idea,
- changes behavior through code,
- has a believable causal story,
- has a clear success test,
- has a clear kill condition,
- teaches the system something even if it fails.

## Standard For A Good Loop
A good loop is not just one where the best score goes up.

A good loop:
- reduces uncertainty,
- removes bad ideas quickly,
- expands coverage,
- avoids duplicate work,
- leaves behind reusable search memory.

## Example: Weak vs Strong

Weak:
- "Increase penalty from `0.1` to `0.15`."

Stronger:
- "Replace the fixed penalty with a size-aware rule that scales with local density, because the current code treats dense and sparse regions identically."

Strongest:
- "Add a local density estimator and anneal the penalty over iteration depth so early exploration stays broad and late behavior becomes selective."

## Hard Stops
Stop and reconsider if:
- all five portfolio slots attack the same family,
- all five slots are exploit slots,
- you are mostly tuning constants,
- you cannot explain why a candidate should work,
- you are revisiting retired neighborhoods without new evidence,
- your postmortem contains no real causal claim,
- you are calling narrow wins 'success'.

## Final Checklist Before Code Changes
Before implementing a candidate, confirm:
- the hypothesis id is named,
- the target region is named,
- the bottleneck is named,
- the operator family is named,
- the overlap check passed,
- the acceptance test is named,
- the kill condition is named,
- the mechanism is code-level rather than knob-level.

If any of the above is missing, do not implement yet.
