# Enigma Playbook

This folder is the search doctrine for AlphaEvolve-style optimization.

Its purpose is not to optimize one kernel once. Its purpose is to teach another LLM how to run a disciplined evolutionary search on any target with:
- explicit variables,
- explicit hypotheses,
- explicit gap analysis,
- explicit negative knowledge,
- explicit loop-by-loop learning.

The main search unit is a code-level mechanism change, not a scalar hyperparameter tweak.

## Start Here
Read these in order before making any code change:
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

Then read the target-specific materials:
- mutable code regions,
- evaluator contract,
- task configuration,
- context files,
- latest run artifacts if they exist.

Only after that should you write or edit code.

## Non-Negotiable Rules
- Do not mutate blindly.
- Do not run a candidate unless it is tied to a named hypothesis.
- Do not keep a hypothesis alive just because it sounds plausible.
- Do not repeat a move family that has already failed without stating what is now different.
- Prefer code changes over exposed knob tuning.
- Treat pure coefficient or threshold retuning as low-value unless it is part of testing a larger mechanism.
- It is valid to invent a new internal variable, state tracker, or schedule if the hypothesis explains why that mechanism should matter.
- Do not overfit to one benchmark slice if holdout behavior regresses.
- Do not let two candidates in the same generation test the same mutation neighborhood unless one is an explicit ablation pair.
- Do not revisit an old mutation neighborhood in a later generation unless `mutation_ledger.md` records why the revisit is justified.
- Every loop must leave behind better search state than it found.

## Core Loop
1. Map the search space in `variables.md`.
2. Map attack surfaces using `EXPERIMENTATION_STRATEGY.md`.
3. Write 10 candidate hypotheses in `hypotheses.md`.
4. Eliminate weak or redundant hypotheses.
5. Run a gap pass in `gaps.md` and add missing but important coverage.
6. Select the top 5 hypotheses in `portfolio.md`.
7. Implement one candidate per selected hypothesis.
8. Evaluate all candidates.
9. Write a postmortem in `loop_log.md`.
10. Promote new negative knowledge into `negative_knowledge.md`.
11. Spawn child hypotheses for the next loop and repeat.

## What Counts As A Good Hypothesis
A good hypothesis must contain:
- a mechanism: what changes and why it should help,
- a target: which metric or bottleneck it is expected to move,
- a risk: what could get worse,
- a test: what evidence would support or refute it,
- a scope: what class of targets it applies to,
- a disproof condition: what result would kill it quickly.

Bad hypothesis:
- "try something with shared memory"

Good hypothesis:
- "If the kernel is bandwidth-bound and repeatedly reloads reusable values, staging the hot tile in shared memory should reduce global transactions enough to improve throughput without increasing register pressure beyond occupancy limits."

## Mechanism-First Search
Default search over code changes in four classes:
- structural changes: loops, decomposition, fusion, ordering, caching, branching
- policy changes: scoring rules, dispatch rules, selection logic, update laws
- state changes: new tracked variables, counters, summaries, phase markers, estimators
- schedule changes: behavior that changes over time, iteration, size, density, confidence, or error

Examples of strong mutations:
- replacing a constant penalty with a rule that depends on runtime state,
- adding a new latent variable and an update equation for it,
- introducing a phase-aware or size-aware schedule,
- changing how work is partitioned or reused,
- swapping a static heuristic for an adaptive control law.

Examples of weak mutations:
- changing `0.2` to `0.25` with no causal story,
- retuning a threshold that humans already exposed unless this supports a larger mechanism test.

## Cold Start Checklist
Before loop 1, answer these explicitly:
- What exactly is being optimized?
- What metric decides survival?
- What metrics can hide overfitting?
- What correctness tests cannot be traded away?
- What code regions are mutable?
- What move families are forbidden or already known-bad?
- What workload slices matter most?
- What would count as a fake win?

## Portfolio Construction Rule
The top 5 should not all be minor variants of the same idea. Aim for a balanced portfolio:
- 1 conservative repair,
- 1 locality or memory move,
- 1 parallelism or scheduling move,
- 1 algorithmic or work-reduction move,
- 1 high-information gamble.

Within a generation, each portfolio slot should differ on at least two of:
- bottleneck attacked,
- operator family,
- target code region,
- benchmark slice being stressed.

Across generations, prefer frontier expansion over local churn:
- do not re-run the same neighborhood with cosmetic changes,
- do re-open a neighborhood if a new hypothesis, new evidence, or new constraint changes the expected outcome.

## Loop Exit Criteria
Stop a hypothesis family when:
- it repeatedly fails correctness,
- it consistently worsens the primary metric,
- it only wins on narrow cases and loses on holdouts,
- a stronger child hypothesis replaces it,
- the remaining upside looks dominated by another family.

## Anti-Patterns
- confusing a code change with a hypothesis,
- confusing knob search with mechanism search,
- keeping only one style of move alive because it is easy to implement,
- declaring victory from one benchmark slice,
- writing vague postmortems with no causal claim,
- turning `negative_knowledge.md` into a dump of one-off accidents,
- filling one generation with overlapping candidates that all test the same idea,
- retesting an old failed neighborhood without recording what changed,
- mutating multiple mechanisms at once and then claiming clear evidence.

## Deliverable Standard
At any point, another LLM should be able to open this folder and answer:
- what space is being searched,
- what has been tried,
- what has failed,
- what is currently worth trying next,
- why those next moves were chosen.
