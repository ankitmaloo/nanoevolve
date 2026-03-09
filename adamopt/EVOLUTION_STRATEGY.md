# AdamOpt Evolution Strategy

This document defines the intended search strategy for AdamOpt.

The goal is to evolve optimizer ideas in stages, not to jump straight to arbitrary code mutation.

The first stage is always:

- mutate optimizer configuration and equations inside a bounded DSL
- keep the non-matrix optimizer path fixed
- keep the training code fixed
- use short, reproducible runs to rank candidates

Only after the evaluator is trustworthy do we move to full code-level changes.

## Search Stages

### Stage 0: Baseline Anchor

Goal:

- reproduce NanoChat's existing optimizer split closely enough to trust the harness

What changes:

- no mutation yet
- only the baseline spec in [`optim_search/spec.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/spec.py)

What must be true:

- parameter routing matches NanoChat's current matrix vs non-matrix split
- baseline metrics are reproducible for the same seed and batch order
- the evaluator can emit stable metrics JSON

Files involved:

- [`optim_search/spec.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/spec.py)
- [`optim_search/candidate_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/candidate_optimizer.py)
- [`optim_search/eval_candidate.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/eval_candidate.py)
- [`tests/test_candidate_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/tests/test_candidate_optimizer.py)
- [`tests/test_eval_repro.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/tests/test_eval_repro.py)

### Stage 1: DSL Evolution Only

Goal:

- search only inside the bounded optimizer DSL
- no arbitrary PyTorch code rewriting

Allowed mutations:

- momentum placement
- trust ratio on/off and clamp range
- clip policy
- decay mode
- update multiplier
- Newton-Schulz step count
- second-moment beta

These mutations are implemented in:

- [`optim_search/mutations.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/mutations.py)

The search space is defined by:

- [`optim_search/spec.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/spec.py)

The tournament logic is owned by:

- [`optim_search/tournament.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/tournament.py)
- [`optim_search/score.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/score.py)
- [`optim_search/archive.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/archive.py)

What must be true before leaving this stage:

- baseline parity is acceptable
- same seed gives same result
- obviously bad variants rank badly
- mild variants produce interpretable metric changes
- top candidates survive 2-3 seed reruns

### Stage 2: Real NanoChat Short-Run Evaluation

Goal:

- replace the toy backend with a real NanoChat-backed short-horizon evaluator

This stage still uses DSL mutation only.

The search loop remains:

- mutate spec
- instantiate candidate optimizer from spec
- run short fixed-budget training
- score and rank candidates
- promote top candidates

The difference is that the evaluator now runs real NanoChat training logic.

Expected ownership:

- [`optim_search/eval_candidate.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/eval_candidate.py) becomes the real NanoChat short-run harness
- [`scripts/search_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/scripts/search_optimizer.py) becomes the default automation entrypoint for spec search

### Stage 3: Full Training Promotion

Goal:

- take only the strongest DSL winners into longer real NanoChat training runs

At this point we still do not rewrite optimizer code.

We use longer runs to answer a different question:

- does the short-horizon winner still win when training time is large?

This stage should be bounded aggressively.

Recommended initial bounds:

- promote only top 1-3 candidates per generation
- use 2-3 seeds
- use a capped total token budget per promotion batch

### Stage 4: Code-Level Optimizer Mutation

Goal:

- only after the DSL evaluator is trusted, allow code-level optimizer changes

This means:

- patching [`nanochat/nanochat/optim.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/nanochat/nanochat/optim.py)
- patching [`nanochat/nanochat/gpt.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/nanochat/nanochat/gpt.py)

Code mutation infrastructure already exists in:

- [`optim_search/command_mutator.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/command_mutator.py)
- [`optim_search/validation.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/validation.py)
- [`optim_search/deployment.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/deployment.py)
- [`optim_search/autonomous.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/autonomous.py)

These should not be the default search path in earlier stages.

## Default Automated Flow

The default automated flow should be spec-first and fully scripted.

### Phase A: Generate Children

Input:

- parent spec
- generation index
- bounded child count

Action:

- mutate parent spec through [`optim_search/mutations.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/mutations.py)

Output:

- N child specs

Recommended initial bounds:

- children per parent: `2-4`
- total children per generation: `8-16`

Start small. The point is to get clean signal, not flood the evaluator.

### Phase B: Short-Run Evaluate

Input:

- candidate specs
- fixed short budget
- fixed seed
- fixed batch order
- fixed eval cadence

Action:

- run cheap candidate screening

Output metrics:

- final validation bpb
- validation/train AUC
- step time or tokens/sec
- NaN/Inf failures
- grad norm spikes
- update norm / param norm ratios
- memory overhead

### Phase C: Rank and Filter

Input:

- evaluated children

Action:

- kill dead candidates
- compute composite score
- keep Pareto frontier
- promote only top K

Owned by:

- [`optim_search/score.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/score.py)
- [`optim_search/tournament.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/tournament.py)

Recommended initial bounds:

- survivors per generation: `2-4`
- promoted candidates: `1-3`

### Phase D: Multi-Seed Promotion

Input:

- promoted candidates

Action:

- rerun with 2-3 seeds
- compare directly to the baseline

A candidate is a winner only if:

- mean validation bpb beats baseline by threshold
- speed regression is acceptable
- memory regression is acceptable
- no stability failures

### Phase E: Longer Real Runs

Input:

- small set of winners

Action:

- rerun on longer real NanoChat budgets

This is where full training starts to matter.

Do not do this for every child.

## Bounded Search Parameters

The search must stay bounded early on.

Recommended initial limits:

- generations: `3-5`
- children per generation: `8-16`
- short-run steps: small fixed budget
- promoted candidates per generation: `1-3`
- promotion seeds: `2-3`
- long-run winners per cycle: `1-2`

The point is to answer:

- do we have signal?
- are winners stable?
- does the evaluator separate good from bad ideas?

If that works, widen the search later.

## Why the DSL Stage Matters

The DSL-first stage is important because it gives:

- faster iteration than full code mutation
- clearer failure modes
- interpretable winners
- easier ablations
- cleaner comparisons across generations

If a DSL winner cannot survive longer real NanoChat runs, that is useful information.
It means the short-horizon evaluator must improve before code-level mutation is allowed.

## What Should Be Automated

The target end state is one command for the spec-first search loop.

That command should do all of the following:

1. create generation children from parent specs
2. run short-horizon evaluation automatically
3. save metrics and lineage
4. score and rank candidates
5. trigger multi-seed promotion
6. trigger longer real runs for the best winners
7. persist a full archive

The current building blocks already exist, but they need to be made the default path:

- [`optim_search/spec.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/spec.py)
- [`optim_search/mutations.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/mutations.py)
- [`optim_search/tournament.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/tournament.py)
- [`optim_search/archive.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/archive.py)
- [`optim_search/score.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/score.py)
- [`scripts/search_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/scripts/search_optimizer.py)

## Which Files To Run

Today:

- baseline compare:
  - [`scripts/search_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/scripts/search_optimizer.py) `compare`
- toy tournament:
  - [`scripts/search_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/scripts/search_optimizer.py) `tournament`

Intended default path:

- one spec-search command in [`scripts/search_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/scripts/search_optimizer.py) that runs:
  - generation creation
  - short-run evaluation
  - ranking
  - promotion
  - long-run escalation

Code-mutation commands should stay separate until the evaluator is solid:

- [`scripts/search_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/scripts/search_optimizer.py) `patch-code`
- [`scripts/search_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/scripts/search_optimizer.py) `validate-code`
- [`scripts/search_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/scripts/search_optimizer.py) `deploy-code`
- [`scripts/search_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/scripts/search_optimizer.py) `autonomous-run`

## Promotion Rule of Thumb

Use this default logic:

- all children start in DSL space
- only short-run winners get multi-seed reruns
- only multi-seed winners get long real runs
- only long-run winners justify code-level mutation

That keeps the expensive part focused on the smallest possible set of candidates.

## Immediate Next Step

The next implementation step should be:

- make spec-only search the default automated path
- keep code mutation infrastructure available but out of the main loop
- replace the toy evaluator with a real NanoChat short-run evaluator

That is the clean progression:

- DSL search first
- real short-run evaluation second
- long-run confirmation third
- code mutation last
