# Optimizer Lab Checkpoint

This file captures the current project state for the composite repo rooted at:

- [`optimizer_lab`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab)

Current subdirectories:

- [`adamopt`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt)
- [`nanochat`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/nanochat)
- [`alphaevolve`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/alphaevolve)

## Project Shape

The intended project model is:

- `adamopt/` = optimizer-search control plane
- `nanochat/` = real training substrate
- `alphaevolve/` = prior evolutionary code / reference material

This is documented in:

- [`README.md`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/README.md)
- [`workspace.toml`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/workspace.toml)

## Core Strategic Decision

The project is intentionally staged.

The current intended progression is:

1. bounded DSL optimizer evolution
2. real NanoChat short-run evaluation
3. long-run promotion for a very small number of winners
4. only then code-level optimizer mutation

The strategy and win criteria are documented in:

- [`adamopt/EVOLUTION_STRATEGY.md`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/EVOLUTION_STRATEGY.md)
- [`adamopt/WIN_HIERARCHY.md`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/WIN_HIERARCHY.md)

## What Is Implemented

### 1. Bounded Optimizer DSL

Implemented in:

- [`adamopt/optim_search/spec.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/spec.py)

This now supports:

- baseline Muon-like matrix optimizer config
- trust ratio config
- clipping config
- decay config
- second-moment config
- update multiplier
- bounded stateful control

The stateful DSL extension now includes training-state-conditioned behavior through:

- `loss_ema`
- `loss_improvement_ema`
- `grad_norm_ema`
- `update_ratio_ema`
- `grad_alignment_ema`
- `step_fraction`

These are combined through a smooth gate and bounded actuator ranges rather than arbitrary code generation.

### 2. Spec-Driven Candidate Optimizer Runtime

Implemented in:

- [`adamopt/optim_search/candidate_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/candidate_optimizer.py)

What it does:

- preserves the NanoChat-style split between matrix params and non-matrix params
- applies the bounded DSL to matrix updates only
- keeps non-matrix params on AdamW-like behavior
- tracks update statistics
- now tracks stateful training proxies and applies gated aggressive vs conservative behavior

### 3. Mutation System

Implemented in:

- [`adamopt/optim_search/mutations.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/mutations.py)

Current mutation coverage includes:

- momentum placement
- trust ratio on/off and clamp range
- clip policy
- decay mode
- update multiplier
- Newton-Schulz step count
- second-moment beta
- stateful control on/off
- gate bias
- gate sensor weights
- stateful EMA beta
- actuator range adjustments

### 4. Evaluation Harness

Implemented in:

- [`adamopt/optim_search/eval_candidate.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/eval_candidate.py)

Current status:

- uses a deterministic toy backend
- supports fixed seed and fixed batch order
- reports structured metrics
- feeds loss/step context into the stateful optimizer DSL

Returned metrics include:

- final validation bpb
- best validation bpb
- train/validation AUC
- step time
- tokens/sec
- NaN/Inf failures
- grad norm spikes
- update/parameter norm ratios
- memory overhead
- stability penalty

### 5. Scoring And Win Hierarchy

Implemented in:

- [`adamopt/optim_search/score.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/score.py)
- [`adamopt/optim_search/tournament.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/tournament.py)

Current scoring logic now explicitly models:

- sample efficiency
- wall-clock / time-to-target
- stability
- throughput-adjusted efficiency
- seed robustness

The win hierarchy is persisted into candidate records and promotion records through structured `win_assessment` payloads.

### 6. Tournament / Search Archive

Implemented in:

- [`adamopt/optim_search/tournament.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/tournament.py)
- [`adamopt/optim_search/archive.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/archive.py)
- [`adamopt/scripts/search_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/scripts/search_optimizer.py)

Current behavior:

- baseline candidate creation
- generation mutation
- short-run evaluation
- Pareto filtering
- multi-seed promotion reruns
- archive persistence
- summary generation

### 7. Code Mutation / Validation / Deployment Infrastructure

Implemented but not intended as the default early-stage search path:

- [`adamopt/optim_search/command_mutator.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/command_mutator.py)
- [`adamopt/optim_search/validation.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/validation.py)
- [`adamopt/optim_search/deployment.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/deployment.py)
- [`adamopt/optim_search/autonomous.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/autonomous.py)

What exists:

- patch tracking for real NanoChat workspaces
- local validation against real NanoChat
- detached remote deployment and trace capture
- autonomous async patch/validate/deploy/poll loop

Important:

- this infrastructure exists
- it is not the intended default search path until the evaluator is strong enough

### 8. Real NanoChat Integration

Current NanoChat integration points are:

- [`nanochat/nanochat/gpt.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/nanochat/nanochat/gpt.py)
- [`nanochat/nanochat/optim.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/nanochat/nanochat/optim.py)

The identified low-touch patch points are:

- `setup_optimizer`
- `adamw_step_fused`
- `muon_step_fused`

Tests for patching and validation now use the real local NanoChat clone, not a fake NanoChat package.

## What Is Working

### Test Status

Latest verified test result:

- `../.venv/bin/python -m pytest adamopt/tests -q`
- result: `18 passed`

This was run from:

- [`optimizer_lab`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab)

### Verified Properties

Currently verified:

- spec mutations work
- stateful DSL spec round-trip works
- candidate optimizer instantiation works
- stateful annealing variant runs
- same-seed deterministic learning behavior works on the toy evaluator
- win hierarchy scoring works
- real NanoChat patching works
- real NanoChat validation works
- autonomous patch/validate/deploy/poll loop works

## Where We Are

The project is at this checkpoint:

- the bounded DSL is now expressive enough to represent phase-aware optimizer behavior
- the runtime can execute that behavior
- the scorer understands the win hierarchy
- the control plane for code mutation and remote execution exists

But:

- the main evaluator is still the toy backend
- the default search loop is not yet running real NanoChat short-horizon evaluation
- the scaling and tuning-robustness axes are modeled in policy but not yet fully measured in evaluation

So the project is beyond scaffold-only status, but not yet at real end-to-end NanoChat optimizer discovery.

## What Is Pending

### Highest Priority

1. Replace the toy evaluator with a real NanoChat short-run evaluator
   Files likely affected:
   - [`adamopt/optim_search/eval_candidate.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/optim_search/eval_candidate.py)
   - [`adamopt/scripts/search_optimizer.py`](/Users/ankit/Documents/dev/RL/paperbench/optimizer_lab/adamopt/scripts/search_optimizer.py)

2. Make spec-only search the default automated path
   Meaning:
   - DSL mutation first
   - real NanoChat short-run evaluation second
   - long-run promotion third
   - code-level mutation only after that

3. Add longer-horizon evaluation strong enough to reward annealing-like behavior
   Reason:
   - very short horizons will over-favor aggressive early movers
   - stateful phase-aware behavior matters most later in training

### Next Evaluation Gaps

Still missing:

- real time-to-target measurement on NanoChat
- real large-scale promotion runs
- real scaling checks
- real tuning-sensitivity / robustness checks

### Search-Policy Gaps

Still needed:

- explicit child-count / generation-budget automation beyond current toy tournament defaults
- tighter control over how many children are spawned per stage and promoted per stage in the real evaluator path
- a clean real-run scheduler for remote providers

### Remote Execution Gaps

The project has detached remote execution, but the provider abstraction for ephemeral managed backends is still pending.

That means:

- remote worker leasing / replacement is not yet implemented as a provider-agnostic scheduler
- infra-failure recovery and mutator-based repair loops are still design intent, not completed implementation

## Current Constraints / Caveats

1. The best current search loop still uses the toy backend.
2. The stateful DSL is implemented, but it has not yet been selected by real NanoChat training dynamics.
3. The code-mutation path exists, but it should remain secondary until the spec evaluator is trusted.
4. The current virtualenv is still one level above the composite repo root:
   - `/Users/ankit/Documents/dev/RL/paperbench/.venv`

## Recommended Next Step

The next correct step is:

- build the real NanoChat short-run evaluator for spec-only candidates

That is the most important missing piece because it unlocks:

- real selection pressure for the bounded DSL
- real measurement of annealing-like behavior
- trustworthy promotion into longer training runs

## Summary

At this checkpoint, the project has:

- a composite repo layout
- strategy and win-criteria documentation
- a bounded DSL for optimizer evolution
- a stateful annealing-capable DSL extension
- a spec-driven runtime
- mutation logic
- scoring and winner logic
- archive and tournament machinery
- real NanoChat patch/validation infrastructure
- autonomous deployment machinery

The main thing still missing is:

- replacing the toy evaluator with a real NanoChat short-run evaluator so the system can discover real optimizer winners instead of only toy-harness winners
