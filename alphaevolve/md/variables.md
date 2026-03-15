# Variables

Fill this first. This file defines the search space.

## Task Snapshot
- Target:
- Primary metric:
- Secondary metrics:
- Baseline score:
- Baseline holdout score:
- Hardware:
- Framework / language:
- Evaluation command:
- Evaluator noise / variance:
- Dominant workload shape:
- Hard constraints:
- Soft preferences:

## Immutable Constraints
- Semantic invariants that must never break:
- Numerical tolerances:
- API / ABI constraints:
- Memory limits:
- Latency limits:
- Compile-time limits:
- Output-format or evaluator-format constraints:

## Controllable Variables

Prefer variables that require code changes rather than just retuning exposed knobs.
If a human-facing hyperparameter exists, ask first whether it should become:
- a derived quantity,
- an adaptive rule,
- a schedule,
- or a function of newly tracked state.

### Algorithmic Structure
- decomposition strategy
- recurrence / iteration structure
- work partitioning
- fusion vs separation
- exact vs approximate computation

### Memory / Data Movement
- shared memory usage
- cache friendliness
- coalescing / locality
- prefetching
- recomputation vs storage
- data layout

### Parallelism / Scheduling
- block / tile shape
- warp mapping
- vector width
- unrolling
- synchronization placement
- occupancy vs register tradeoff

### Control Flow
- branching structure
- specialization paths
- masking strategy
- early exit conditions
- invariant hoisting

### Numerical Strategy
- precision choices
- accumulator type
- normalization / rescaling
- fast math opportunities

### Policy / Decision Rules
- scoring rules
- priority rules
- dispatch rules
- threshold logic
- tie-break logic
- update laws

### State / Latent Variables
- new counters or summaries
- rolling estimates
- error or confidence signals
- phase markers
- density / sparsity estimates
- local workload descriptors

### Schedules / Annealing / Adaptation
- iteration-dependent behavior
- size-dependent behavior
- density-dependent behavior
- confidence-dependent behavior
- error-dependent behavior
- phase transitions

## Known Bottlenecks
- suspected bottleneck:
  evidence:
  confidence:

## Unknowns That Need Evidence
- unknown:
  why it matters:
  cheapest way to resolve it:

## Ignored Variables Worth Questioning
- variable humans usually ignore:
  why it might matter:
  possible mechanism:
  cheapest code-level test:

## What Is Allowed To Change
- files:
- regions / evolve blocks:
- launch config:
- build flags:
- evaluator knobs:

## What Must Not Change
- externally visible behavior:
- interfaces:
- benchmarking protocol:
- reference outputs:

## What Must Be Measured
- correctness signals:
- primary win condition:
- regressions to watch:
- holdout scenarios:
- variance across repeated runs:
- compile time / resource usage if relevant:

## Benchmark Slices
List the slices that matter so the search does not optimize only the average.

| Slice | Why it matters | Current baseline | Notes |
| --- | --- | --- | --- |
| Typical case | | | |
| Small inputs | | | |
| Large inputs | | | |
| Pathological case | | | |
| Holdout / unseen case | | | |

## Space Coverage Check
Mark each area as `untouched`, `lightly explored`, or `well explored`.

| Area | Coverage | Notes |
| --- | --- | --- |
| Algorithmic structure | untouched | |
| Memory / data movement | untouched | |
| Parallelism / scheduling | untouched | |
| Control flow | untouched | |
| Numerical strategy | untouched | |
| Specialization / dispatch | untouched | |
| Build / compiler tactics | untouched | |

## Attack Surface Matrix
Summarize the main surfaces before narrowing.

| Surface ID | Region | Bottleneck | Change Class | Mechanism | Observability | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S01 | | | | | | unexplored | |

## Evidence Notes
- profiler observations:
- generated-code observations:
- evaluator caveats:
- suspected fake-win patterns:
- places where a static knob may need to become an adaptive mechanism:
