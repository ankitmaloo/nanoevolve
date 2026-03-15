# Experimentation Strategy

This file explains how to discover attack surfaces and cover a wide search area before narrowing down.

Evolution alone is not enough.
You also need a deliberate experimentation plan so the search does not get trapped in one easy region of the space.

## Goal
For any target problem:
- identify where meaningful code changes can attack performance or quality,
- probe multiple distinct surfaces early,
- learn which surfaces have leverage,
- then narrow only after broad evidence exists.

## Definition: Attack Surface
An attack surface is a place where a code change could plausibly move the objective.

An attack surface is defined by:
- code region,
- bottleneck or failure mode,
- change class,
- observable metric or benchmark slice,
- expected mechanism.

Examples:
- a hot memory path with poor reuse,
- a static scoring rule that should depend on runtime state,
- a missing latent variable in a decision point,
- a phase-blind heuristic that should anneal over time,
- a stage boundary with avoidable traffic,
- a control-flow hotspot with divergence.

## Main Principle
Cover surfaces first, optimize winners second.

Do not start by asking:
- "What is the best mutation?"

Start by asking:
- "What are the major surfaces where a code change could matter?"

## Surface Classes
Every target should be scanned for these surface classes:

1. Structural surfaces
- loop structure
- decomposition
- fusion / defusion
- dataflow ordering
- reuse placement

2. Memory surfaces
- locality
- staging
- layout
- cache behavior
- recomputation vs storage

3. Parallelism surfaces
- mapping
- tiling
- vectorization
- occupancy tradeoffs
- synchronization structure

4. Policy surfaces
- scoring rules
- thresholds
- tie-breaks
- dispatch logic
- update rules

5. State surfaces
- missing summaries
- missing counters
- missing confidence or density signals
- missing phase markers

6. Schedule surfaces
- phase-aware behavior
- size-aware behavior
- depth-aware behavior
- error-aware behavior
- annealing or cooling patterns

7. Robustness surfaces
- narrow benchmark wins
- holdout regressions
- correctness drift
- variance instability

## Surface Mapping Procedure
Use this exact procedure for a new problem.

### Step 1: Slice The Program
Break the target into meaningful mutable regions.

For each region, ask:
- what does this region decide,
- what bottleneck could live here,
- what inputs or regimes pass through here,
- what metrics could this region influence?

### Step 2: Cross With Change Classes
For each region, test whether each change class is plausible:
- structural,
- policy,
- state,
- schedule.

Do not assume one class fits all regions.

### Step 3: Score Each Surface
For each surface, score from 1 to 5:
- leverage: how much upside if correct?
- plausibility: how believable is the mechanism?
- observability: can the evaluator detect the effect?
- implementation_cost: how hard is the first probe?
- overlap_risk: how likely is duplication with another surface?

### Step 4: Select Surface Scouts
Choose 5 to 8 surfaces for first-pass probing.

The first-pass set should:
- span multiple code regions,
- span multiple change classes,
- include at least one adaptive-rule or latent-state surface,
- include at least one schedule surface if the problem has phases,
- include at least one high-risk, high-information surface.

## Experiment Types
Use three experiment types.

### 1. Scout Experiments
Purpose:
- test whether a surface has leverage at all.

Properties:
- minimal change,
- one mechanism,
- cheap to falsify,
- broad coverage value.

Use scouts early.

### 2. Exploit Experiments
Purpose:
- deepen a surface that already showed real promise.

Properties:
- more focused,
- more locally optimized,
- narrower search near a winning mechanism.

Use exploits only after a scout produced evidence.

### 3. Ablation Experiments
Purpose:
- prove what actually caused a win.

Properties:
- intentional overlap allowed,
- remove one mechanism or isolate one signal,
- clarify causality.

Use ablations sparingly.

## Coverage Budget
Within one generation, allocate the portfolio by role.

Default 5-slot budget:
- 2 scout slots for broad new surfaces,
- 2 exploit slots for the strongest live surfaces,
- 1 wildcard slot for a risky or ignored surface.

If there is no strong winner yet:
- use 3 scouts,
- 1 exploit,
- 1 wildcard.

If the search is already converging:
- use 1 scout,
- 3 exploits,
- 1 ablation or wildcard.

## Surface Coverage Rules
Do not allow all active slots to cluster on:
- one code region,
- one bottleneck,
- one change class,
- one benchmark slice.

At least one slot per generation should try one of:
- a new code region,
- a new change class,
- a new benchmark slice,
- a new latent-variable or schedule idea.

## Exploration Triggers
Force exploration if any of these happen:
- best score plateaus,
- holdouts lag behind train slices,
- three loops focus on the same family,
- no new surfaces have been opened recently,
- results are dominated by weak constant retuning.

## Exploitation Triggers
Allow narrowing when:
- a surface has repeated wins,
- holdouts also improve,
- the causal mechanism is partly understood,
- overlapping nearby probes are now worth the cost.

## Surface Retirement
Retire a surface when:
- repeated scouts fail,
- observability is too weak,
- the mechanism lacks a believable causal story,
- it is dominated by a stronger neighboring surface,
- it repeatedly yields fake wins.

## Surface Reopening
Reopen a retired surface only if:
- the parent code changed materially,
- a new state variable makes the surface newly plausible,
- a new benchmark slice reveals a previously hidden regime,
- a prior negative result was confounded by a now-fixed assumption.

## Minimum Standard For Broad Coverage
Before saying the search has good coverage, confirm that you have attempted:
- one structural rewrite,
- one policy rewrite,
- one state invention,
- one schedule invention if phases plausibly exist,
- one memory or locality attack,
- one algorithmic work-reduction attack,
- one robustness-oriented probe on holdouts.

If not, coverage is incomplete.

## Surface Matrix Template
Use this table in planning.

| Surface ID | Region | Bottleneck | Change Class | Mechanism | Leverage | Observability | Cost | Overlap Risk | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S01 | | | | | | | | | unexplored | |

## Practical Rule For Weak Implementers
If you do not know what to do:
1. map 8 surfaces,
2. choose 5 with the best combined coverage,
3. make 3 of them scouts,
4. keep only 2 for local exploitation,
5. write down what each slot is teaching you.

This is safer than prematurely optimizing one favorite idea.
