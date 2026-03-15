# Hypotheses

Write candidates here before running them. Start wide, then prune.

## Status Legend
- `proposed`
- `shortlisted`
- `active`
- `tested`
- `killed`
- `promoted`

## Scoring Rubric
Score each hypothesis from 1 to 5 on:
- upside
- feasibility
- distinctness
- information_gain
- transferability

Do not shortlist a hypothesis unless it has:
- a bottleneck it attacks,
- a cheap falsification path,
- a reason it is not redundant with an existing active hypothesis.

Prefer hypotheses that alter program behavior through a mechanism change:
- new structure,
- new rule,
- new state,
- or new schedule.

Deprioritize hypotheses that only retune constants unless they are explicit baselines or ablations.

## Hypothesis Template
### HXX: Title
- status:
- family:
- bottleneck attacked:
- mechanism:
- code change class:
- expected win:
- main risk:
- evidence needed:
- disproof signal:
- cheapest test:
- applies when:
- avoid when:
- parent hypothesis:
- novelty vs prior loops:
- upside:
- feasibility:
- distinctness:
- information_gain:
- transferability:
- notes:

## Seed Families
Use these as starting classes, not as dogma.

### H01: Improve locality by staging or reordering data access
- status: proposed
- family: memory
- bottleneck attacked: memory movement
- mechanism: reduce redundant global reads and improve locality through tiling, staging, or access reordering
- code change class: structural
- expected win: higher throughput or lower memory stall time
- main risk: register pressure or synchronization overhead
- evidence needed: bandwidth-sensitive benchmarks improve without occupancy collapse
- disproof signal: no gain on bandwidth-sensitive slices or a clear occupancy/resource cliff
- cheapest test: apply the smallest locality-oriented rewrite to the hottest reusable region
- applies when: repeated reuse exists inside a tile or neighborhood
- avoid when: working set is too large or reuse is weak
- parent hypothesis:
- novelty vs prior loops:
- upside: 4
- feasibility: 4
- distinctness: 4
- information_gain: 4
- transferability: 5
- notes:

### H02: Change work partitioning to improve parallel efficiency
- status: proposed
- family: scheduling
- bottleneck attacked: underutilization or imbalance
- mechanism: alter tile sizes, per-thread work, or warp/block mapping to improve utilization
- code change class: structural
- expected win: fewer idle lanes and better hardware occupancy
- main risk: imbalance, extra synchronization, or lower locality
- evidence needed: utilization-oriented metrics improve with stable correctness
- disproof signal: utilization-focused slices do not improve or locality regresses enough to erase gains
- cheapest test: change one mapping dimension or per-thread work unit without altering algorithmic semantics
- applies when: current mapping leaves lanes idle or causes imbalance
- avoid when: mapping is already hardware-aligned
- parent hypothesis:
- novelty vs prior loops:
- upside: 4
- feasibility: 4
- distinctness: 4
- information_gain: 4
- transferability: 5
- notes:

### H03: Reduce total work through reuse, pruning, or reformulation
- status: proposed
- family: algorithmic
- bottleneck attacked: unnecessary work
- mechanism: avoid unnecessary computation rather than making the same computation faster
- code change class: structural
- expected win: step-change speedup
- main risk: semantic drift or hidden worst-case regressions
- evidence needed: operation count or effective work drops with stable outputs
- disproof signal: work reduction is negligible or worst-case behavior regresses
- cheapest test: test one semantically narrow pruning or reuse idea with explicit correctness checks
- applies when: redundant work or dominated paths exist
- avoid when: evaluator punishes approximation or simplification
- parent hypothesis:
- novelty vs prior loops:
- upside: 5
- feasibility: 3
- distinctness: 5
- information_gain: 5
- transferability: 4
- notes:

### H04: Regularize control flow to reduce divergence
- status: proposed
- family: control_flow
- bottleneck attacked: divergence
- mechanism: restructure branches, masks, or path specialization to reduce divergent execution
- code change class: structural
- expected win: smoother warp execution and more predictable latency
- main risk: extra arithmetic or code complexity
- evidence needed: branch-heavy cases improve more than branch-light cases
- disproof signal: branch-heavy slices do not improve or extra arithmetic cancels the gain
- cheapest test: replace one hot divergent branch pattern with a more regular equivalent
- applies when: divergent branches are on hot paths
- avoid when: branches are already rare or highly predictable
- parent hypothesis:
- novelty vs prior loops:
- upside: 3
- feasibility: 4
- distinctness: 3
- information_gain: 3
- transferability: 4
- notes:

### H05: Change data layout or packing
- status: proposed
- family: layout
- bottleneck attacked: memory format inefficiency
- mechanism: alter representation to better match vectorization, alignment, or cache lines
- code change class: structural
- expected win: more efficient loads, stores, and compute packing
- main risk: conversion overhead or interface breakage
- evidence needed: end-to-end gains exceed transformation cost
- disproof signal: layout conversion overhead dominates or interface constraints block adoption
- cheapest test: change layout only inside the hottest internal region while preserving the external interface
- applies when: memory format is a bottleneck
- avoid when: layout changes are blocked by external interfaces
- parent hypothesis:
- novelty vs prior loops:
- upside: 4
- feasibility: 2
- distinctness: 4
- information_gain: 4
- transferability: 4
- notes:

### H06: Change precision or accumulation strategy safely
- status: proposed
- family: numerical
- bottleneck attacked: arithmetic cost
- mechanism: reduce precision selectively or use mixed accumulation to lower cost
- code change class: policy
- expected win: better throughput with acceptable error
- main risk: silent quality regression
- evidence needed: accuracy stays inside tolerance while speed improves
- disproof signal: tolerance breaks or quality-sensitive holdouts regress
- cheapest test: lower precision in one isolated hot path while preserving reference accumulation where needed
- applies when: math tolerance exists
- avoid when: evaluator is extremely sensitive to numeric drift
- parent hypothesis:
- novelty vs prior loops:
- upside: 4
- feasibility: 3
- distinctness: 4
- information_gain: 4
- transferability: 4
- notes:

### H07: Fuse or defuse stages to move the bottleneck
- status: proposed
- family: pipeline
- bottleneck attacked: stage overhead or poor pipeline balance
- mechanism: fuse stages to save traffic or split stages to improve scheduling and locality
- code change class: structural
- expected win: reduced overhead or better pipeline balance
- main risk: bigger kernels, spills, or lost reuse
- evidence needed: bottleneck shifts in the intended direction
- disproof signal: stage changes increase spills, traffic, or synchronization enough to offset any benefit
- cheapest test: fuse or split exactly one boundary with the clearest overhead signal
- applies when: stage boundaries are artificial or costly
- avoid when: current decomposition is already matched to hardware limits
- parent hypothesis:
- novelty vs prior loops:
- upside: 4
- feasibility: 3
- distinctness: 4
- information_gain: 4
- transferability: 4
- notes:

### H08: Trade occupancy for more useful per-thread state
- status: proposed
- family: occupancy_tradeoff
- bottleneck attacked: redundant traffic or weak per-thread reuse
- mechanism: spend registers or shared memory to reduce memory traffic or redundant work
- code change class: state
- expected win: net throughput gain despite lower occupancy
- main risk: occupancy cliff
- evidence needed: slower occupancy but faster wall-clock
- disproof signal: occupancy falls with no compensating throughput gain
- cheapest test: add a small amount of state to eliminate one confirmed hot reload or recomputation path
- applies when: kernel is not occupancy-limited in the first place
- avoid when: current occupancy is already marginal
- parent hypothesis:
- novelty vs prior loops:
- upside: 3
- feasibility: 3
- distinctness: 4
- information_gain: 5
- transferability: 4
- notes:

### H09: Add specialization or dispatch for common regimes
- status: proposed
- family: specialization
- bottleneck attacked: common-case inefficiency
- mechanism: create fast paths for common shapes, sizes, or value regimes
- code change class: policy
- expected win: outsized gains on dominant workload slices
- main risk: code bloat or overfitting
- evidence needed: representative distribution improves, not just one handpicked case
- disproof signal: only one slice wins while holdouts or tail cases lose
- cheapest test: specialize exactly one dominant regime with a guarded fallback
- applies when: workload has repeated common regimes
- avoid when: input distribution is highly diffuse
- parent hypothesis:
- novelty vs prior loops:
- upside: 3
- feasibility: 4
- distinctness: 4
- information_gain: 3
- transferability: 3
- notes:

### H10: Use compiler-facing simplifications or structure hints
- status: proposed
- family: compiler
- bottleneck attacked: weak code generation
- mechanism: rewrite expressions or structure loops so the compiler can optimize better
- code change class: structural
- expected win: free performance from cleaner generated code
- main risk: low upside or non-portable gains
- evidence needed: generated behavior improves across more than one test case
- disproof signal: generated code or runtime behavior is unchanged in any meaningful way
- cheapest test: perform one structural rewrite that should simplify codegen without altering semantics
- applies when: codegen quality is clearly part of the problem
- avoid when: bottleneck is algorithmic
- parent hypothesis:
- novelty vs prior loops:
- upside: 2
- feasibility: 5
- distinctness: 3
- information_gain: 3
- transferability: 5
- notes:

### H11: Replace a static knob with an adaptive rule
- status: proposed
- family: adaptive_policy
- bottleneck attacked: regime mismatch from one-size-fits-all behavior
- mechanism: replace a fixed coefficient, threshold, or priority with a rule driven by runtime state or input regime
- code change class: policy
- expected win: better behavior across multiple slices without manual retuning
- main risk: added instability or hidden branching costs
- evidence needed: slices with different regimes improve relative to a fixed-rule baseline
- disproof signal: adaptive behavior tracks noise rather than useful state, or only one slice improves
- cheapest test: convert one influential constant into a simple function of one measured signal
- applies when: the workload is non-stationary or regime-dependent
- avoid when: the fixed rule is already robust and stable
- parent hypothesis:
- novelty vs prior loops:
- upside: 5
- feasibility: 4
- distinctness: 5
- information_gain: 5
- transferability: 5
- notes:

### H12: Introduce a new latent state variable
- status: proposed
- family: latent_state
- bottleneck attacked: missing information in current decision making
- mechanism: track a new internal summary, confidence, density, backlog, or phase variable that existing logic ignores
- code change class: state
- expected win: better decisions because the program reacts to hidden regime information
- main risk: noisy or misleading state causes instability
- evidence needed: the new state separates regimes that previously needed contradictory behavior
- disproof signal: the new state does not correlate with better decisions or adds overhead without benefit
- cheapest test: add one lightweight tracked variable and use it in one hot decision point
- applies when: current logic treats distinct regimes as if they were the same
- avoid when: no useful hidden regime seems measurable at low cost
- parent hypothesis:
- novelty vs prior loops:
- upside: 5
- feasibility: 3
- distinctness: 5
- information_gain: 5
- transferability: 5
- notes:

### H13: Invent a schedule or annealing pattern over an ignored variable
- status: proposed
- family: schedule
- bottleneck attacked: static behavior across changing phases
- mechanism: make a decision rule vary over iteration, depth, size, density, confidence, or error instead of staying fixed
- code change class: schedule
- expected win: better phase-specific behavior and fewer tradeoffs between early and late regimes
- main risk: overengineering or brittle schedules
- evidence needed: early and late regime metrics move in the intended directions without correctness regressions
- disproof signal: schedule complexity adds no benefit beyond a simpler static rule
- cheapest test: add a simple monotonic or piecewise schedule tied to one justified signal
- applies when: the task has clear phases or changing conditions
- avoid when: regime conditions are effectively stationary
- parent hypothesis:
- novelty vs prior loops:
- upside: 5
- feasibility: 3
- distinctness: 5
- information_gain: 5
- transferability: 5
- notes:

## Elimination Rule
Kill a hypothesis quickly if:
- its cheapest test fails cleanly,
- it produces no useful evidence,
- it is dominated by a more specific child hypothesis,
- it duplicates an already active portfolio slot,
- it depends on assumptions invalidated by the evaluator or holdouts.
