# Stage 5 Hypotheses

This file follows the AlphaEvolve flow:

1. generate 20 hypotheses,
2. prune to 10,
3. add 5 gap-fillers,
4. prune 15 to 8 execution candidates,
5. add 2 baselines to form the 10-run production slate.

Stage 5 preference update:

- outside ideas are explicitly encouraged,
- separating Muon-side and AdamW-side changes is a valid and often cleaner search move,
- prefer mechanism/code changes over pure hyperparameter retuning,
- H64 lineage is one live family, not the whole search space.

## Status Legend

- `proposed`
- `shortlisted`
- `held`
- `killed`
- `selected`

## Initial 20

### H501: Faithful H64 Port
- status: shortlisted
- family: fidelity_repair
- bottleneck attacked: evaluator / implementation mismatch
- mechanism: port the true Stage 4 decoupled Nesterov-blend schedule into the current runner
- code change class: structural
- expected win: restore the real H64 effect and make H64 compounds interpretable
- main risk: implementation complexity in Muon `_step_muon`
- evidence needed: H64 regains a meaningful solo advantage over production at 20k
- disproof signal: faithful H64 remains flat or negative once correctly ported
- cheapest test: 20k faithful H64 solo run
- applies when: a prior win depends on a nontrivial kernel/control-law patch
- avoid when: the prior result was already measured with the same code path
- parent hypothesis: Stage 4 H64
- novelty vs prior loops: not a new idea, but a necessary repair
- upside: 5
- feasibility: 4
- distinctness: 5
- information_gain: 5
- transferability: 5
- notes:

### H502: H60 + H73 Long-Horizon Compound
- status: shortlisted
- family: compound_cross_path
- bottleneck attacked: independent Muon and AdamW late-phase inefficiencies
- mechanism: combine trusted H60 beta2 warmup with trusted H73 epsilon schedule
- code change class: schedule
- expected win: additive or near-additive gain because Muon and AdamW paths are distinct
- main risk: schedules help different phases and dilute each other
- evidence needed: beats both production and H60 parent at 20k
- disproof signal: fails to beat H60 parent
- cheapest test: 20k direct compound vs parent baseline
- applies when: two trusted wins touch disjoint optimizer paths
- avoid when: both changes act on the same control variable
- parent hypothesis: H60, H73
- novelty vs prior loops: first long-horizon clean compound of the two most trustworthy families
- upside: 5
- feasibility: 5
- distinctness: 4
- information_gain: 5
- transferability: 5
- notes:

### H503: Full Composite H64 + H60 + H73 + H71
- status: shortlisted
- family: composite
- bottleneck attacked: multi-path schedule interaction
- mechanism: combine the four live Stage 4 winners in one 20k run
- code change class: schedule
- expected win: best-case frontier candidate if the families are complementary
- main risk: interaction cancellation or over-scheduling
- evidence needed: beats all parents and justified by leave-one-out attribution
- disproof signal: loses to H60 parent or is only marginally above H60+H73
- cheapest test: one 20k full-stack run plus leave-one-out ablations
- applies when: multiple families remain live but their interaction is unknown
- avoid when: component mechanisms are already known to conflict
- parent hypothesis: H64, H60, H73, H71
- novelty vs prior loops: first faithful 20k full-stack evaluation
- upside: 5
- feasibility: 4
- distinctness: 4
- information_gain: 5
- transferability: 4
- notes:

### H504: Warmdown-Aware Beta2 Second Phase
- status: shortlisted
- family: muon_second_moment
- bottleneck attacked: late-phase variance adaptation
- mechanism: keep H60 early warmup, then change beta2 behavior again when LR warmdown begins
- code change class: schedule
- expected win: exploit the observed H60 phase flip more directly
- main risk: overfitting to one observed crossover
- evidence needed: earlier or larger late-phase gain than H60 parent
- disproof signal: no advantage beyond H60 parent at 20k
- cheapest test: 20k single new beta2 schedule
- applies when: optimizer benefit emerges only in late training
- avoid when: horizon is too short to see second-phase behavior
- parent hypothesis: H60
- novelty vs prior loops: new descendant based on the 20k postmortem
- upside: 5
- feasibility: 4
- distinctness: 4
- information_gain: 5
- transferability: 4
- notes:

### H505: Prolonged Beta2 Warmup to 2k
- status: killed
- family: muon_second_moment
- bottleneck attacked: early second-moment lag
- mechanism: extend H60 warmup horizon from 500 steps to 2000
- code change class: schedule
- expected win: smoother transition into stable beta2
- main risk: spends too much of training in the noisier regime
- evidence needed: stronger 10k-20k behavior than H60
- disproof signal: further worsens the already-bad early regime without stronger late payoff
- cheapest test: 20k single schedule variant
- applies when: early adaptation remains a bottleneck past 500 steps
- avoid when: the mechanism already appears phase-selective rather than duration-limited
- parent hypothesis: H60
- novelty vs prior loops: local extension only
- upside: 3
- feasibility: 5
- distinctness: 2
- information_gain: 3
- transferability: 3
- notes: killed in prune because it is too close to H504 and offers lower information gain

### H506: Loss-Reactive Beta2
- status: shortlisted
- family: stateful_muon
- bottleneck attacked: fixed second-moment response across changing regimes
- mechanism: map loss-improvement EMA to beta2 instead of using a fixed time schedule
- code change class: state
- expected win: adapt to actual training regime rather than wall-clock step
- main risk: noisy control signal destabilizes Muon
- evidence needed: smoother crossover and better late train_bpb than H60
- disproof signal: volatility increases and BPB regresses
- cheapest test: one state variable, one mapping
- applies when: regime changes are not well aligned to absolute step count
- avoid when: loss signal is too noisy to control directly
- parent hypothesis: H60
- novelty vs prior loops: first explicit stateful descendant of the H60 family
- upside: 5
- feasibility: 3
- distinctness: 5
- information_gain: 5
- transferability: 5
- notes:

### H507: Per-Shape Beta2 on Muon
- status: held
- family: muon_specialization
- bottleneck attacked: different matrix families may need different variance windows
- mechanism: assign beta2 schedules by Muon shape group
- code change class: policy
- expected win: attention and MLP matrices stop sharing one second-moment timescale
- main risk: too many degrees of freedom for one loop
- evidence needed: stable improvement over H60 on at least one dominant shape family
- disproof signal: no improvement or inconsistent gains across groups
- cheapest test: two-family split only
- applies when: grouped Muon params have systematically different gradient statistics
- avoid when: per-group statistics are too noisy
- parent hypothesis: H60
- novelty vs prior loops: new specialization surface
- upside: 4
- feasibility: 3
- distinctness: 4
- information_gain: 4
- transferability: 4
- notes:

### H508: H73 Long-Horizon Confirmation
- status: shortlisted
- family: adamw_eps
- bottleneck attacked: AdamW early-step instability and late-step adaptivity
- mechanism: run H73 over the full 20k horizon with eps scheduled across total steps
- code change class: schedule
- expected win: confirm whether H73 remains live beyond 5k
- main risk: the longer schedule weakens the early H73 effect
- evidence needed: beats production and stays competitive with H60 parent
- disproof signal: flat or negative at 20k
- cheapest test: 20k H73 solo
- applies when: a trusted 5k winner has not been validated long-run
- avoid when: the family is already demoted
- parent hypothesis: H73
- novelty vs prior loops: long-horizon validation of a live family
- upside: 4
- feasibility: 5
- distinctness: 4
- information_gain: 5
- transferability: 5
- notes:

### H509: Embedding-Only Epsilon Schedule
- status: held
- family: adamw_eps
- bottleneck attacked: overbroad AdamW scheduling
- mechanism: apply H73 only to embedding-like groups, leaving scalars/lm_head untouched
- code change class: policy
- expected win: keep the stabilization where it matters most
- main risk: removes useful help from other AdamW groups
- evidence needed: matches or beats H73 with less collateral effect
- disproof signal: weaker than global H73 everywhere
- cheapest test: per-group eps patch
- applies when: embedding groups dominate the instability story
- avoid when: late gains come from all AdamW groups
- parent hypothesis: H73
- novelty vs prior loops: first per-group AdamW eps specialization
- upside: 3
- feasibility: 4
- distinctness: 4
- information_gain: 4
- transferability: 4
- notes:

### H510: Late Beta1 Relaxation on AdamW
- status: killed
- family: adamw_momentum
- bottleneck attacked: late-stage AdamW over-smoothing
- mechanism: lower beta1 only during warmdown instead of warming it early
- code change class: schedule
- expected win: more responsive late AdamW updates
- main risk: destabilizes embeddings when LR is already low
- evidence needed: beats H73 in late phase without early damage
- disproof signal: noisy late curve and worse train_bpb
- cheapest test: one late schedule
- applies when: AdamW stale momentum is the late-phase bottleneck
- avoid when: second moment is the dominant issue
- parent hypothesis: H71
- novelty vs prior loops: directional reversal of H71
- upside: 3
- feasibility: 4
- distinctness: 3
- information_gain: 3
- transferability: 3
- notes: killed in prune because it is weaker than H508 and H513-style ablations for this loop

### H511: Faithful H64 + H73 Compound
- status: shortlisted
- family: compound_cross_path
- bottleneck attacked: Muon lookahead + AdamW denominator interaction
- mechanism: combine faithful H64 with H73
- code change class: schedule
- expected win: reproduce the old Stage 4 intuition with a distinct AdamW path helper
- main risk: H64 remains weak even when ported faithfully
- evidence needed: beats H73 solo and approaches or exceeds H60 parent
- disproof signal: underperforms both faithful H64 solo and H73 solo
- cheapest test: one 20k compound
- applies when: both component families stay live
- avoid when: H64 faithful port fails
- parent hypothesis: H64, H73
- novelty vs prior loops: first faithful cross-path H64 compound
- upside: 4
- feasibility: 4
- distinctness: 4
- information_gain: 5
- transferability: 4
- notes:

### H512: Faithful H64 + H60 Compound
- status: shortlisted
- family: compound_muon
- bottleneck attacked: Muon lookahead and Muon second-moment interaction
- mechanism: combine faithful H64 with H60
- code change class: schedule
- expected win: stronger Muon-only optimizer
- main risk: same-path schedules interfere, as seen in earlier compounding lessons
- evidence needed: beats both H60 parent and faithful H64 solo
- disproof signal: no better than H60 parent
- cheapest test: one 20k Muon-only compound
- applies when: the Muon path remains the dominant frontier
- avoid when: compounding same-path schedules is already strongly negative
- parent hypothesis: H64, H60
- novelty vs prior loops: first faithful retest of the previously confounded neighborhood
- upside: 4
- feasibility: 4
- distinctness: 3
- information_gain: 5
- transferability: 4
- notes:

### H513: Leave-One-Out of Full Composite, Removing H64
- status: shortlisted
- family: ablation
- bottleneck attacked: attribution inside the composite
- mechanism: run H60 + H73 + H71
- code change class: schedule
- expected win: show whether the full stack really needs H64
- main risk: consumes slot budget without introducing a new mechanism
- evidence needed: full composite minus H64 changes the ranking materially
- disproof signal: identical performance to the full composite
- cheapest test: one ablation run
- applies when: composite attribution matters
- avoid when: no composite is being tested
- parent hypothesis: H503
- novelty vs prior loops: ablation, not new mechanism
- upside: 3
- feasibility: 5
- distinctness: 4
- information_gain: 5
- transferability: 3
- notes:

### H514: Leave-One-Out of Full Composite, Removing H60
- status: shortlisted
- family: ablation
- bottleneck attacked: attribution inside the composite
- mechanism: run H64 + H73 + H71
- code change class: schedule
- expected win: reveal whether H60 is actually the core driver of the stack
- main risk: same as above
- evidence needed: removing H60 causes the largest drop if H60 is the real anchor
- disproof signal: composite survives unchanged without H60
- cheapest test: one ablation run
- applies when: H60 is the promoted parent
- avoid when: the parent is not in the full stack
- parent hypothesis: H503
- novelty vs prior loops: attribution ablation
- upside: 3
- feasibility: 5
- distinctness: 4
- information_gain: 5
- transferability: 3
- notes:

### H515: Leave-One-Out of Full Composite, Removing H73
- status: shortlisted
- family: ablation
- bottleneck attacked: attribution inside the composite
- mechanism: run H64 + H60 + H71
- code change class: schedule
- expected win: show whether the AdamW epsilon family is essential inside the stack
- main risk: same as above
- evidence needed: composite deteriorates materially without H73
- disproof signal: removing H73 does not matter
- cheapest test: one ablation run
- applies when: H73 is a live secondary parent
- avoid when: H73 has already been demoted
- parent hypothesis: H503
- novelty vs prior loops: attribution ablation
- upside: 3
- feasibility: 5
- distinctness: 4
- information_gain: 5
- transferability: 3
- notes:

### H516: Leave-One-Out of Full Composite, Removing H71
- status: shortlisted
- family: ablation
- bottleneck attacked: attribution inside the composite
- mechanism: run H64 + H60 + H73
- code change class: schedule
- expected win: test whether H71 helps only inside the stack or is dead weight
- main risk: same as above
- evidence needed: measurable composite drop if H71 is useful
- disproof signal: removing H71 improves or leaves the stack unchanged
- cheapest test: one ablation run
- applies when: H71 remains medium-confidence
- avoid when: H71 is already fully retired
- parent hypothesis: H503
- novelty vs prior loops: attribution ablation
- upside: 3
- feasibility: 5
- distinctness: 4
- information_gain: 5
- transferability: 3
- notes:

### H517: Shared Phase Variable Controls H60 and H73
- status: shortlisted
- family: stateful_cross_path
- bottleneck attacked: uncoordinated phase changes across Muon and AdamW
- mechanism: define one phase variable from warmdown onset or loss-improvement EMA and use it to drive both beta2 and eps schedules
- code change class: state
- expected win: cleaner coordination than independent schedules
- main risk: too much novelty for the first Stage 5 slate
- evidence needed: earlier crossover and stronger late train_bpb than H60+H73
- disproof signal: more noise and no better end state
- cheapest test: one shared phase scalar
- applies when: late gains seem tied to regime shifts rather than static step count
- avoid when: simple compounds remain unresolved
- parent hypothesis: H60, H73
- novelty vs prior loops: first explicit shared-state coordination mechanism
- upside: 5
- feasibility: 3
- distinctness: 5
- information_gain: 5
- transferability: 5
- notes:

### H518: Muon Volatility Gates AdamW Beta1
- status: held
- family: stateful_cross_path
- bottleneck attacked: stale AdamW momentum during large Muon moves
- mechanism: use Muon update/param ratio EMA to lower AdamW beta1 when Muon is volatile
- code change class: state
- expected win: adaptive cross-path coordination
- main risk: noisy signal and hidden feedback loops
- evidence needed: smoother compound curves than static H71
- disproof signal: instability or no gain over static schedules
- cheapest test: one EMA and one map
- applies when: cross-path coupling is the real bottleneck
- avoid when: same-path schedules remain unresolved
- parent hypothesis: H71, H60
- novelty vs prior loops: new feedback law
- upside: 4
- feasibility: 2
- distinctness: 5
- information_gain: 5
- transferability: 4
- notes:

### H519: Warmdown-Triggered Epsilon Floor
- status: held
- family: adamw_eps
- bottleneck attacked: overaggressive late eps shrinkage
- mechanism: keep eps from collapsing past a floor until warmdown begins, then allow sharper decay
- code change class: schedule
- expected win: retain early H73 stabilization while timing the aggressive phase better
- main risk: overengineering the schedule before long-run H73 is confirmed
- evidence needed: better than H73 solo at 20k
- disproof signal: weaker than plain H73
- cheapest test: two-phase eps schedule
- applies when: H73 stays alive but seems horizon-sensitive
- avoid when: H73 already fails long-run
- parent hypothesis: H73
- novelty vs prior loops: horizon-aware H73 descendant
- upside: 4
- feasibility: 4
- distinctness: 4
- information_gain: 4
- transferability: 4
- notes:

### H520: Attention-vs-MLP Beta2 Split
- status: held
- family: muon_specialization
- bottleneck attacked: shared second-moment policy across different matrix roles
- mechanism: split beta2 schedule by attention shapes vs MLP shapes
- code change class: policy
- expected win: better match of optimizer memory to gradient statistics
- main risk: implementation complexity and overfitting to GPT-2 shape taxonomy
- evidence needed: one family improves without the other regressing enough to erase gains
- disproof signal: no separation benefit
- cheapest test: two buckets only
- applies when: matrix families have persistent statistical differences
- avoid when: grouped shapes are too heterogeneous
- parent hypothesis: H60
- novelty vs prior loops: structured specialization of the H60 family
- upside: 4
- feasibility: 3
- distinctness: 4
- information_gain: 4
- transferability: 3
- notes:

## Initial Prune to 10

The 10 survivors after the first prune are:

1. H501 — Faithful H64 Port
2. H503 — Full Composite H64 + H60 + H73 + H71
3. H504 — Warmdown-Aware Beta2 Second Phase
4. H506 — Loss-Reactive Beta2
5. H508 — H73 Long-Horizon Confirmation
6. H517 — Shared Phase Variable Controls H60 and H73
7. H518 — Muon Volatility Gates AdamW Beta1
8. H519 — Warmdown-Triggered Epsilon Floor
9. H520 — Attention-vs-MLP Beta2 Split
10. H502 — H60 + H73 Long-Horizon Compound

Kept alive but not shortlisted for this loop:

- H507
- H509
- H511
- H512
- H513
- H514
- H515
- H516

Killed in the first prune:

- H505
- H510

Why this prune:

- preserve one fidelity-repair thread,
- keep distinct Muon, AdamW, and cross-path code-change families alive,
- avoid filling the whole shortlist with H64 ablations before the gap pass,
- delay direct LOO selection until after the explicit set-level gap audit.
