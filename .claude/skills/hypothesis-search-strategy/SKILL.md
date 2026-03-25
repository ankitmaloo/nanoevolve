---
name: hypothesis-search-strategy
description: Design staged hypothesis searches for optimization, training, benchmarking, and evolutionary experiments. Use when Codex needs to generate distinct hypotheses, narrow a search space, define controls, build early screens, separate exploration from exploitation, or turn vague experiment ideas into a concrete sequence with kill rules and promotion logic.
---

# Hypothesis Search Strategy

Treat hypothesis design as search-space engineering, not idea listing.

Start by decomposing the final objective into latent components. Write the score as:

- training quality
- packaging or export penalty
- evaluation lift
- systems throughput or step budget
- artifact or code-size constraints

Do not screen all components with one method unless they share the same observability. A short training run can measure early learning dynamics. It usually cannot cleanly measure eval-only or export-only effects.

Knowledge work here means sampling from the solution distribution on purpose.

Do not default to the first plausible cluster of ideas. Instead:

- map the full editable or searchable surface
- sample distinct mechanisms from across that surface
- measure which regions are covered
- identify set-level gaps
- course-correct the pack before spending expensive runs

The goal is not just to rank hypotheses. The goal is to maintain coverage over the solution space while narrowing it.

## Workflow

### 1. State the target and constraints

Write the target in one line:

- optimize what metric
- at what horizon
- under what hardware, wallclock, size, or policy constraints

Then write the three most important budgets:

- compute budget
- artifact budget
- decision budget

Decision budget means how many expensive runs are actually affordable.

### 2. Factor the score

Break the objective into a small set of causal lanes. Use only lanes that matter for the task.

Common lanes:

- training dynamics
- architecture or capacity allocation
- optimizer law or schedule
- export or quantization
- evaluation policy
- systems throughput
- data or curriculum

For each lane, answer:

- what mechanism changes the score
- what cheap signal is predictive
- what expensive run is needed before promotion

If a lane does not have a cheap predictive signal, admit that and treat it as later-stage.

### 3. Map the attack surface

List the actual places where the system can change.

Examples:

- model graph
- optimizer law
- schedule
- export format
- evaluation policy
- data ordering
- tokenizer
- systems throughput

For each surface, write:

- editable region
- causal role
- likely effect on the score
- whether it is cheap or expensive to test

Do not let the hypothesis list get ahead of the surface map. If the surface map is weak, the hypothesis set will be narrow by accident.

### 4. Generate mechanism-level families

Generate hypotheses as distinct causal mechanisms, not knob variants.

A good family changes one story:

- more updates in fixed wallclock
- better per-step optimizer geometry
- smaller post-export degradation
- more eval context without retraining

A bad family is just a nearby retune of the same story:

- width `480` vs `512`
- warmdown `2500` vs `3000`
- stride `128` vs `256`

Use retunes only after a mechanism survives.

### 5. Build the coverage matrix

Create a simple coverage table before picking the pack.

Each row should be one family.
Each column should be one important surface or lane.

Mark each family as:

- `primary` when it directly explores that region
- `secondary` when it touches it indirectly
- `none` when it does not cover it

Then inspect the set, not the rows.

Ask:

- which important surfaces are uncovered
- which surfaces are only weakly covered
- which surfaces are overrepresented
- which families are redundant because they probe the same mechanism

This is the set-level gap check.

### 6. Find set-level gaps and course-correct

After the coverage matrix, write a gap list.

Typical gap types:

- missing lane
- no direct attack on a first-order bottleneck
- too many variants from one neighborhood
- no control for a child branch
- no hypothesis that could falsify the dominant story
- no “wildcard” family from a distant region of the space

Then course-correct before running:

- add one family if a critical surface is uncovered
- remove one family if it is redundant
- split one family if it mixes multiple causal stories
- defer one family if the chosen horizon cannot observe it well

Do not wait for bad results to discover a bad pack design.

### 7. Classify each family by observability

Put every family into one lane:

- `train-screen`
- `shared-checkpoint export`
- `shared-checkpoint eval`
- `full aligned final`

Do not mix lanes inside the same early pack unless there is a strong reason.

Rules:

- training mutations need training screens
- export mutations should be compared on the same checkpoint first
- eval mutations should be compared on the same checkpoint first
- architecture or large interaction claims usually need later confirmation

### 8. Build matched controls

Every candidate needs the right control.

Use:

- one primary control for the lane
- one repeated control if short-run noise may matter
- one child control when some candidates inherit from an intermediate branch

Do not compare every candidate to the whole pack indiscriminately.

Examples:

- optimizer branch vs optimizer control
- record-derivative branch vs record-like control
- fp16-export child vs fp16-export control, not raw baseline

### 9. Design the screen pack

Use the budget to maximize information, not symmetry.

For an `8` GPU screen, prefer:

- `2` controls
- `6` candidates

Only use more candidates if the lane is stable enough that control repeat is unnecessary.

Each screen candidate must answer one question:

- does it learn faster per wallclock
- does it reduce quantization gap
- does it improve eval lift
- does it save enough step time to buy more updates

If a candidate cannot answer a clear question in the chosen horizon, remove it from that pack.

### 10. Define kill rules before running

Write kill rules up front.

For training screens, kill if:

- slower and not better
- same speed and clearly worse
- better pre-quant but worse post-quant when post-quant is observable
- unstable against repeated control noise

For export or eval bakeoffs, kill if:

- bigger and not better
- better only by a negligible amount relative to noise
- operationally too slow

Do not rescue a weak result with storytelling after the fact.

### 11. Define promotion rules before running

Promote only survivors with a causal read.

Promotion ladder:

1. sanity
2. screen
3. semifinal
4. aligned final

Default policy:

- kill aggressively at sanity
- promote only the best `1` by default from screen
- promote `2` only when margins are close or interactions are important

Use full aligned runs only for hypotheses that already survived a cheaper predictive test.

### 12. Separate exploration from exploitation

In exploration, maximize mechanism diversity.

In exploitation, narrow to:

- local retunes of winners
- clean composites of validated winners
- one challenge branch that tests whether a previously weak family only needed support

Do not spend early exploration slots on tightly related variants.

## Failure Modes

Watch for these mistakes:

- using one cheap screen for training, export, and eval ideas together
- treating knob changes as distinct families
- no repeated control in a noisy short horizon
- comparing a child branch to the wrong parent
- ranking by a metric that the screen cannot observe reliably
- concluding “quant win” when the actual gain came from more steps or faster runtime
- building composites before validating single-factor moves
- no explicit coverage matrix
- no set-level gap pass before launching
- confusing “many ideas” with “broad solution-space sampling”
- staying inside one local mode because the first few ideas looked reasonable

## Output Format

When using this skill, produce the experiment plan in this order:

1. objective and hard constraints
2. score decomposition into causal lanes
3. attack-surface map
4. mechanism families
5. coverage matrix
6. set-level gaps and course corrections
7. hypothesis table with:
   - family
   - lane
   - matched control
   - cheap signal
   - kill rule
   - promotion trigger
8. screen pack layout
9. run sequence
10. interpretation rules

Keep the hypothesis table mechanism-level. Put local retunes into a later section called `follow-on tuning`, not into the initial family set.

## Default Heuristics

Use these defaults unless the task clearly argues otherwise:

- start from the strongest credible baseline, not the cleanest old one, if the goal is frontier-chasing
- use the cleanest controlled baseline only for attribution
- use repeated controls in short noisy screens
- prefer one real winner over many ambiguous survivors
- separate train-side screens from export/eval bakeoffs
- ask what signal would falsify the idea before asking how to run it
- sample from distant parts of the solution space before doing local exploitation
- include at least one family whose main job is coverage, not confidence
- treat uncovered first-order surfaces as planning failures, not just future work

If the user asks for “better hypotheses,” first rewrite the ontology of the search space. Only then propose new experiments.
