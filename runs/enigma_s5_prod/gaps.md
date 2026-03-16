# Stage 5 Gaps

This is the post-prune gap pass. We started from 10 shortlisted hypotheses and now add 5 gap-fillers to bring the live consideration set to 15 before the final prune to 8 execution candidates.

## Five Set-Level Gaps

### G01: No explicit secondary-parent carry of H73
- uncovered area: long-horizon validation of the strongest AdamW family
- why it matters: H73 is trusted at 5k but not yet at 20k
- current portfolio miss: without H73 solo, compounds cannot be interpreted cleanly
- candidate hypothesis to add: H522 carry H73 as a Stage 5 parent/comparator
- expected information gain: very high
- what evidence is missing: whether H73 remains frontier-relevant long-run
- urgency: critical

### G02: No full leave-one-out attribution on the composite
- uncovered area: interaction attribution
- why it matters: a full stack score without LOOs does not tell us which pieces are carrying it
- current portfolio miss: the shortlist kept the composite but not the execution-grade ablation set
- candidate hypothesis to add: H523 restore LOO removing H64
- expected information gain: high
- what evidence is missing: whether the full stack needs H64 at all once H64 is implemented faithfully
- urgency: high

### G03: No LOO removing H60 from the composite
- uncovered area: attribution against the promoted parent
- why it matters: if removing H60 does not hurt the composite, the whole Stage 5 baseline assumption changes
- current portfolio miss: H60 is the parent but not yet stress-tested inside the stack
- candidate hypothesis to add: H524 restore LOO removing H60
- expected information gain: very high
- what evidence is missing: whether H60 is actually the anchor of the stack
- urgency: critical

### G04: No LOO removing H73 from the composite
- uncovered area: AdamW contribution attribution
- why it matters: H73 is the main trusted AdamW family
- current portfolio miss: no execution-grade proof of H73’s marginal value inside the stack
- candidate hypothesis to add: H525 restore LOO removing H73
- expected information gain: high
- what evidence is missing: whether AdamW eps scheduling matters once Muon-side changes are present
- urgency: high

### G05: No LOO removing H71 from the composite
- uncovered area: conditional usefulness of H71
- why it matters: H71 is weak solo but may still help compositionally
- current portfolio miss: we cannot tell whether H71 is signal or dead weight
- candidate hypothesis to add: H526 restore LOO removing H71
- expected information gain: medium
- what evidence is missing: whether H71 survives only as a stack helper
- urgency: medium

## Gap Hypotheses Added

### H522: H73 As Secondary Parent Baseline
- status: added_gap
- uncovered area closed: G01
- why now: compounds need a live AdamW parent comparator
- expected information gain: 5

### H523: LOO Removing H64
- status: added_gap
- uncovered area closed: G02
- why now: the composite must reveal H64’s marginal value
- expected information gain: 5

### H524: LOO Removing H60
- status: added_gap
- uncovered area closed: G03
- why now: the promoted parent must be stress-tested inside the stack
- expected information gain: 5

### H525: LOO Removing H73
- status: added_gap
- uncovered area closed: G04
- why now: H73 needs explicit marginal attribution inside the stack
- expected information gain: 5

### H526: LOO Removing H71
- status: added_gap
- uncovered area closed: G05
- why now: H71’s conditional value is still unresolved
- expected information gain: 4

## 15-Hypothesis Pool Before Final Prune

From `hypotheses.md` survivors:

- H501
- H503
- H504
- H506
- H508
- H517
- H518
- H519
- H520
- H502

Gap additions:

- H522
- H523
- H524
- H525
- H526

## Final Prune to 8 Execution Candidates

Selected for execution:

1. H522 — H73 secondary parent baseline
2. H504 — Warmdown-Aware Beta2 Second Phase
3. H517 — Shared Phase Variable Controls H60 and H73
4. H503 — full composite H64 + H60 + H73 + H71
5. H523 — LOO removing H64
6. H524 — LOO removing H60
7. H525 — LOO removing H73
8. H526 — LOO removing H71

Held for later loops:

- H506
- H501
- H502
- H508
- H518
- H519
- H520

Why the final prune looks like this:

- Stage 5 still includes the full composite and all four LOOs as requested
- the remaining two free candidate slots are spent on one Muon-side new code path (H504) and one cross-path new code path (H517)
- H64 fidelity repair is still implemented, but not given a dedicated solo slot because the composite attribution set already measures its marginal value
