# Stage 5 Context — Singles-First Rerun

## Why This Rerun Exists

The previous Stage 5 production batch mixed new singles and composites in one 10-run slate. That was the wrong search shape.

This rerun fixes the process:

- spend most population on new single mechanisms,
- keep a pure production baseline,
- keep one trusted prior single winner,
- keep the accidental composite winner as a frontier sentinel,
- defer new composite generation until after single survivors are known.

## Fixed Comparators

- `none`
  - pure production baseline
  - final BPB from prior 20k batch: `1.425995`

- `H73_solo`
  - strongest prior single in the mixed Stage 5 batch
  - final BPB from prior 20k batch: `1.413164`
  - interpretation: AdamW denominator control is still a live family at 20k

- `H64_H60_H73`
  - accidental best overall composite from the mixed batch
  - final BPB from prior 20k batch: `1.397882`
  - role here: frontier sentinel, not a new hypothesis

## What The Code Says Is Still Underexplored

Reading the production optimizer and `setup_optimizer()` points to several concrete surfaces that have not been sampled cleanly:

1. Muon cautious weight decay semantics
   - current mask uses post-orthogonalized update sign
   - alternate interpretation: use raw gradient sign so decay gating tracks the original signal

2. Muon state transitions at warmdown onset
   - H60 suggests the late regime matters disproportionately
   - current production path carries early second-moment state straight into warmdown
   - no run has tested a deliberate state reset or compression at the regime boundary

3. Muon group specialization by matrix shape
   - production groups matrices by shape for stacking
   - square-ish and rectangular matrices almost certainly do not want identical variance behavior

4. Orthogonalization depth as a scheduled mechanism
   - `ns_steps` is fixed today
   - no production-faithful run has tested whether early lower-depth updates help before late full orthogonalization

5. AdamW subgroup specialization
   - `setup_optimizer()` creates stable subgroups for:
     - `lm_head`
     - `embedding`
     - `value_embeds`
     - `resid_lambdas`
     - `x0_lambdas`
   - prior AdamW wins were global; no run has isolated embedding-only or x0-only behavior

6. AdamW state resets around warmdown
   - if warmdown is effectively a phase change, carrying all first-moment inertia unchanged may be suboptimal

7. AdamW initialization of second moment
   - current path seeds `exp_avg_sq` at zero for all groups
   - embedding-like groups might benefit from seeding from the first observed gradient to avoid an overly aggressive first denominator transition

## What We Believe Now

- H60 taught us that some wins are late-phase wins, not uniformly better from step 0.
- H73 showed AdamW denominator control is real and can win at 20k.
- H71 is not trusted as a standalone parent.
- H64 remains important, but its value is currently best treated as part of the carryover composite until a cleaner decomposition is rerun.

## Batch Design Rule

New population should be mostly mechanism changes, not constant nudges.

Good candidates in this rerun:

- alter what state is kept, reset, or seeded,
- split behavior by path or subgroup,
- change which signal a rule keys off,
- exploit the warmdown phase boundary explicitly.

Weak candidates:

- global coefficient twiddles without a new causal mechanism,
- reusing rejected `H504` or `H517` with cosmetic edits.
