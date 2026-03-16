# Stage 5 Portfolio

This is the 10-run 20k production slate.

## Construction Rule

- 8 execution candidates from the final prune
- 1 promoted parent baseline (`H60_solo`)
- 1 pure production baseline (`none`)

This loop is intentionally attribution-heavy, but not H64-only:

- trusted parent carry,
- one Muon-side new code path,
- one cross-path new code path,
- one full composite,
- all four leave-one-out ablations.

## Active Production Slate

| Slot | Role | Mutation | Family | Target region | Operator family | Why selected now | Expected signal | Acceptance test | Kill condition | Overlap check | Candidate owner / prompt | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | baseline | `none` | baseline | full optimizer | none | pure production comparator | reference BPB at 20k | completes cleanly | invalid baseline | unique | production baseline | `1.425995` BASE |
| P1 | parent | `H60_solo` | muon_second_moment | Muon | schedule | promoted frontier parent | reproduce H60 late win | beats production again | loses to production | unique | carry parent unchanged | `1.428568` LOST to prod |
| P2 | parent2 | `H73_solo` | adamw_eps | AdamW | schedule | preserve strongest AdamW family as comparator | stays competitive long-run | beats production and stays near H60 | flat or negative | unique | 20k H73 confirmation | `1.413164` BEST_SINGLE |
| P3 | exploit | `H504_beta2_phase2` | muon_second_moment | Muon | schedule/state | new Muon-side code path driven by the 20k phase-shift result | earlier or larger late gain than H60 | beats H60 or shows stronger late regime | no advantage beyond H60 | unique | warmdown-aware beta2 phase shift | `1.472454` KILLED |
| P4 | wildcard | `H517_shared_phase` | stateful_cross_path | Muon + AdamW | state | first explicit shared phase variable across the two paths | coordinated late-phase gain | beats production and is competitive with H60/H73 | noisy or flat | unique | shared phase controller | `1.438283` interesting, lost |
| P5 | exploit | `H64_H60_H73_H71` | composite | full optimizer | schedule | highest-upside full frontier stack | best raw score if families complement | beats all parents | underperforms H60_H73 or H60 | unique | full composite | `1.430382` full stack lost |
| P6 | loo | `H60_H73_H71` | ablation | full optimizer | schedule | remove H64 from full stack | quantify H64 contribution | clearly shifts from P5 | indistinguishable from P5 | labeled ablation | LOO no H64 | `1.424924` better than full |
| P7 | loo | `H64_H73_H71` | ablation | full optimizer | schedule | remove H60 from full stack | test whether H60 is the core driver | clearly shifts from P5 | indistinguishable from P5 | labeled ablation | LOO no H60 | `1.436129` worse than full |
| P8 | loo | `H64_H60_H71` | ablation | full optimizer | schedule | remove H73 from full stack | quantify H73 contribution | clearly shifts from P5 | indistinguishable from P5 | labeled ablation | LOO no H73 | `1.422695` better than full |
| P9 | loo | `H64_H60_H73` | ablation | full optimizer | schedule | remove H71 from full stack | test whether H71 helps conditionally | clearly shifts from P5 | indistinguishable from P5 | labeled ablation | LOO no H71 | `1.397882` BEST overall |

## Mutation Naming Map

- `none`
- `H60_solo`
- `H73_solo`
- `H504_beta2_phase2`
- `H517_shared_phase`
- `H64_H60_H73_H71`
- `H60_H73_H71`
- `H64_H73_H71`
- `H64_H60_H71`
- `H64_H60_H73`

The runner extracts `H60`, `H64`, `H71`, `H73`, `H504`, and `H517` features from the mutation name, so these names are descriptive labels around the actual feature set.

## Why This Portfolio And Not A More Inventive One

Stage 5 is the first long-horizon production loop after the Stage 4 extension. The highest-value unresolved questions are:

1. Does a new Muon-side mechanism beat or extend H60?
2. Does a new cross-path mechanism beat simple independent schedules?
3. If a full stack wins, which components actually matter?

This portfolio answers those while still satisfying the composite + LOO requirement.

## Rejected This Loop

- hypothesis: H501 faithful H64 solo
  reason: implementation prerequisite is kept, but a dedicated solo slot was traded for broader coverage
  what would need to change to revive it: if the LOO no H64 result is ambiguous

- hypothesis: H502 H60 + H73 compound
  reason: valuable exploit path, but displaced by the stronger requirement to include two outside ideas plus full composite attribution
  what would need to change to revive it: if the composite suggests the H60+H73 core is dominant

- hypothesis: H506 loss-reactive beta2
  reason: promising next-loop stateful Muon path, but not selected in the first 20k production slate
  what would need to change to revive it: if H504 confirms warmdown-phase sensitivity

- hypothesis: H518 Muon volatility gates AdamW beta1
  reason: higher-risk cross-path control than H517
  what would need to change to revive it: if H517 is directionally promising but too weak

- hypothesis: H520 attention-vs-MLP beta2 split
  reason: strong specialization hypothesis, but lower urgency than the selected slate
  what would need to change to revive it: if H60-family ideas still dominate after Stage 5
