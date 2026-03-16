# Stage 5 Singles Rerun Portfolio

This batch has 11 runs:

- 8 new single-mechanism candidates
- 1 pure production baseline
- 1 prior single winner carryover
- 1 accidental composite winner carryover

## Active Slate

| Slot | Role | Mutation | Type | Why in batch now |
| --- | --- | --- | --- | --- |
| P0 | baseline | `none` | comparator | pure production delta |
| P1 | carry | `H73_solo` | prior single winner | strongest prior single at 20k |
| P2 | carry | `H64_H60_H73` | prior composite winner | accidental best overall frontier sentinel |
| P3 | new | `H531_raw_grad_wd` | Muon semantics | tests raw-signal cautious decay gating |
| P4 | new | `H532_muon_vreset` | Muon state reset | tests warmdown boundary second-moment reset |
| P5 | new | `H533_shape_beta2` | Muon specialization | tests rectangular vs square-ish beta2 split |
| P6 | new | `H534_ns_stage` | Muon algorithm schedule | tests staged orthogonalization depth |
| P7 | new | `H535_embed_eps` | AdamW subgroup | tests embedding-only epsilon schedule |
| P8 | new | `H536_x0_beta1_late` | AdamW subgroup | tests x0 late first-moment adaptation |
| P9 | new | `H537_embed_mom_reset` | AdamW state reset | tests warmdown boundary first-moment reset |
| P10 | new | `H538_seed_vsq` | AdamW init/state | tests first-gradient seeding of second moment |

## Mutation Naming Rule

The runner extracts `H###` features from the mutation string, so these labels are descriptive wrappers around the feature id:

- `none`
- `H73_solo`
- `H64_H60_H73`
- `H531_raw_grad_wd`
- `H532_muon_vreset`
- `H533_shape_beta2`
- `H534_ns_stage`
- `H535_embed_eps`
- `H536_x0_beta1_late`
- `H537_embed_mom_reset`
- `H538_seed_vsq`
