# Stage 5 Frontier Composition Portfolio R1

This batch corrects the earlier process miss by carrying the previous frontier family directly into the composition step.

## Parent And New Family

- previous frontier parent: `H64_H60_H73`
- strongest new singles: `H533`, `H538`
- strongest new conditional helper pair: `H533_H535`

## Active Slate

| Slot | Role | Mutation | Why included |
| --- | --- | --- | --- |
| P0 | baseline | `none` | pure production comparator |
| P1 | parent | `H64_H60_H73` | previous best frontier composite |
| P2 | new single | `H533_shape_beta2` | strongest confirmed new Muon win |
| P3 | new single | `H538_seed_vsq` | strongest confirmed new AdamW/state win |
| P4 | new pair | `H533_H535` | strongest new pair discovered so far |
| P5 | frontier ext | `H64_H60_H73_H538` | does H538 extend the old frontier? |
| P6 | full | `H64_H60_H73_H533_H535` | chosen full stack for attribution |
| P7 | loo | `H60_H73_H533_H535` | remove `H64` from full stack |
| P8 | loo | `H64_H73_H533_H535` | remove `H60` from full stack |
| P9 | loo | `H64_H60_H533_H535` | remove `H73` from full stack |
| P10 | loo | `H64_H60_H73_H535` | remove `H533` from full stack |
| P11 | loo | `H64_H60_H73_H533` | remove `H535` from full stack |

## Main Questions

1. Can the previous frontier be improved by adding `H533` or `H538`?
2. Is `H533_H535` only a local family win, or can it extend the old frontier?
3. In the full stack, which component is genuinely carrying the gain?
