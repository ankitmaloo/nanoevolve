# Stage 5 Composite Portfolio R1

This batch is built from the singles rerun survivor set plus one promising AdamW-side unfinished candidate.

## Active Slate

| Slot | Role | Mutation | Why included |
| --- | --- | --- | --- |
| P0 | baseline | `none` | pure production comparator |
| P1 | parent | `H533_shape_beta2` | best current confirmed single |
| P2 | pair | `H532_H533` | strongest confirmed Muon pair |
| P3 | pair | `H532_H538` | tests warmdown reset + AdamW second-moment seeding |
| P4 | pair | `H533_H538` | tests shape-aware Muon + AdamW second-moment seeding |
| P5 | pair | `H533_H535` | tests whether the weaker AdamW eps split becomes conditional |
| P6 | full | `H532_H533_H535_H538` | maximal 4-way stack for attribution |
| P7 | loo | `H533_H535_H538` | full stack without `H532` |
| P8 | loo | `H532_H535_H538` | full stack without `H533` |
| P9 | loo | `H532_H533_H538` | full stack without `H535` |
| P10 | loo | `H532_H533_H535` | full stack without `H538` |

## Read Rule

The full stack and its LOOs are the main readout. The key question is which of the four mechanisms are additive, redundant, or harmful in combination.
