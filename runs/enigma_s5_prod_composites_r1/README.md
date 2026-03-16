# Stage 5 Production Composites Rerun R1

Composite follow-up to the Stage 5 singles rerun.

This batch is staged after the singles-first correction and is focused on:

- key pairwise compounds,
- a full 4-way stack,
- leave-one-out ablations of that 4-way stack.

The component set is:

- `H532` Muon warmdown second-moment reset
- `H533` shape-aware Muon beta2
- `H535` embedding/value-embed epsilon schedule
- `H538` seeded embedding/value-embed second moment

Why this set:

- `H533` and `H532` are the current real single winners from the rerun.
- `H538` is the strongest unfinished AdamW-side live candidate.
- `H535` is weak solo so far, but it is on a distinct AdamW denominator surface and is plausible as a conditional helper.

This folder is fresh and does not overwrite previous Stage 5 artifacts.
