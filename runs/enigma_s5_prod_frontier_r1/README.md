# Stage 5 Production Frontier Composition R1

This is the corrected second act for Stage 5.

It explicitly carries the previous-generation frontier into the composition batch instead of only composing within the new rerun family.

Design rule:

- keep pure production baseline
- keep previous best frontier composite as parent
- keep strongest new singles
- keep strongest new conditional pair
- test frontier extensions against that parent
- use leave-one-out ablations on the chosen full stack

This folder is fresh and does not overwrite prior Stage 5 artifacts.
