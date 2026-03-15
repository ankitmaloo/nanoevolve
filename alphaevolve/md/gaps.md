# Gaps

This file answers one question:
What important part of the search space is not currently represented by the shortlisted hypotheses?

## Gap-Finding Procedure
After pruning the initial hypothesis list:
1. Look at the surviving families.
2. Look at `variables.md` coverage.
3. Identify major untouched regions.
4. Add 2 to 3 gap-filling hypotheses even if they are lower confidence.

## Common Gap Categories
- no hypothesis attacks total work
- no hypothesis attacks memory movement
- no hypothesis attacks scheduling or occupancy
- no hypothesis attacks control-flow divergence
- no hypothesis tests a risky but informative move
- no hypothesis checks evaluator brittleness or overfitting
- no hypothesis examines correctness-preserving approximations
- no hypothesis addresses common-case specialization

## Gap Entry Template
### GXX: Title
- uncovered area:
- why it matters:
- current portfolio miss:
- candidate hypothesis to add:
- expected information gain:
- what evidence is missing:
- urgency:

## Coverage Audit Questions
- Which bottleneck classes have zero active hypotheses?
- Which benchmark slices are unsupported by the current portfolio?
- Which metric can currently regress without any portfolio slot noticing?
- Which move family is overrepresented because it is easy rather than strong?
- Which unknown in `variables.md` is still unresolved?

## Current Gaps
Add active entries below.
