# Portfolio

This is the execution slate for the current loop.

## Selection Rule
Choose the top 5 hypotheses based on:
- high upside,
- high information gain,
- real distinctness,
- enough feasibility to test now,
- broad enough coverage across the search space.

Do not fill this list with five variants of the same idea.

## Desired Mix
- 1 conservative fix
- 1 memory or locality move
- 1 scheduling or parallelism move
- 1 algorithmic or work-reduction move
- 1 risky but high-information move

## Active Portfolio Template
| Slot | Role | Hypothesis | Family | Target region | Operator family | Why selected now | Expected signal | Acceptance test | Kill condition | Overlap check | Candidate owner / prompt | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | scout | | | | | | | | | unique | | pending |
| P2 | scout | | | | | | | | | unique | | pending |
| P3 | exploit | | | | | | | | | unique | | pending |
| P4 | exploit | | | | | | | | | unique | | pending |
| P5 | wildcard | | | | | | | | | unique | | pending |

## Execution Rules
- One slot should test one main mechanism.
- If a candidate mixes multiple mechanisms, state why that is necessary.
- Every slot must name the exact evidence that would count as success.
- Every slot must name the exact evidence that would kill it.
- Keep at least one slot focused on information gain, not just expected score.
- Before a generation is final, compare every slot against every other slot.
- If two slots hit the same region with the same operator family and same bottleneck, merge or kill one of them.
- The only acceptable overlap is an intentional ablation pair, and it must be labeled as such.
- Do not let all slots become local exploit probes unless the search already has broad confirmed coverage.

## Elimination Notes
Use this section to explain why shortlisted hypotheses did not make the active top 5.

### Rejected This Loop
- hypothesis:
  reason:
  what would need to change to revive it:
