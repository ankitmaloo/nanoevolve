# Enigma: Experiment Design Principles

How to design hypotheses and experiments that maximize information regardless of outcome.

---

## The core rule: every experiment must be informative both ways

A good hypothesis isn't one you expect to win. It's one where **winning tells you X** and **losing tells you Y**, and both X and Y narrow the search space.

Before running any hypothesis, write down:
1. **If it wins**: what does that prove about the optimizer?
2. **If it loses**: what does that rule out?
3. **If it's within noise**: what can't we conclude?

If you can't answer all three, the hypothesis isn't ready.

---

## Examples from Stages 1-5

### H73 (eps schedule 1e-6 → 1e-10)
- **Won (+0.092 BPB)**: Proves early-phase eps protection matters. The zero-initialized second moments + tiny eps = dangerous effective step sizes. AdamW's variance estimate needs time to converge.
- **If it had lost**: Would have proved that production's fixed eps=1e-10 is already fine, and early-phase instability isn't a real problem at this scale. Would have killed the entire "adaptive eps" cluster before it started.

### H71 (beta1 warmup 0.8 → 0.95)
- **Lost (-0.019 BPB, harmful)**: Proves that momentum warmup hurts. The optimizer needs full momentum from the start — gradients are already stable enough. Rules out the entire "gentle momentum ramp" family.
- **If it had won**: Would have proved early gradients are noisy and need damping. Would have opened a cluster of momentum schedule ideas.

### H64_H60_H73 compound vs H73 solo
- **Lost (H73 solo 3x better)**: Proves same-surface compounds compete rather than stack. Schedule warmups fight each other for the same early-phase window. Rules out all same-cluster compounds.
- **If it had won**: Would have proved that multiple early-phase corrections are additive. Would have justified deeper compound search within Cluster A.

### 3-shard vs 1500-shard validation
- **1500-shard reversed rankings**: Proves data recycling inflates compound scores. Small-data results are unreliable for ranking mutations. Rules out any result from <100 shards.
- **If rankings had held**: Would have proved 3 shards is sufficient for screening (huge compute savings). Would have validated the original Stage 5 compound rankings.

---

## How to design hypotheses that maximize information

### 1. Test mechanisms, not parameters

Bad: "Try eps=1e-7 instead of 1e-10"
Good: "Schedule eps from high to low based on second-moment convergence"

The bad version tells you one point worked or didn't. The good version tests whether the *mechanism* (variance estimate convergence) is the right lens. If it wins, you know the mechanism matters and can tune. If it loses, you know the mechanism is wrong and can skip all its variants.

### 2. Design pairs that bracket the truth

Run hypotheses in pairs that make opposite predictions:

| Pair | H_A predicts | H_B predicts | Outcome tells you |
|------|-------------|-------------|-------------------|
| H601 (eps from bias correction) vs H607 (homeostatic eps) | Eps should follow a fixed mathematical schedule | Eps should respond to actual training dynamics | Whether the optimizer needs feedforward or feedback control |
| H610 (adaptive WD) vs H618 (WD-eps co-schedule) | WD should respond to overfit signals | WD should be mechanically coupled to eps | Whether regularization needs to be data-aware or just optimizer-aware |
| H614 (adaptive Muon momentum) vs H619 (Muon variance reset) | Muon needs continuous adaptation | Muon needs sharp regime changes | Whether Muon's problem is gradual drift or phase mismatch |

When one wins and the other loses, you learn about the *mechanism*, not just the *mutation*.

### 3. Use cluster structure to identify surfaces

Group hypotheses by what they modify (the "surface"):

| Surface | Hypotheses | What winning here proves |
|---------|-----------|------------------------|
| AdamW eps | H601-H605, H607 | Early-phase protection is the dominant effect |
| Cross-optimizer | H603, H604, H606, H608 | Muon and AdamW interfere destructively |
| Weight decay | H610, H618 | Regularization dynamics matter |
| Muon internals | H611, H614, H615, H619 | Muon's default settings are suboptimal |
| Layer-wise | H609, H616 | Not all layers need the same treatment |

If an entire cluster loses, the surface is tapped out. If a cluster wins, go deeper into that surface next stage.

### 4. Solos isolate; compounds confirm

Solo runs tell you **whether a surface matters**. Compound runs tell you **whether surfaces are independent**.

- Solo A wins, Solo B wins, A+B wins more → surfaces are independent, effects stack
- Solo A wins, Solo B wins, A+B ≈ best solo → surfaces interact, only one effect is real
- Solo A wins, Solo B wins, A+B worse than both → surfaces conflict, they cancel each other

Never compound before you have solo data. The compound result is uninterpretable without it.

### 5. Control for one variable at a time

Each hypothesis should change exactly one thing from the baseline (or from H73 if building on it). If you change eps AND momentum AND weight decay, a win tells you nothing about which mattered.

Exception: compounds in Round 3, but only after solos have validated each component.

---

## Information value ranking

When choosing what to run next, prioritize by information value:

### High information value
- **New surface, simple mechanism**: H609 (layer-wise LR). Win or lose, you learn whether layer differentiation matters. Zero overlap with existing knowledge.
- **Mechanism test with opposite prediction**: H601 vs H607. One must be closer to right.
- **Direct test of a Stage 5 lesson**: "Do same-cluster compounds stack at 1500 shards?" Already answered (no), but the lesson generalizes.

### Medium information value
- **Refinement of known winner**: H601-H605 are all adaptive eps variants. If H73 is the right idea, one of these might be better. But they don't tell you anything new about the search space.
- **Compound of known winners**: Confirms independence but doesn't discover new surfaces.

### Low information value
- **Parameter sweep**: "Try eps=1e-7, 1e-8, 1e-9." You learn the optimal value but nothing about why.
- **Re-testing killed ideas**: H71 is dead. Running H71 with a slightly different schedule is unlikely to change the conclusion.
- **Compound of same-cluster ideas**: Stage 5 already proved these don't stack.

---

## The postmortem loop

After every round, update the knowledge base:

1. **What won?** → New frontier candidate
2. **What lost?** → Surfaces/mechanisms ruled out
3. **What was within noise?** → Needs longer runs or different scale to resolve
4. **What surprised us?** → Assumptions that were wrong. These are the most valuable.
5. **What should we test next?** → Derived from 1-4, not from the original hypothesis list

The hypothesis list is a starting point, not a contract. If Round 1 reveals that all Cluster A ideas lose but H614 (Muon momentum) wins big, Stage 6 Round 2 should pivot to Muon momentum variants, not continue testing eps ideas because they were on the list.

---

## Checklist before running any experiment

- [ ] Hypothesis has a win interpretation AND a loss interpretation
- [ ] Baseline (`none`) is in the same batch
- [ ] Previous winner (H73) is in the same batch
- [ ] Data shards ≥ 1500 (or explicitly labeled as screening)
- [ ] Single variable changed from baseline
- [ ] Result JSON schema matches the tracking format
- [ ] Postmortem template ready before the run starts
