# Stage 6 Evolution Strategy

## Lessons from Stages 1-5

These are hard-won rules, not theory. Each comes from a specific failure.

### 1. Eval fidelity is everything
**Stage 1-3 failure**: The SpecCandidateOptimizer didn't include production schedules (LR warmdown, momentum warmup, WD decay). Trust ratio "won" — later proven to be an artifact of the broken eval loop. Three stages of compute wasted.

**Rule**: Every eval run must use the exact production optimizer (`MuonAdamW`) with all production schedules active. The monkey-patch approach (Stage 4+) is the minimum viable fidelity. Never build a separate optimizer implementation to "approximate" production.

### 2. Small data lies
**Stage 5 failure**: 3 data shards caused data recycling (~11% of data seen per epoch). Compound mutations (H64_H60_H73) won on 3 shards because they overfit to the repeated data patterns. On 1500 shards, H73 solo destroyed every compound by 3x.

**Rule**: Use enough data shards that zero recycling occurs within the step budget. For 20k steps × 1024 tokens = 20M tokens, need at least 1 shard (~92MB ≈ ~60M tokens). Use 1500+ shards to be safe. Never draw conclusions from <100 shards.

### 3. Solos before compounds
**Stage 5 failure**: Jumped to 4-way compounds before understanding solo contributions. H71 was harmful solo but was included in compounds, dragging everything down. H73 was the real winner but was obscured by being part of compounds.

**Rule**: Always test solos first. Only compound mutations that win solo. Never include a mutation in a compound unless it has positive solo delta. LOO (leave-one-out) ablation is mandatory for any compound.

### 4. Compounds from the same cluster don't stack
**Stage 5 fulldata finding**: H64_H60_H73 (all schedule warmups) performed 3x worse than H73 solo. Mutations attacking the same surface (optimizer early-phase) compete rather than compound.

**Rule**: Cross-cluster compounds (eps + momentum + WD) are more likely to stack than same-cluster compounds (eps variant A + eps variant B). When composing, always pick from different clusters/surfaces.

### 5. Simpler wins
**Stage 5 finding**: Full 4-way was worst. 3-way was mediocre. Solo was best. Every stage showed the same pattern: adding more mutations degrades performance.

**Rule**: Default to the simplest winner. Only compound if the compound provably beats the best solo. The burden of proof is on the compound, not the solo.

### 6. Baselines must be in-batch
**Stage 4 finding**: Baseline BPB varied between job batches due to data ordering, GPU state, etc. Comparing across batches gave misleading deltas.

**Rule**: Every experiment batch must include a `none` (production baseline) run with identical data, seed, and hardware. Never compare across batches.

### 7. Single seed is directional, not conclusive
**All stages**: seed=42 only. Rankings within 0.005 BPB are noise. H64_H60_H73 vs H533_H535 differed by 0.0015 BPB on 3 shards — this is meaningless.

**Rule**: A delta >0.01 BPB on single seed is directional signal. A delta >0.03 BPB is strong signal. Anything <0.01 requires multi-seed validation before trusting.

---

## Stage 6 Tournament Design

### Hypothesis inventory

20 hypotheses in 3 clusters:
- **Cluster A** (6): Adaptive eps — H601, H602, H603, H604, H605, H607
- **Cluster B** (5): Cross-optimizer coordination — H603, H604, H606, H608, H620
- **Cluster C** (9): New surfaces — H609, H610, H611, H612, H613, H614, H615, H616, H617, H618, H619

(Some hypotheses appear in multiple clusters — H603, H604 are both adaptive eps AND cross-optimizer.)

### Round 1: Screen (kill bad ideas fast)

**Setup**: 5k steps, 1 GPU, 1500 shards, seed=42, batch=1024
**Runs**: 20 hypotheses + 1 baseline + 1 H73 carry = 22 runs
**Parallelism**: 8 runs at once on 8xH100 (3 batches of 8)
**Time**: ~3 min/run × 3 batches = ~10 min total
**Kill threshold**: Must beat baseline. Doesn't need to beat H73 yet.

**Output**: Ranked list. Kill bottom 50% (any that don't beat baseline). ~10 survivors.

### Round 2: Solo validation (20k steps, full data)

**Setup**: 20k steps, 1 GPU, 1500 shards, seed=42, batch=1024
**Runs**: ~10 survivors + baseline + H73 carry = ~12 runs
**Parallelism**: 8 at once + 4 on second pass
**Time**: ~10 min/run × 2 batches = ~20 min
**Win threshold**: Must beat H73 solo, or beat baseline by >0.03 BPB on a different surface (potential compound partner).

**Output**: Top 5 solos, ranked. Note which cluster each belongs to.

### Round 3: Cross-cluster compounds

**Setup**: 20k steps, 1 GPU, 1500 shards, seed=42, batch=1024

Take the top solo from each cluster. Form 2-way and 3-way cross-cluster compounds only. Example: if H601 (eps, Cluster A) and H614 (momentum, Cluster C) both win solo, test H601+H614.

**Runs**: ~10-15 compounds + baseline + best solo carry
**Mandatory**: LOO ablation on any 3-way compound
**Kill rule**: If compound doesn't beat the best solo, kill it immediately

### Round 4: Multi-seed validation

**Setup**: 20k steps, 1 GPU, 1500 shards, seeds={42, 137, 256}
**Runs**: Top 3 candidates × 3 seeds + baseline × 3 seeds = 12 runs
**Win threshold**: Mean BPB across seeds must beat baseline by >0.01. No seed can be worse than baseline.

### Round 5: Production batch validation

**Setup**: Production batch (524288 tokens), 8 GPUs, `DistMuonAdamW`, auto-computed steps
**Runs**: Top 2 candidates + baseline = 3 runs
**Runner**: `enigma/run_stage5_dist.py`
**Purpose**: Verify the mutation holds at 512x batch size with distributed training

### Round 6: Speedrun integration

**If a candidate passes Round 5:**
1. Port the change into `nanochat/nanochat/optim.py` (or `base_train.py` for schedule-only)
2. Run full `speedrun.sh`: d24, target-param-data-ratio=9.5, fp8, 8xH100
3. Measure wall-clock to GPT-2 CORE score 0.256525
4. Compare to baseline speedrun

---

## What to track per run

Every run must produce a JSON with:
```json
{
  "mutation": "H601",
  "features": ["H601"],
  "final_validation_bpb": 1.628,
  "best_validation_bpb": 1.628,
  "curve": [{"step": 0, "val_bpb": 3.19}, ...],
  "mean_step_time_ms": 28.0,
  "total_time_s": 550,
  "steps_completed": 20000,
  "config": {"batch_size": 1024, "depth": 12, "seed": 42, "shards": 1500}
}
```

Every experiment batch must produce:
- `results/*.json` — per-run metrics
- `logs/*.log` — full training output
- `portfolio.md` — what was tested and why
- Postmortem JSON after analysis

---

## Decision rules

| Situation | Action |
|-----------|--------|
| Mutation worse than baseline at 5k | Kill |
| Mutation worse than baseline at 20k | Kill |
| Mutation beats baseline but not H73 at 20k | Keep as potential compound partner only |
| Mutation beats H73 solo at 20k | Promote to multi-seed |
| Compound doesn't beat best solo | Kill the compound, keep the solos |
| Multi-seed mean < baseline + 0.01 | Kill |
| Any seed worse than baseline | Flag, investigate |
| Production batch delta < 0.005 | Not worth deploying |

---

## What if nothing beats baseline?

If all 20 hypotheses lose to baseline in Round 1, that itself is data. Here's the decision tree:

### All lose to baseline
**Diagnosis**: The search space is wrong, not just the hypotheses.
- H73 remains the winner. Deploy H73 as-is.
- Analyze the failure patterns: did everything lose by the same margin (surface is tapped out) or different margins (some surfaces are more promising)?
- Pivot to the gaps identified in README.md — LR schedule shape, mixed precision interactions, data-aware scheduling. These are fundamentally different surfaces.

### Some beat baseline, none beat H73
**Diagnosis**: H73 is strong but the new ideas add to a different surface.
- Keep the best baseline-beaters as compound partners for H73.
- Run Round 3 with H73 + best new surface winner. The compound might beat H73 solo even though the new idea alone doesn't.
- If H73 + partner still doesn't beat H73 solo, H73 is the ceiling for this architecture/scale. Move to production validation.

### Everything is within noise of baseline (±0.005 BPB)
**Diagnosis**: 5k steps isn't enough to separate these ideas. Or the model is too small.
- Skip Round 1 screening. Go straight to Round 2 (20k steps) with the top 10 by raw BPB.
- If 20k also shows no separation, the mutations are too subtle for this scale. They might only matter at d24/production batch.

### H73 itself doesn't beat baseline (regression)
**Diagnosis**: Something changed in the eval setup. Stop everything.
- Compare this batch's baseline BPB to Stage 5's baseline BPB. If they differ by >0.01, the infrastructure changed.
- Check: data shard ordering, torch version, CUDA version, random seed path. Don't run experiments on broken infrastructure.

---

## Anti-patterns to avoid

1. **Don't re-test what Stage 5 killed**: H71 (beta1 warmup) is dead. H504 (beta2 phase2) is dead. Don't recombine them.
2. **Don't compound early**: Solos first. Always.
3. **Don't trust 3-shard results**: 1500 shards minimum.
4. **Don't compare across batches**: Always include baseline in the same job.
5. **Don't add complexity for <0.01 BPB**: If H601 (zero new hyperparams) gets +0.08 and H607 (new hyperparams) gets +0.09, prefer H601.
6. **Don't optimize for 20k-step small-batch**: The real target is the speedrun (production batch, d24, 8xH100). Small-batch results are screening only.
