# Enigma Checkpoint — 2026-03-17

## Current Frontier

**Best result: H73 solo = 1.6289 BPB at 20k steps on 1500 shards** — beats production baseline (1.7206) by **+0.092 BPB**.

This is a single change: AdamW eps log-linear schedule from 1e-6→1e-10 over total steps. Zero-overhead schedule change to production MuonAdamW. Validated on 1500 data shards (~23% of full ClimbMix-400B, vs 3 shards in Stage 5), largely eliminating the data-recycling confounder.

Previous champion H64_H60_H73 (3-compound) drops to #3 at 1.6907 BPB (+0.030 vs baseline). **H73 alone accounts for most of the gain — the other mutations add noise at scale.**

### 1500-shard validation results (Stage 5 Fulldata, 2026-03-17)

| Rank | BPB | Delta | Mutation | Notes |
|------|-----|-------|----------|-------|
| 1 | **1.6289** | **+0.0917** | H73_solo | Clear winner, eps schedule |
| 2 | 1.6899 | +0.0307 | H64_H60_H71 | LOO (no H73) |
| 3 | 1.6907 | +0.0299 | H64_H60_H73 | Previous champion |
| 4 | 1.6946 | +0.0260 | H60_H73_H71 | LOO (no H64) |
| 5 | 1.6966 | +0.0240 | H517 | Shared phase beta2 |
| 6 | 1.7013 | +0.0194 | H60_solo | Beta2 warmup |
| 7 | 1.7206 | — | none | **PRODUCTION BASELINE** |
| 8 | 1.7242 | -0.0036 | H64_H73_H71 | LOO (no H60) |
| 9 | 1.7281 | -0.0074 | H504 | Beta2 phase2 |
| 10 | 1.7689 | -0.0483 | H64_H60_H73_H71 | Full 4-way — worst |

### What the winning mutation does

| ID | Mechanism | Where applied | Code | Diff |
|----|-----------|--------------|------|------|
| H73 | AdamW eps log-linear 1e-6→1e-10 over total steps | AdamW kernel patch | `enigma/stage5_patch.py:_patch_adamw_eps_schedule` | [`diff`](runs/enigma_s5_fulldata/diffs/H73_eps_schedule.diff) |

**Why it works**: Production already uses `eps=1e-10` (see `nanochat/nanochat/gpt.py:376-380`) — the lowest reasonable value. But this is set from step 0, when second moments (`exp_avg_sq`) are zero-initialized and unreliable. With `eps=1e-10` and tiny second moments, early AdamW steps have enormous effective step sizes on low-variance dimensions. H73 starts at `eps=1e-6` (1000x higher), providing implicit per-coordinate clipping during the noisy early phase, then anneals to the same `1e-10` production value. The final state is identical to production — it's only the early-phase protection that differs.

### Stage history at a glance

| Stage | Steps | Data | Eval method | Key outcome |
|-------|-------|------|-------------|-------------|
| 1-3 | 1k-5k | ~3 shards | SpecCandidateOptimizer (broken: no prod schedules) | Trust ratio won — later proven artifact |
| 4 | 5k | ~3 shards | Production MuonAdamW monkey-patched | H64 +0.063, H73 +0.046, H60 +0.031 |
| 4ext | 5k+20k | ~3 shards | Same | H60 confirmed at 20k, H71 weakened |
| 5 | 20k | ~3 shards | Same, + new mutations H531-H538 | H64_H60_H73 = 1.3979 (best) |
| 5-fulldata | 20k | 1500 shards (~23% ClimbMix) | Same code, 8xH100, 1 GPU/run | **H73 solo = 1.6289 (best)** |

---

## Key Findings from 1500-Shard Validation

**H73 (eps schedule) is the real deal.** +0.092 BPB solo — 3x the gain of any compound. The eps schedule addresses a fundamental issue: AdamW has `warmup_ratio=0.0` (no LR warmup) + zero-initialized second moment = enormous effective step sizes on low-variance dimensions early. The eps schedule acts as an implicit per-coordinate clip. With diverse data (1500 shards ≈ 138B tokens, ~zero recycling at 20k×1024 = 20M tokens processed), this effect is even stronger than on 3 shards.

**Compounds hurt at scale.** H64_H60_H73 was the 3-shard champion but is 3x worse than H73 solo on 1500 shards. The compounding was likely overfitting to the small dataset's noise patterns. H64 (Nesterov blend) and H60 (beta2 warmup) add marginal or negative value when data is diverse.

**H71 confirmed harmful.** Every compound containing H71 is worse than its H71-free counterpart. The full 4-way (H64_H60_H73_H71) is the worst performer at 1.769 BPB, below baseline.

**H60 (beta2 warmup) has modest solo value.** +0.019 BPB solo, which is real but small compared to H73.

---

## Experimental Caveats

### 1500-shard run (Stage 5 Fulldata) — caveats

**Partial dataset**: 1500 out of 6543 shards (~23% of ClimbMix-400B, ~138B tokens). No data recycling at 20M tokens processed, but the data distribution may not fully represent the complete dataset.

**Small batch (512x below production)**: 1024 tokens (device_batch=2, seq_len=512). Production uses 524,288 tokens/step — 512x larger. Larger batch means lower gradient variance per step, so `exp_avg_sq` fills in with accurate estimates faster. The "dangerous early phase" where eps dominates the denominator is shorter at large batch. H73's benefit should still be positive (second moments are still zero-initialized regardless of batch size) but the magnitude may shrink. This is the single biggest remaining uncertainty.

**Small model**: GPT-2 scale (12 layers, 768 dim, 6 heads, ~124M params). Optimizer behavior changes at larger scales.

**Single seed**: seed=42 only. No multi-seed robustness validation.

**Single GPU per run**: Each mutation ran on 1 H100 GPU. Production uses 8-GPU `DistMuonAdamW` (distributed). However, with batch=1024 and no grad accumulation, single vs multi-GPU should be equivalent.

**BPB absolute values differ from Stage 5**: 1.63-1.77 range (1500-shard) vs 1.40-1.47 (3 shards). Higher BPB on 1500-shard is expected — more diverse data is harder to learn from in 20k steps with tiny batch. Relative rankings and deltas are what matter.

### 3-shard runs (Stages 1-5) — caveats

**Reduced dataset**: ~3 out of 6543 shards. Data recycling inflated all results and may have favored compounds that overfit to repeated patterns.

**Bottom line**: The 1500-shard run is the most authoritative result so far. H73 is the validated winner. Compounds from Stage 5 were likely artifacts of data recycling. Full dataset (6543 shards) and production batch size validation still pending.

---

## Prediction Scorecard

Prior predictions (from "What Will and Won't Hold at Full Scale") vs actual full-data results:

| Prediction | Actual | Correct? |
|-----------|--------|----------|
| H73 likely to hold | **+0.092 BPB, #1 overall** | YES |
| H60 likely to hold | +0.019 BPB solo, modest | PARTIALLY — holds but small |
| H64 uncertain | +0.030 only in compound, not solo tested | PARTIALLY — compounds weaker |
| H71 likely won't hold | Worst in every compound | YES |
| Exact compound rankings will flip | H73 solo > all compounds | YES |

---

## Stage 5 — Full Results (33 runs, all 20k steps)

### Sub-run 1: `enigma_s5_prod` (10 runs — solos, composite, LOO ablations)

| Rank | BPB | Mutation | Role |
|------|-----|----------|------|
| 1 | **1.3979** | H64_H60_H73 | LOO (no H71) — **BEST** |
| 2 | 1.4131 | H73_solo | Best solo single |
| 3 | 1.4227 | H64_H60_H71 | LOO (no H73) |
| 4 | 1.4249 | H60_H73_H71 | LOO (no H64) |
| 5 | 1.4260 | none | **PRODUCTION BASELINE** |
| 6 | 1.4286 | H60_solo | Lost at 20k solo |
| 7 | 1.4304 | H64_H60_H73_H71 | Full 4-way — worse than LOO |
| 8 | 1.4361 | H64_H73_H71 | LOO (no H60) |
| 9 | 1.4383 | H517_shared_phase | Lost |
| 10 | 1.4725 | H504_beta2_phase2 | Killed |

### Sub-run 2: `enigma_s5_prod_composites_r1` (11 runs — new mechanisms H531-H538)

New singles: H531 raw-grad WD mask, H532 Muon variance reset, H533 shape-aware beta2, H534 staged NS depth, H535 embedding eps, H536 x0 beta1 late, H537 embedding momentum reset, H538 seeded second moment.

| Rank | BPB | Mutation |
|------|-----|----------|
| 1 | **1.3994** | H533_H535 — near-champion pair |
| 2 | 1.4247 | H532_H533_H538 |
| 3 | 1.4285 | H532_H533_H535 |
| 6 | 1.4339 | none (baseline) |
| 9 | 1.4639 | H532_H533_H535_H538 (full stack regresses) |

### Sub-run 3: `enigma_s5_prod_frontier_r1` (12 runs — extend old frontier with new singles)

| Rank | BPB | Mutation |
|------|-----|----------|
| 1 | 1.4056 | H60_H73_H533_H535 (old frontier + new pair, no H64) |
| 2 | 1.4077 | H64_H60_H73 (repeat of champion) |
| 5 | 1.4164 | H64_H60_H73_H533_H535 (full extension — worse) |
| 7 | 1.4240 | none (baseline) |

Adding new singles to H64_H60_H73 doesn't help. H64 and H533 may interfere.

---

## Knowledge Base

### What works
- **H73 (eps schedule)** — validated on 1500 shards, +0.092 BPB solo, the single biggest win
- H60 (beta2 warmup) — modest solo value (+0.019), helps in some compounds
- Simple monotonic schedules beat non-monotonic ones

### What doesn't work
- **Compounds** — all compounds underperform H73 solo on 1500 shards. 3-shard compound wins were artifacts
- H71 (beta1 warmup) — hurts in every compound, confirmed harmful across data scales
- H64 (Nesterov blend) — uncertain solo value, doesn't help H73 in compounds at scale
- H504 (beta2 phase2), H517 (shared phase) — below baseline or marginal
- Trust ratio — artifact of broken eval loop (Stages 1-3)
- Embedding weight decay, scalar LR warmup, momentum overshoot
- Full stacks consistently underperform simpler combinations

### Gotchas
- `TORCHDYNAMO_DISABLE=1` mandatory on Slurm cluster (triton fails)
- Stage 4 H64 was approximated by overwriting `group['momentum']` — not faithful. `stage5_patch.py` has the faithful port
- Stage 1-3 results are unreliable — eval loop was missing production schedules
- Baseline BPB varies between batches — always compare within same batch

---

## Key Code Files

| File | Purpose |
|------|---------|
| `enigma/stage5_patch.py` | **Current monkey-patch module**. H64 faithful Nesterov blend, H71 beta1 warmup, H73 eps schedule, H71+H73 combined, H531 raw grad WD, H517 shared phase, H538 seed second moment. Schedule-only mutations applied in training loop. |
| `enigma/run_stage5.py` | Single-GPU training loop with compound feature flags |
| `enigma/run_stage5_dist.py` | 8xH100 distributed version (DistMuonAdamW, production batch) |
| `nanochat/nanochat/optim.py` | Production optimizer: `adamw_step_fused` (L20-49), `muon_step_fused` (L90-147), `MuonAdamW` (L152-291) |
| `nanochat/scripts/base_train.py` | Production schedules: `get_lr_multiplier` (L362), `get_muon_momentum` (L374), `get_weight_decay` (L380) |
| `nanochat/nanochat/gpt.py` | `GPT.setup_optimizer` (L356-394) — creates param groups |
| `runs/enigma_s5_fulldata/diffs/` | **Annotated diffs for all 1500-shard mutations** (see below) |
| `runs/enigma_s4_prod/diffs/` | Annotated diffs for all Stage 4 mutations |
| `runs/enigma_stage*_postmortem.json` | Postmortems for stages 2, 3, 4, 4ext, 5, 5-fulldata |

### Diffs for 1500-shard validated mutations

| Diff | Mutation | Result | Type |
|------|----------|--------|------|
| [`H73_eps_schedule.diff`](runs/enigma_s5_fulldata/diffs/H73_eps_schedule.diff) | H73 solo | **+0.092 #1** | AdamW kernel |
| [`H60_beta2_warmup.diff`](runs/enigma_s5_fulldata/diffs/H60_beta2_warmup.diff) | H60 solo | +0.019 #6 | Training loop |
| [`H64_nesterov_blend.diff`](runs/enigma_s5_fulldata/diffs/H64_nesterov_blend.diff) | H64 | in compounds only | Muon kernel |
| [`H71_beta1_warmup.diff`](runs/enigma_s5_fulldata/diffs/H71_beta1_warmup.diff) | H71 | HARMFUL | AdamW kernel |
| [`H504_beta2_phase2.diff`](runs/enigma_s5_fulldata/diffs/H504_beta2_phase2.diff) | H504 | -0.007 #9 | Training loop |
| [`H517_shared_phase.diff`](runs/enigma_s5_fulldata/diffs/H517_shared_phase.diff) | H517 | +0.024 #5 | AdamW kernel + loop |
| [`H64_H60_H73.diff`](runs/enigma_s5_fulldata/diffs/H64_H60_H73.diff) | H64+H60+H73 | +0.030 #3 | Combined |
| [`H64_H60_H71.diff`](runs/enigma_s5_fulldata/diffs/H64_H60_H71.diff) | H64+H60+H71 | +0.031 #2 | Combined |
| [`H60_H73_H71.diff`](runs/enigma_s5_fulldata/diffs/H60_H73_H71.diff) | H60+H73+H71 | +0.026 #4 | Combined |
| [`H64_H73_H71.diff`](runs/enigma_s5_fulldata/diffs/H64_H73_H71.diff) | H64+H73+H71 | -0.004 #8 | Combined |
| [`H64_H60_H73_H71.diff`](runs/enigma_s5_fulldata/diffs/H64_H60_H73_H71.diff) | Full 4-way | -0.048 #10 | Combined |

### Run artifacts
- `runs/enigma_s5_fulldata/` — **1500-shard validation**: 10 runs, results, logs (8xH100 Hyperbolic)
- `runs/enigma_s5_prod/` — Primary S5: 10 runs, results, logs (3 shards)
- `runs/enigma_s5_prod_singles_r1/` — S5 new singles: 11 mechanisms (3 shards)
- `runs/enigma_s5_prod_composites_r1/` — S5 composites: 11 runs (3 shards)
- `runs/enigma_s5_prod_frontier_r1/` — S5 frontier extension: 12 runs (3 shards)
- `runs/enigma_s4_prod/` — Stage 4: 10 original runs + diffs
- `runs/enigma_s4ext_*` — Stage 4 extension: compounds + 20k H60

---

## Next Steps

1. **Production batch validation** — Run H73 solo + baseline at production batch size (524288 tokens) on 8xH100 with `DistMuonAdamW`. The 1500-shard validation used small batch (1024 tokens). Production gradient noise is different.
2. **Time-to-GPT-2 speedrun** — Integrate H73 into `base_train.py`, run full d24 speedrun targeting GPT-2 CORE score 0.256525 on 8xH100. Current record: 1.65 hours.
3. **Deploy H73 to production** — Modify `adamw_step_fused` in `nanochat/nanochat/optim.py` to add eps scheduling. Simple: `eps = 10 ** (-6 + -4 * step/total_steps)`.
4. **Don't** add H64/H60/H71 to production — compounds don't help H73 at scale. H73 solo is the cleanest win.
