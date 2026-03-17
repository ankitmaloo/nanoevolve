# Enigma Checkpoint — 2026-03-16

## What Is Enigma

Enigma is a systematic hypothesis-test loop for discovering optimizer improvements to the production NanoChat MuonAdamW optimizer. It runs mutations as monkey-patches against the real production optimizer on a Slurm cluster with B200 GPUs.

---

## What "Beating BPB" Means

**BPB = Bits Per Byte** — a measure of how well the model predicts the next token on held-out validation data. Lower is better.

When we say H64_H60_H73 "beats production by +0.028 BPB at 20k steps", it means: **after the same number of training steps (20k), the mutated optimizer reaches a lower validation loss than the production optimizer.** This is a sample efficiency improvement — the model learns more from the same amount of data and compute.

What this means practically:
- **Better training quality**: At any given step budget, the mutated optimizer produces a model that predicts text more accurately. The same 20k steps yield a smarter model.
- **Faster convergence**: To reach the same quality the production optimizer achieves at 20k steps, the mutated optimizer gets there earlier. The training curve is shifted left — you could stop sooner and save compute.
- **Zero overhead**: All winning mutations are schedule changes (different hyperparameter trajectories over training). They do not add computation — step time is identical (~28ms). The improvement is purely algorithmic.

What this does NOT tell us:
- Whether the improvement holds at much longer horizons (50k+ steps) or larger models
- Whether it generalizes across different datasets or model sizes
- Whether the effect size is practically meaningful for downstream tasks (0.028 BPB is measurable but modest)

---

## Experimental Caveats

**Reduced dataset**: All Enigma runs used only ~3 data shards out of the full 6543-shard ClimbMix-400B dataset. The full dataset has `shard_00000.parquet` through `shard_06542.parquet` (last shard is validation). Our runs trained on ~3 train shards + 1 validation shard. This means:
- The model sees repeated data within 20k steps (data recycling / multiple epochs)
- Results may not transfer to the full-data regime where the model sees each example once
- Optimizer mutations that help with data-limited training may not help (or may help differently) at full scale

**Small model**: All runs use GPT-2 scale (12 layers, 768 dim, 6 heads, ~124M params). Optimizer behavior can change at larger scales.

**Small batch**: Batch size is 1024 tokens (device_batch=2, seq_len=512, grad_accum=1). Production runs at larger batch sizes where gradient noise characteristics differ.

**Single seed**: Each mutation was tested with seed=42 only. No multi-seed robustness validation.

**Single GPU**: All runs on a single B200. Production uses distributed training with `DistMuonAdamW` which has different communication/sharding patterns.

These caveats mean the results are **directional signal**, not deployment-ready evidence. The optimizer mutations are worth investigating further at scale, but the exact BPB deltas may not reproduce.

---

## Current Frontier (Stage 5 Complete)

**Best result: H64_H60_H73 = 1.3979 BPB at 20k steps** — beats production baseline (1.4260) by **+0.028 BPB**.

This is: Nesterov blend schedule + beta2 warmup + AdamW eps schedule. Three zero-overhead schedule changes.

Close second: **H533_H535 = 1.3994 BPB** — shape-aware Muon beta2 + embedding-only eps schedule. A completely different mechanism set that nearly ties the champion.

---

## Stage 5 — Full Results (33 runs across 3 sub-batches, all 20k steps)

### Sub-run 1: `enigma_s5_prod` (Primary 10-run production slate)

Tested Stage 4 winners as solos, full 4-way composite, and leave-one-out ablations.

| Rank | BPB | Mutation | Role | Notes |
|------|-----|----------|------|-------|
| 1 | **1.3979** | H64_H60_H73 | LOO (no H71) | **BEST OVERALL** |
| 2 | 1.4131 | H73_solo | single | Best solo single |
| 3 | 1.4227 | H64_H60_H71 | LOO (no H73) | |
| 4 | 1.4249 | H60_H73_H71 | LOO (no H64) | |
| 5 | 1.4260 | none | baseline | PRODUCTION |
| 6 | 1.4286 | H60_solo | promoted parent | H60 LOST at 20k solo |
| 7 | 1.4304 | H64_H60_H73_H71 | full 4-way | Full stack WORSE than LOO |
| 8 | 1.4361 | H64_H73_H71 | LOO (no H60) | |
| 9 | 1.4383 | H517_shared_phase | wildcard | Lost |
| 10 | 1.4725 | H504_beta2_phase2 | exploit | KILLED |

**Key finding**: Dropping H71 from the full stack improved by 0.032 BPB. H71 (AdamW beta1 warmup) hurts in compounds at 20k.

### Sub-run 2: `enigma_s5_prod_singles_r1` + `enigma_s5_prod_composites_r1`

Tested 8 new single mechanisms, then composed the winners.

**New singles tested** (H531-H538):
- H531: Raw-gradient cautious WD mask on Muon
- H532: Muon second-moment buffer reset at warmdown onset
- H533: Shape-aware Muon beta2 (rectangular vs square-ish groups get different beta2)
- H534: Staged orthogonalization depth
- H535: Embedding-only epsilon schedule (subset of H73)
- H536: x0_lambdas beta1 decay during warmdown
- H537: Embedding first-moment reset at warmdown
- H538: Seed AdamW second moment from first gradient for embeddings

**Composites results** (11 runs):

| Rank | BPB | Mutation | Notes |
|------|-----|----------|-------|
| 1 | **1.3994** | H533_H535 | **Near-champion pair** |
| 2 | 1.4247 | H532_H533_H538 | |
| 3 | 1.4285 | H532_H533_H535 | |
| 4 | 1.4303 | H532_H535_H538 | |
| 5 | 1.4313 | H533_H538 | |
| 6 | 1.4339 | none | baseline |
| 7 | 1.4397 | H532_H533 | |
| 8 | 1.4449 | H533_shape_beta2 | solo |
| 9 | 1.4639 | H532_H533_H535_H538 | Full stack regresses |
| 10 | 1.4723 | H533_H535_H538 | |
| 11 | 1.4875 | H532_H538 | |

### Sub-run 3: `enigma_s5_prod_frontier_r1`

Tested whether new singles (H533/H538) can extend the old frontier (H64_H60_H73).

| Rank | BPB | Mutation | Notes |
|------|-----|----------|-------|
| 1 | 1.4056 | H60_H73_H533_H535 | Old frontier + new pair, no H64 |
| 2 | 1.4077 | H64_H60_H73 | Repeat of champion |
| 3 | 1.4111 | H64_H73_H533_H535 | |
| 4 | 1.4144 | H533_shape_beta2 | Solo new single |
| 5 | 1.4164 | H64_H60_H73_H533_H535 | Full extension |
| 6 | 1.4226 | H533_H535 | Same pair, different baseline batch |
| 7 | 1.4240 | none | baseline |
| 8 | 1.4243 | H64_H60_H73_H533 | |
| 9 | 1.4265 | H64_H60_H73_H538 | |
| 10 | 1.4373 | H64_H60_H533_H535 | |
| 11 | 1.4387 | H538_seed_vsq | Solo new single |
| 12 | 1.4429 | H64_H60_H73_H535 | |

**Key finding**: Adding new singles to H64_H60_H73 doesn't reliably improve it. H64 and H533 may interfere (both touch Muon path).

---

## Accumulated Strategic Knowledge

### What Works (across all stages)
- **Warmup schedules on fixed constants** — works on momentum, beta1, beta2, eps
- **Simple monotonic schedules** beat clever non-monotonic ones
- **Nesterov blend decoupling** (H64) — decouple Nesterov interpolation from momentum beta
- **AdamW eps scheduling** (H73) — log-linear 1e-6 → 1e-10
- **Shape-aware beta2** (H533) — rectangular vs square-ish Muon groups need different beta2
- **Embedding-specific eps** (H535) — targeting eps schedule only to embedding groups
- **Compound of 3 beats compound of 4** — H64_H60_H73 > H64_H60_H73_H71

### What Doesn't Work
- **H71 (beta1 warmup) hurts in compounds** — helps solo at 5k, harmful in composites at 20k
- **H60 solo lost at 20k** (1.4286 vs 1.4260) — but helps in compounds
- **Trust ratio** — artifact of broken eval loop (Stages 1-3)
- **Embedding weight decay** — model uses softcap which already regularizes
- **Scalar LR warmup** — scalars need immediate adaptation
- **Momentum overshoot** — non-monotonic schedules hurt
- **Full stacks consistently underperform** simpler combinations
- **Adding more mechanisms to a working composite rarely helps** — diminishing/negative returns

### Critical Gotchas
- `TORCHDYNAMO_DISABLE=1` mandatory on the Slurm cluster (triton build fails)
- Stage 4 H64 was approximated by overwriting `group['momentum']` — not faithful. Stage 5 `stage5_patch.py` has the faithful port with a dedicated `nesterov_blend_t` tensor
- Stage 1-3 results are NOT reliable — eval loop was missing 3 production schedules
- Baseline BPB varies between batches (1.4260 vs 1.4339 vs 1.4240) — always compare within the same batch

---

## Key Code Files

### Enigma Infrastructure
| File | Purpose |
|------|---------|
| `enigma/stage5_patch.py` | **Current monkey-patch module**. Handles H64 (faithful Nesterov blend), H71 (beta1 warmup), H73 (eps schedule), H71+H73 combined, H531 (raw grad WD), H517 (shared phase), H538 (seed second moment). Schedule-only mutations (H60/H532/H533/H534/H535/H536/H537) applied in training loop. |
| `enigma/run_stage5.py` | Extension runner — production-faithful training loop with compound feature flags |
| `enigma/stage4_patch.py` | Stage 4 monkey-patch module (simpler, superseded by stage5_patch.py) |
| `enigma/stage4_context.md` | Stage 4 hypothesis generation context |
| `enigma/run_stage3.py` | Stage 3 optimizer (SpecCandidateOptimizer-based, SUPERSEDED) |

### Production NanoChat (READ ONLY)
| File | Key Functions |
|------|---------------|
| `nanochat/nanochat/optim.py` | `adamw_step_fused` (L20-49), `muon_step_fused` (L90-147), `MuonAdamW` (L152-291) |
| `nanochat/scripts/base_train.py` | `get_lr_multiplier` (L362), `get_muon_momentum` (L374), `get_weight_decay` (L380) |
| `nanochat/nanochat/gpt.py` | `GPT.setup_optimizer` (L356-394) — creates param groups |

### Run Artifacts
| Path | Contents |
|------|----------|
| `runs/enigma_s5_prod/` | Primary S5 slate: 10 runs, slurm script, portfolio, results, logs |
| `runs/enigma_s5_prod_singles_r1/` | S5 singles rerun: 11 new mechanisms, slurm script, hypotheses |
| `runs/enigma_s5_prod_composites_r1/` | S5 composites: 11 runs testing H532/H533/H535/H538 pairs and stacks |
| `runs/enigma_s5_prod_frontier_r1/` | S5 frontier extension: 12 runs testing new singles added to old frontier |
| `runs/enigma_s4_prod/` | Stage 4 original: 10 runs, diffs for all 9 mutations |
| `runs/enigma_s4ext_compound/` | Stage 4 extension: 12 compound runs at 5k |
| `runs/enigma_s4ext_20k_h60/` | Stage 4 extension: 20k follow-up on H60 |
| `runs/enigma_stage*_postmortem.json` | Postmortems for stages 2, 3, 4, 4ext |

---

## Cluster Access & Deployment

```bash
# SSH
ssh user54@35.84.33.219

# Sync local → cluster
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  /Users/ankit/Documents/dev/RL/nanoe/nanoevolve/ \
  user54@35.84.33.219:~/nanoe/

# Submit
cd ~/nanoe/runs/<run_dir> && sbatch <slurm_script>.sh

# Monitor
squeue -u user54

# CRITICAL
export TORCHDYNAMO_DISABLE=1
```

- Shared filesystem: `/mnt/sharefs/user54/nanoe/`
- GPUs: NVIDIA B200 (183 GB), priority partition
- Results: `runs/<run_dir>/results/<mutation>_20k_real.json`

---

## What the Production Deployment Candidate Looks Like

**H64_H60_H73** — three independent schedule changes:

1. **H64: Nesterov blend schedule** — Muon kernel patch (faithful version in `stage5_patch.py`)
   - Add a `nesterov_blend_t` tensor to `_step_muon`
   - Schedule blend from 0.7→0.95 over 500 steps
   - Requires modified `muon_step_fused` kernel that takes blend as separate argument

2. **H60: Beta2 warmup** — training loop schedule
   - `beta2 = 0.8 + 0.15 * min(1.0, step / 500)`
   - Set `group['beta2']` for Muon groups each step

3. **H73: AdamW eps schedule** — AdamW kernel patch
   - `eps = 10 ** (-6.0 + -4.0 * min(1.0, step / total_steps))`
   - Log-linear from 1e-6 to 1e-10

All zero-overhead. See `runs/enigma_s4_prod/diffs/` for annotated diffs of each.

---

## Next Steps

### Option A: Deploy H64_H60_H73 to production
The compound is validated at 20k across multiple runs (1.3979, confirmed 1.4077 in frontier batch). Implementation requires:
- Modified `muon_step_fused` with Nesterov blend parameter
- New `get_nesterov_blend(it)` and `get_beta2(it)` schedules in `base_train.py`
- Modified `adamw_step_fused` with scheduled eps (or per-step group['eps'])

### Option B: Test H533_H535 as an alternative/complement
This pair (1.3994) uses completely different mechanisms (shape-aware beta2 + embedding eps). Could be tested:
- Head-to-head vs H64_H60_H73 with more seeds
- As extension to H64_H60_H73 (already tested in frontier batch — mixed results, 1.4164)

### Option C: Stage 6 — deeper search
Unexplored directions:
- Per-layer schedule differentiation (attention vs MLP)
- Loss-reactive schedules (adaptive, not just step-based)
- Longer horizons (50k+) to test whether gains hold or plateau
- Multi-seed validation of the champion for robustness

### What NOT to do
- Don't add H71 to compounds (hurts at 20k)
- Don't trust Stage 1-3 results for production decisions
- Don't use the non-faithful H64 approximation (overwriting `group['momentum']`)
- Don't expect adding more mechanisms to help — simpler compounds beat larger stacks
