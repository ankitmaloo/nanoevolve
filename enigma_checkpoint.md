# Enigma Checkpoint — 2026-03-16

## Current Frontier

**Best result: H64_H60_H73 = 1.3979 BPB at 20k steps** — beats production baseline (1.4260) by **+0.028 BPB**.

This is: Nesterov blend schedule + Muon beta2 warmup + AdamW eps schedule. Three zero-overhead schedule changes to the production MuonAdamW optimizer.

Close second: **H533_H535 = 1.3994 BPB** — shape-aware Muon beta2 + embedding-only eps schedule. Completely different mechanism set, nearly ties the champion.

### What the winning mutations do

| ID | Mechanism | Where applied | Code |
|----|-----------|--------------|------|
| H64 | Nesterov blend 0.7→0.95 over 500 steps (decoupled from momentum) | Muon kernel patch | `enigma/stage5_patch.py:_patch_muon_h64` |
| H60 | Muon beta2 warmup 0.8→0.95 over 500 steps | Training loop schedule | `group['beta2'] = 0.8 + 0.15 * min(1.0, step / 500)` |
| H73 | AdamW eps log-linear 1e-6→1e-10 over total steps | AdamW kernel patch | `enigma/stage5_patch.py:_patch_adamw_eps_schedule` |

### Stage history at a glance

| Stage | Steps | Data | Eval method | Key outcome |
|-------|-------|------|-------------|-------------|
| 1-3 | 1k-5k | ~3 shards | SpecCandidateOptimizer (broken: no prod schedules) | Trust ratio won — later proven artifact |
| 4 | 5k | ~3 shards | Production MuonAdamW monkey-patched | H64 +0.063, H73 +0.046, H60 +0.031 |
| 4ext | 5k+20k | ~3 shards | Same | H60 confirmed at 20k, H71 weakened |
| 5 | 20k | ~3 shards | Same, + new mutations H531-H538 | H64_H60_H73 = 1.3979 (best) |

---

## Experimental Caveats (IMPORTANT)

**Reduced dataset**: All runs used ~3 data shards out of 6543 total (ClimbMix-400B). Each shard is ~60M tokens. With ~180M total tokens and 20k steps of 1024 tokens = 20M tokens processed, the model sees ~11% of its small dataset. With the recommended 170 shards (~10B tokens), there would be zero data repetition in 20k steps. Results on 3 shards involve data recycling and may not transfer to the full-data regime.

**Small model**: GPT-2 scale (12 layers, 768 dim, 6 heads, ~124M params). Optimizer behavior changes at larger scales.

**Small batch**: 1024 tokens (device_batch=2, seq_len=512). Production uses larger batches with different gradient noise.

**Single seed**: seed=42 only. No multi-seed robustness validation.

**Single GPU**: All on single B200. Production uses `DistMuonAdamW` (distributed).

**Bottom line**: Results are directional signal, not deployment-ready.

---

## What Will and Won't Hold at Full Scale

### Likely to hold

**H73 (eps schedule)** — Strongest case. AdamW has `warmup_ratio=0.0` (no LR warmup) + zero-initialized second moment = enormous effective step sizes on low-variance dimensions early. Eps schedule acts as implicit per-coordinate clip. This problem gets *worse* with more diverse data. Novel dimension — no prior work explores eps scheduling.

**H60 (beta2 warmup)** — Zero-initialized EMA buffers are universally noisy early regardless of data volume. The 20k curve (worse early, better late, crossover at ~11k) is the hallmark of a real phase-aware improvement. More diverse data makes fast early variance adaptation (low beta2) even more valuable.

### Uncertain

**H64 (Nesterov blend schedule)** — On 3 shards, early gradients are less informative (repeated data). Lower blend early helps when gradients are noisy. With full data, early gradients carry *more* signal — benefit of downweighting momentum may shrink. The +0.063 at 5k was suspiciously large; at 20k it only contributed as part of compound. The uninitialized momentum buffer argument still holds but effect size probably smaller.

**H533 (shape-aware beta2)** — Tuned to 768-dim architecture's aspect ratios. Different model sizes have different matrix shapes. Could be fragile.

### Likely won't hold

**H71 (beta1 warmup)** — Already harmful in compounds at 20k. Won't survive full data.

**Exact compound rankings** — H64_H60_H73 vs H533_H535 differ by only 0.0015 BPB. Could flip with different data/seed.

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
- Warmup schedules on every fixed constant (momentum, beta1, beta2, eps)
- Simple monotonic schedules beat non-monotonic ones
- Nesterov blend decoupling (H64)
- AdamW eps scheduling (H73) — novel dimension
- Compound of 3 > compound of 4 (drop H71)

### What doesn't work
- H71 (beta1 warmup) hurts in compounds at 20k
- H60 solo lost at 20k (but helps in compounds)
- Trust ratio — artifact of broken eval loop
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
| `enigma/run_stage5.py` | Production-faithful training loop with compound feature flags |
| `nanochat/nanochat/optim.py` | Production optimizer: `adamw_step_fused` (L20-49), `muon_step_fused` (L90-147), `MuonAdamW` (L152-291) |
| `nanochat/scripts/base_train.py` | Production schedules: `get_lr_multiplier` (L362), `get_muon_momentum` (L374), `get_weight_decay` (L380) |
| `nanochat/nanochat/gpt.py` | `GPT.setup_optimizer` (L356-394) — creates param groups |
| `runs/enigma_s4_prod/diffs/` | Annotated diffs for all Stage 4 mutations |
| `runs/enigma_stage*_postmortem.json` | Postmortems for stages 2, 3, 4, 4ext |

### Run artifacts
- `runs/enigma_s5_prod/` — Primary S5: 10 runs, results, logs
- `runs/enigma_s5_prod_singles_r1/` — S5 new singles: 11 mechanisms
- `runs/enigma_s5_prod_composites_r1/` — S5 composites: 11 runs
- `runs/enigma_s5_prod_frontier_r1/` — S5 frontier extension: 12 runs
- `runs/enigma_s4_prod/` — Stage 4: 10 original runs + diffs
- `runs/enigma_s4ext_*` — Stage 4 extension: compounds + 20k H60

---

## Cluster Access

```bash
# Slurm cluster
ssh user54@35.84.33.219
export TORCHDYNAMO_DISABLE=1
# Shared fs: /mnt/sharefs/user54/nanoe/
# GPUs: B200 (183GB), priority partition

# Sync local → cluster
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  /Users/ankit/Documents/dev/RL/nanoe/nanoevolve/ user54@35.84.33.219:~/nanoe/
```

---

## How to Run Full-Data Validation

```bash
# Download 170 shards (~100GB, ~1hr)
export NANOCHAT_BASE_DIR=/path/to/persistent/storage
cd nanochat && python -m nanochat.dataset -n 170 -w 8

# Setup (works on any Ubuntu+CUDA machine)
./scripts/setup_azure_a100.sh

# Run experiments
python enigma/run_stage5.py --mutation H64_H60_H73 --steps 20000 --depth 12
python enigma/run_stage5.py --mutation none --steps 20000 --depth 12
# Set ENIGMA_TOTAL_STEPS=20000 for correct eps schedule denominator
```

Each 20k-step run takes ~10min on H100 with torch.compile.

---

## Next Steps

1. **Full-data validation** — Run baseline + H73 + H60 + H64_H60_H73 on 170 shards at 20k. If H73 and H60 still show phase-shift signature, the findings are real.
2. **Deploy to production** if validated — modify `muon_step_fused`, add `get_nesterov_blend(it)` and `get_beta2(it)` to `base_train.py`, schedule eps in AdamW groups.
3. **Don't** add H71 to compounds, trust Stage 1-3 results, or use the non-faithful H64 approximation.
