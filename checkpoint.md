# NanoEvolve Checkpoint

## Current State (2026-03-16)

**Project Enigma** — evolutionary optimizer mutation search against production NanoChat GPT-2 training — has completed 4 stages and found 4 zero-overhead optimizer improvements that beat production.

The best single mutation (H64: Nesterov blend schedule) improves validation BPB by +0.063 over production baseline at 5000 steps, with accelerating gains in the endgame. Compounding and longer-horizon testing are the natural next step.

---

## Enigma: What It Is

Enigma is a systematic hypothesis-test loop for discovering optimizer improvements to the production NanoChat MuonAdamW optimizer. It works by:

1. Generating mutation hypotheses (manually or via LLM subagents)
2. Implementing each as a minimal code patch to the production optimizer
3. Running all mutations + baseline in parallel on the Slurm cluster (B200 GPUs)
4. Scoring by validation BPB delta vs unpatched production baseline
5. Documenting results, negative knowledge, and meta-lessons in postmortems

### Cluster Access

- **Slurm cluster**: `user54@35.84.33.219` (login node)
- **Shared filesystem**: `/mnt/sharefs/user54/nanoe/`
- **Repo on cluster**: `$HOME/nanoe` (rsync'd from local)
- **GPU**: NVIDIA B200 (183 GB VRAM)
- **Partition**: `priority`
- **Critical env var**: `export TORCHDYNAMO_DISABLE=1` (triton compilation fails on this cluster, must disable torch.compile)

### How to Deploy and Run

```bash
# From local machine — sync repo to cluster
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
  /Users/ankit/Documents/dev/RL/nanoe/nanoevolve/ \
  user54@35.84.33.219:~/nanoe/

# On the cluster — submit job
ssh user54@35.84.33.219
cd ~/nanoe/runs/enigma_s4_prod
sbatch slurm_s4.sh

# Monitor
squeue -u user54
# Results appear in runs/enigma_s4_prod/results/<mutation>_real.json
```

---

## Enigma Stage History

### Stage 1-2: DSL-Based Search (Toy + Real GPU)
- **Eval method**: `SpecCandidateOptimizer` with cosine LR only, no production schedules
- **Best finding**: H02 trust ratio [0.5, 1.5] — consistent winner
- **Hardware**: Slurm B200 GPUs, 1000 steps
- **Key files**: `enigma/run_stage2.py`, `runs/enigma_s2_*/`, `runs/enigma_stage2_postmortem.json`

### Stage 3: Code Mutations (Real GPU, 1k + 5k steps)
- **Eval method**: `SpecCandidateOptimizer` with cosine LR, no production schedules
- **Best finding**: H41 momentum warmup 0.85→0.95/200 steps (+0.032 BPB at 5k)
- **Key lesson**: Compounding hurts — H41 solo > H41+H39+H46 compound
- **Key files**: `enigma/run_stage3.py`, `runs/enigma_s3_*/`, `runs/enigma_stage3_postmortem.json`
- **Diff**: `runs/enigma_s3_compound/H41_momentum_warmup.diff`

### Stage 4: Production-Patched (Current Best Results)
- **Eval method**: Production MuonAdamW with ALL production schedules (momentum warmup, WD decay, LR warmdown). Mutations applied as monkey-patches.
- **Critical discovery**: Stages 1-3 eval loop was BROKEN — missing production schedules systematically biased results. Trust ratio (H02) was compensating for missing schedules and actually HURTS on production.
- **Key files**: `enigma/stage4_patch.py`, `runs/enigma_s4_prod/slurm_s4.sh`, `runs/enigma_s4_prod/results/`, `runs/enigma_s4_prod/diffs/`, `runs/enigma_stage4_postmortem.json`

---

## Stage 4 Results (5000 steps, production baseline)

| Rank | ID | Mutation | BPB | Δ vs Prod | Verdict |
|------|----|----------|-----|-----------|---------|
| 1 | H64 | Nesterov blend schedule 0.7→0.95/500 steps | 1.4765 | **+0.063** | WIN |
| 2 | H73 | AdamW eps schedule 1e-6→1e-10 log-linear | 1.4936 | **+0.046** | WIN |
| 3 | H60 | Muon beta2 warmup 0.8→0.95/500 steps | 1.5081 | **+0.031** | WIN |
| 4 | H71 | AdamW beta1 warmup 0.7→0.8/300 steps | 1.5193 | **+0.020** | WIN |
| 5 | H62 | WD cosine decay | 1.5374 | +0.002 | marginal |
| 6 | — | Production baseline (unpatched) | 1.5395 | — | BASE |
| 7 | H70 | Embedding weight_decay=0.01 | 1.5432 | -0.004 | LOSE |
| 8 | H81 | Trust ratio [0.5, 1.5] on Muon | 1.5456 | -0.006 | LOSE |
| 9 | H63 | Momentum overshoot 0.85→0.97→0.95 | 1.5472 | -0.008 | LOSE |
| 10 | H78 | Scalar LR warmup 200 steps | 1.5506 | -0.011 | LOSE |

### What the Top 4 Winners Do

**H64 — Nesterov Blend Schedule** (Muon path, training loop change)
- Production warms momentum 0.85→0.95/300 steps. This is used as both the momentum beta AND the Nesterov interpolation coefficient.
- H64 replaces the Nesterov blend with a separate schedule: 0.7→0.95 over 500 steps.
- Lower initial blend (0.7) gives more weight to raw gradient vs momentum buffer when buffer is uninitialized.
- Starts slower than baseline but crosses over at ~step 3500 and accelerates dramatically.
- Code: 4 lines in training loop. See `runs/enigma_s4_prod/diffs/H64_nesterov_schedule.diff`

**H73 — AdamW Epsilon Schedule** (AdamW path, kernel-level monkey-patch)
- Production uses fixed eps (typically 1e-8). H73 schedules eps log-linearly from 1e-6 to 1e-10.
- Large eps early caps per-coordinate step sizes when variance estimates are noisy.
- Small eps late restores full adaptivity as second moment stabilizes.
- Especially important because production uses warmup_ratio=0.0 (no LR warmup).
- Code: Replace `adamw_step_fused` via monkey-patch. See `runs/enigma_s4_prod/diffs/H73_eps_schedule.diff`

**H60 — Beta2 Warmup** (Muon path, training loop change)
- NorMuon's second moment beta2 is fixed. H60 warms it from 0.8→0.95 over 500 steps.
- Lower beta2 early = shorter EMA window = faster response to changing gradient variance.
- Code: 3 lines in training loop. See `runs/enigma_s4_prod/diffs/H60_beta2_warmup.diff`

**H71 — AdamW Beta1 Warmup** (AdamW path, kernel-level monkey-patch)
- Production warms Muon momentum but leaves AdamW beta1 fixed. H71 applies the same idea.
- Warms beta1 from (beta1-0.1) to beta1 over 300 steps.
- Code: Replace `adamw_step_fused` via monkey-patch. See `runs/enigma_s4_prod/diffs/H71_beta1_warmup.diff`

---

## Accumulated Knowledge

### Positive Knowledge (What Works)
- **PK05**: Nesterov blend decoupling is the highest-leverage optimizer change (+0.063)
- **PK06**: Epsilon scheduling is a completely novel dimension with large payoff (+0.046)
- **PK07**: Warmup schedules work on every fixed constant — momentum, beta1, beta2, eps
- **PK08**: Simple monotonic schedules beat clever non-monotonic ones (H64 linear > H63 overshoot)
- **PK09**: The 4 winners touch 4 independent parameters — high compound potential

### Negative Knowledge (What Doesn't Work)
- **NK17**: Trust ratio LOSES on production (-0.006) — was an artifact of broken eval loop
- **NK18**: Embedding weight decay LOSES (-0.004) — model uses softcap which already regularizes
- **NK19**: Momentum overshoot LOSES (-0.008) — 0.97 peak causes excess smoothing
- **NK20**: Scalar LR warmup is WORST (-0.011) — scalars need immediate adaptation from step 0
- **NK21**: ALL Stage 1-3 results are unreliable for production decisions — eval loop gap

### Meta-Lessons
- **ML08**: Eval loop fidelity is EVERYTHING — production-faithful eval reveals different winners
- **ML09**: Nesterov blend decoupling is the highest-leverage change discovered
- **ML10**: AdamW path has massive untapped potential (ignored in 100% of prior evolution)
- **ML11**: Warmup schedules are the dominant mutation pattern
- **ML12**: Simple monotonic schedules beat clever non-monotonic ones
- **ML13**: Mutations that work on broken baselines may be harmful on correct baselines

---

## Key Code Files

### Enigma Infrastructure
| File | Purpose |
|------|---------|
| `enigma/stage4_patch.py` | Monkey-patch module for production optimizer. Reads `ENIGMA_MUTATION` env var, patches `muon_step_fused` or `adamw_step_fused` |
| `enigma/stage4_context.md` | Shared context doc used to generate Stage 4 hypotheses |
| `enigma/run_stage3.py` | Stage 3 optimizer with code mutations (SpecCandidateOptimizer-based, SUPERSEDED by Stage 4) |
| `enigma/run_stage2.py` | Stage 2 optimizer (SpecCandidateOptimizer-based, SUPERSEDED) |

### Production NanoChat (READ ONLY — do not modify without explicit user approval)
| File | Key Functions |
|------|---------------|
| `nanochat/nanochat/optim.py` | `adamw_step_fused` (L20-49), `muon_step_fused` (L90-147), `MuonAdamW` class (L152-291) |
| `nanochat/scripts/base_train.py` | `get_lr_multiplier` (L362-371), `get_muon_momentum` (L374-377), `get_weight_decay` (L380-381) |
| `nanochat/nanochat/gpt.py` | `GPT.setup_optimizer` (L356-394) — creates param groups |

### Run Artifacts
| Path | Contents |
|------|----------|
| `runs/enigma_s4_prod/slurm_s4.sh` | Stage 4 slurm script (10 mutations, inline Python training loop) |
| `runs/enigma_s4_prod/results/*.json` | Raw results with learning curves for all 10 runs |
| `runs/enigma_s4_prod/diffs/*.diff` | Annotated diffs for all 9 mutations |
| `runs/enigma_stage4_postmortem.json` | Full Stage 4 analysis, curves, mutation analysis, knowledge |
| `runs/enigma_stage3_postmortem.json` | Stage 3 analysis |
| `runs/enigma_stage2_postmortem.json` | Stage 2 analysis |

### adamopt Infrastructure (DSL-based search — not used in Stage 4)
| File | Purpose |
|------|---------|
| `adamopt/optim_search/spec.py` | Bounded optimizer DSL |
| `adamopt/optim_search/candidate_optimizer.py` | Spec-driven optimizer runtime |
| `adamopt/optim_search/mutations.py` | 13 composable DSL mutation operators |
| `adamopt/optim_search/eval_candidate.py` | Evaluation harness (toy backend) |
| `adamopt/optim_search/tournament.py` | Generation loop, promotion |

---

## What Is Pending / Next Steps

### Immediate: Stage 5 — Compound Testing

**Priority 1: Test H64+H73 compound** (highest confidence)
- H64 modifies Muon momentum/Nesterov blend (training loop)
- H73 modifies AdamW epsilon (kernel-level patch)
- Independent paths (Muon vs AdamW), different parameter types
- Expected: additive, ~+0.10 BPB combined
- Implementation: combine the Stage 4 patches — use H64's training loop schedule AND H73's monkey-patched adamw_step_fused

**Priority 2: Test H64+H60 compound**
- Both Muon schedule changes but different parameters (momentum blend vs beta2)
- Moderate risk of interaction

**Priority 3: Test all 4 winners (H64+H73+H60+H71)**
- All 4 touch different optimizer parameters
- Low interaction risk but compounding lesson from Stage 3 warrants caution
- If this works, it's the full production deployment candidate

**Priority 4: Test at 10k+ steps**
- H64's learning curve shows accelerating advantage in endgame
- 10k steps would confirm the trend continues (or if it plateaus)
- Can be combined with compound testing

### How to Implement Stage 5

1. **Create `runs/enigma_s5_compound/slurm_s5.sh`** based on Stage 4's slurm script
2. Mutation array should include:
   - `none` (baseline)
   - `H64_solo` (control — should reproduce Stage 4)
   - `H73_solo` (control — should reproduce Stage 4)
   - `H64_H73` (compound: Nesterov blend + eps schedule)
   - `H64_H60` (compound: Nesterov blend + beta2 warmup)
   - `H64_H73_H60_H71` (all 4 winners)
3. For compound mutations:
   - H64 is a training loop schedule override (set `group['momentum']` per step)
   - H73 is a kernel monkey-patch (`stage4_patch.py` replaces `adamw_step_fused`)
   - H60 is a training loop schedule override (set `group['beta2']` per step)
   - H71 is a kernel monkey-patch (replaces `adamw_step_fused` — **CONFLICTS with H73**)
   - H73 and H71 both replace `adamw_step_fused` — need a COMBINED kernel that does BOTH eps schedule AND beta1 warmup
4. Steps: 5000 minimum, consider 10000 for the best compound
5. `--array=0-5` (6 runs)

### Kernel Conflict Resolution for H73+H71

Both H73 and H71 monkey-patch `adamw_step_fused`. To compound them, you need a single replacement kernel:

```python
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_combined(p, grad, exp_avg, exp_avg_sq, step_t, lr_t,
                        beta1_t, beta2_t, eps_t, wd_t):
    # H71: beta1 warmup
    warmup = 300.0
    frac_b1 = (step_t / warmup).clamp(max=1.0)
    effective_beta1 = (beta1_t - 0.1) + 0.1 * frac_b1
    # H73: eps schedule
    log_ratio = -4.0
    frac_eps = (step_t / 5000.0).clamp(max=1.0)
    log_eps = -6.0 + log_ratio * frac_eps
    eps = 10.0 ** log_eps
    # Standard AdamW with both modifications
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - effective_beta1)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - effective_beta1 ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)
```

### Longer-Term: Production Deployment

If compound testing succeeds:
1. The Muon changes (H64, H60) are training loop schedule additions — add `get_nesterov_blend(it)` and `get_beta2(it)` functions to `base_train.py`
2. The AdamW changes (H73, H71) require modifying `adamw_step_fused` in `optim.py` — either directly or via a new schedule parameter
3. All changes are zero-overhead (no extra computation, only different hyperparameter values)
4. Consider making Nesterov blend a first-class tunable in MuonAdamW param groups

### Do NOT Do
- Do NOT deploy trust ratio (H02/H81) to production — it's harmful with correct schedules
- Do NOT use SpecCandidateOptimizer for production decisions — eval loop gap makes results unreliable
- Do NOT compound mutations that modify the same function without a combined kernel

---

## Repository Structure

```
nanoevolve/
  adamopt/       # Optimizer search control plane (DSL, mutations, scoring, tournament)
  nanochat/      # Real GPT training substrate (model, optimizer, training loop)
  alphaevolve/   # Prior evolutionary code and reference material
  enigma/        # Enigma mutation search (patches, runners, context docs)
  runs/          # All experiment artifacts (slurm scripts, results, postmortems, diffs)
```

## Development

```bash
source .venv/bin/activate
pip install -e adamopt/
pip install -e nanochat/
python -m pytest adamopt/tests -q  # Expected: 18 passed
```

Python 3.10+. PyTorch >= 2.0.
