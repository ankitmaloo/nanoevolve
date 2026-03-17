# Stage 6: Beyond H73 — Adaptive, Cross-Optimizer, and Structural

## Premise

Stage 5 proved that H73 (eps schedule 1e-6→1e-10) is the single strongest optimizer mutation, validated on 1500 shards (+0.092 BPB). But H73 is a fixed time-based schedule discovered by trial and error. It doesn't know about batch size, model scale, or actual training dynamics.

Stage 6 explores three directions:
1. **Adaptive eps** — make H73's insight batch/scale-invariant (H601-H605, H607)
2. **Cross-optimizer coordination** — Muon and AdamW share signals (H603, H604, H606, H608)
3. **New attack surfaces** — weight decay, momentum, Muon internals, layer-wise, initialization, loss geometry (H609-H620)

## Overlap and gaps in the hypothesis set

### Cluster A: Adaptive AdamW eps (6 hypotheses)
H601, H602, H603, H605, H607 are all variations on "make eps respond to actual state instead of time." H604 also touches eps. This cluster is deep but narrow — it only generalizes H73. If H73's mechanism is the whole story, one of these wins. If not, this cluster is a dead end at scale.

### Cluster B: Cross-optimizer coordination (4 hypotheses)
H603, H604, H606, H608 share information between Muon and AdamW. This is novel — no prior work does this. But all four use scalar summary statistics (conditioning ratio, noise metric, agreement score, cos angle). Richer signals (per-neuron, per-layer) might work better.

### Cluster C: New attack surfaces (H609-H620)
What Stage 5 never touched: Muon's orthogonalization itself, weight decay dynamics, layer-wise differentiation, gradient preprocessing, initialization beyond warmup, loss geometry. These are orthogonal to Cluster A and could compound with the eps winner.

## Hypotheses

See `hypotheses/` for detailed writeups with pseudocode.

### Cluster A: Adaptive eps (build on H73)

| ID | Name | Mechanism | Cross-domain | Surface |
|----|------|-----------|--------------|---------|
| H601 | Adaptive eps from bias correction | `eps = base_eps / (1 - beta2^step)` | Kalman filter gain | AdamW eps |
| H602 | Eps from second-moment convergence | Track `exp_avg_sq` volatility | Adaptive filters (LMS/RLS) | AdamW eps |
| H603 | Muon variance → AdamW eps | Muon conditioning signal scales eps | Cascade control | AdamW eps + cross |
| H604 | Shared noise-phase detector | Global noise metric → both optimizers | Regime detection (finance) | Both eps + beta2 |
| H605 | Batch-aware eps warmup | Duration ∝ `1/batch_size` | Central limit theorem | AdamW eps |
| H607 | Homeostatic eps | Feedback loop maintains target step size | Homeostatic plasticity | AdamW eps |

### Cluster B: Cross-optimizer coordination

| ID | Name | Mechanism | Cross-domain | Surface |
|----|------|-----------|--------------|---------|
| H606 | Gradient agreement gating | Muon-AdamW curvature agreement → LR gate | Ensemble boosting | LR for both |
| H608 | Orthogonalization angle → embedding confidence | How much Muon rotates gradient → embedding LR | Residual analysis | Embedding LR |

### Cluster C: New attack surfaces

| ID | Name | Mechanism | Cross-domain | Surface |
|----|------|-----------|--------------|---------|
| H609 | Layer-wise LR scaling from gradient norms | Deeper layers get different LR based on gradient magnitude | Thermodynamic depth (physics) | Per-layer LR |
| H610 | Adaptive weight decay from param-grad correlation | WD increases when params and grads are aligned (overfit signal) | Regularization theory | Weight decay |
| H611 | Muon NS iteration count from condition number | Fewer polar express iterations when well-conditioned, more when not | Iterative solvers (numerical LA) | Muon internals |
| H612 | Gradient clipping from loss curvature | Clip threshold adapts to local Hessian-vector product estimate | Trust region methods | Gradient preprocessing |
| H613 | Embedding re-initialization at phase boundaries | Reset embedding momentum at warmdown start | Annealing restarts (SA) | Initialization/reset |
| H614 | Muon momentum from gradient noise ratio | `momentum = signal / (signal + noise)` estimated online | Wiener filter (signal processing) | Muon momentum |
| H615 | Sharpness-aware Muon | Perturb params before computing Muon gradient (SAM-like) | SAM / loss geometry | Muon internals |
| H616 | Cross-layer gradient decorrelation | Penalize correlated updates across layers | Whitening (statistics) | Gradient preprocessing |
| H617 | Adaptive beta2 from effective sample size | `beta2 = 1 - 1/ESS` where ESS tracks gradient diversity | Importance sampling (MC) | Both beta2 |
| H618 | Weight decay warm-start from H73 | Schedule WD inversely to eps — tighter regularization when eps is low | Regularization-optimization duality | Weight decay |
| H619 | Muon variance reset at phase boundaries | Reset `second_momentum_buffer` when LR warmdown begins | Cold restart (optimization) | Muon state |
| H620 | Embedding-matrix gradient coherence loss | Auxiliary loss penalizing gradient direction disagreement | Multi-task gradient alignment | Architecture |

## Set-Level Gap Analysis

What this hypothesis set covers vs what's still unexplored:

### Covered

| Surface | Hypotheses | Depth |
|---------|-----------|-------|
| AdamW eps | H601, H602, H603, H604, H605, H607 | Deep (6 variants) |
| Cross-optimizer signals | H603, H604, H606, H608, H620 | Moderate (5 variants) |
| Muon momentum | H614 | Shallow (1 variant) |
| Weight decay | H610, H618 | Shallow (2 variants) |
| Muon internals (NS iterations, variance) | H611, H615, H619 | Moderate (3 variants) |
| Layer-wise | H609, H616 | Shallow (2 variants) |
| Beta2 (both) | H617 | Shallow (1 variant) |
| Phase boundary | H613, H619 | Shallow (2 variants) |
| Gradient preprocessing | H612, H616 | Shallow (2 variants) |

### NOT covered — remaining gaps

| Gap | Why it matters | Possible direction |
|-----|---------------|-------------------|
| **LR schedule shape** | Production uses linear warmdown. Cosine, exponential, stepped could all differ. Nobody tested LR schedule shape in Enigma. | Cosine warmdown, sqrt warmdown, stepped with plateaus |
| **AdamW beta1 (positive direction)** | H71 (beta1 warmup) was harmful. But beta1 was only tested as a warmup. What about beta1 cooldown? Or layer-wise beta1? | Beta1 that responds to gradient quality per group |
| **Mixed precision interactions** | Production uses bf16 for Muon's polar express, fp32 elsewhere. The precision boundary creates quantization noise that interacts with eps and second moments. | Adaptive precision: use fp32 polar express early when updates are sensitive, switch to bf16 later |
| **Data-aware scheduling** | All schedules are time-based or state-based. None respond to data properties (loss variance across batches, domain shifts within ClimbMix). | Track batch-to-batch loss variance, adjust eps/momentum when data becomes harder/easier |
| **Embedding-specific optimization** | Embeddings are the largest AdamW group (~25M params) but get the same eps, beta1, beta2 as tiny scalar groups. | Separate eps/beta schedules for embeddings vs scalars. Embeddings may need different treatment (they interact with all layers) |
| **Muon cautious mask alternatives** | Muon uses `(g * p) >= 0` for weight decay masking. This binary mask could be replaced with a soft version or different alignment criterion. | Soft cautious mask: `wd * sigmoid(k * g * p)` instead of binary |
| **Multi-seed robustness** | Everything is seed=42. A mutation that wins by +0.01 on one seed might lose on another. | Run top candidates with 3+ seeds before declaring winners |
| **Gradient accumulation effects** | At production batch (524K), grad_accum=4. The accumulation boundary creates micro-batch structure that could interact with momentum/variance estimation. | Adjust beta2 based on grad_accum steps (fewer accum = noisier gradients) |

## Evolutionary Strategy

### Round 1: Screen (fast, 5k steps, small batch)
Run all 20 hypotheses at 5k steps on 1 GPU each. Kill the bottom 50%.

### Round 2: Validate (20k steps, full data)
Run surviving 10 at 20k steps on 1500 shards. Rank against H73 baseline.

### Round 3: Compound
Take top 3-5 survivors from different clusters. Test pairwise and triple compounds.
Key rule from Stage 5: compounds from the same cluster (all eps variants) are unlikely to stack. Cross-cluster compounds (eps + momentum + WD) are more promising.

### Round 4: Production scale
Run top 2-3 at production batch (524K) on 8xH100 with DistMuonAdamW.

### Round 5: Speedrun
Integrate winner into base_train.py, run Time-to-GPT2.
