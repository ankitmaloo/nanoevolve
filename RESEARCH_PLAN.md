# NanoEvolve: Evolving State-Dependent Optimizers for Neural Network Training

> **Discovering training-phase-aware optimizer policies through evolutionary search.**

---

## Executive Summary

Modern deep learning optimizers use hand-designed, fixed schedules — learning rate warmup, cosine decay, beta annealing — applied uniformly regardless of what is actually happening during training. These schedules work, but they are static: they cannot *react* to the training state itself.

This project aims to **discover optimizer policies that sense where training is and adapt their behavior accordingly**, using evolutionary search over a bounded optimizer DSL. The key insight is to not evolve an entirely novel optimizer from scratch, but evolve **a gated optimizer whose behavior shifts based on measurable training-state proxies**. A richer, learned version of what practitioners already do crudely with hand-tuned schedules.

The substrate for this search is [NanoChat](https://github.com/karpathy/nanochat) — a real, competitive GPT training codebase. Any optimizer policy that wins on NanoChat is a policy that wins on real transformer training. NanoChat is the proving ground to stay close to real world use and a reasonable basis that discoveries generalize. 

---

## Precedent: AlphaEvolve Already Works For This Kind of Problem

Evolutionary search can discover better algorithmic policies. It has already been demonstrated. Google confirmed [AlphaEvolve](https://arxiv.org/abs/2506.13131) improved their pretraining by enhancing Gemini's kernel engineering. That was gemini-2-flash and gemini-2.0-pro. With better models as mutators, we can definitely go further. 

Li, Schultz, Hennes & Lanctot (2026) applied AlphaEvolve to multi-agent learning algorithms and discovered two new state-of-the-art variants ([arXiv:2602.16928](https://arxiv.org/abs/2602.16928)):

### VAD-CFR: Volatility-Adaptive Discounted Counterfactual Regret Minimization

Standard CFR variants use fixed discount factors for cumulative regrets. Evolution discovered that making these parameters **reactive to training dynamics** — specifically, tracking the volatility of regret updates via an EWMA and increasing discounting when the strategy is in flux — outperforms every hand-designed baseline including DCFR, PCFR+, and DPCFR+.

The discovered mechanisms were explicitly non-intuitive:
- **Volatility-adaptive discounting**: dynamically forgets unstable history faster when learning is noisy, retains more when it stabilizes
- **Asymmetric instantaneous boosting**: positive regrets (currently good actions) are amplified by 1.1×, enabling immediate exploitation without accumulation lag
- **Hard warm-start with regret-magnitude weighting**: delays policy averaging until iteration 500 to prevent early noise from polluting the solution, then weights accumulated policies by regret magnitude — a filter the authors note was "non-intuitive to human designers".

### SHOR-PSRO: Smoothed Hybrid Optimistic Regret PSRO

For population-based training, evolution discovered a meta-solver that **dynamically anneals its blending factor** between regret-based stability and greedy exploitation — shifting from λ=0.3 (more exploitation) to λ=0.05 (more stability) over training. It also discovered a decaying diversity bonus (0.05→0.001) that ensures early population expansion followed by late-stage equilibrium refinement.

This is precisely the kind of **training-phase-aware behavior shift** that this project targets for neural network optimizers.

### Why This Matters For Us

The structural parallel is direct:

| Li et al. 2026 | This project |
|---|---|
| Fixed CFR discount factors → evolved volatility-reactive discounting | Fixed LR/beta schedules → evolved state-reactive optimizer gating |
| Static PSRO meta-solvers → evolved dynamic annealing schedules | Static optimizer configs → evolved aggressive-to-conservative actuators |
| Sensors: regret volatility EWMA, iteration count | Sensors: loss EMA, grad norm EMA, alignment EMA, step fraction |
| Mechanism: smooth blending with annealed coefficients | Mechanism: sigmoid gate interpolating aggressive/conservative poles |
| Outcome: non-intuitive rules that outperform hand-designed baselines | Target: non-intuitive optimizer policies that outperform hand-designed schedules |

Evolution already discovered state-adaptive algorithmic behavior for regret minimization and population-based training. We are applying the same approach to the same class of problem — reactive policy discovery — in the domain of neural network optimization.

---

## Motivation: Why Hand-Designed Schedules Are Insufficient

Training a neural network has at least two distinct regimes:

| Regime | Characteristic | What the optimizer should do |
|---|---|---|
| **Early training** | High loss, noisy gradients, rapidly shifting loss surface | Aggressive exploration, noise-tolerant updates, high effective step size |
| **Late training** | Low loss, stable gradients, fine curvature structure | Conservative updates, tighter clipping, higher trust in second-moment statistics |

Every modern optimizer already acknowledges this implicitly:

- **LR decay** reduces aggressiveness over time
- **Warmup** protects early steps from instability
- **Beta schedules** change momentum over training
- **Gradient clipping** prevents catastrophic updates
- **Weight decay interaction** shifts with training stage

But these are all **time-based schedules**, not **state-based reactions**. They operate on a fixed calendar, not on what the optimizer is actually observing. If training stalls, the schedule doesn't know. If loss improves faster than expected, the schedule doesn't accelerate. If gradient noise drops because the model has entered a low-loss basin, the schedule doesn't stiffen.

The hypothesis is:

> **An optimizer that can sense and react to training dynamics will outperform one that follows a fixed schedule** — and evolution can discover the right reactive policy.

---

## The Core Idea: Gated Optimizer Behavior

Rather than letting evolution generate arbitrary code (which is fragile and hard to evaluate), we constrain the search to a **bounded optimizer DSL** where the evolvable surface is a set of smooth gates conditioned on training-state signals.

### Training-State Sensors

The optimizer has access to six slow-moving state variables, each maintained as an exponential moving average:

| Sensor | What it captures | Why it matters |
|---|---|---|
| `loss_ema` | Smoothed current loss | Where are we in training? |
| `loss_improvement_ema` | Rate of loss decrease | Are we still making progress? |
| `grad_norm_ema` | Average gradient magnitude | Is the signal strong or fading? |
| `update_ratio_ema` | Update magnitude relative to parameter magnitude | Are we making big moves or small ones? |
| `grad_alignment_ema` | Cosine similarity between consecutive gradient means | Are gradients consistent or noisy? |
| `step_fraction` | Progress through total training budget | Simple temporal context |

### The Gating Mechanism

These sensors feed into a smooth gate:

$$g_t = \sigma\!\left(\text{sharpness} \cdot \left(\text{bias} + \sum_{k} w_k \cdot \text{sensor}_k\right)\right)$$

where $\sigma$ is the sigmoid function, and the weights $w_k$, bias, and sharpness are all evolvable.

The gate output $g_t \in (0, 1)$ then interpolates between two behavior poles:

$$\text{actuator}_t = g_t \cdot \text{value}_{\text{aggressive}} + (1 - g_t) \cdot \text{value}_{\text{conservative}}$$

### Actuated Dimensions

The gate controls five optimizer behaviors simultaneously:

| Actuator | Aggressive pole | Conservative pole | Effect |
|---|---|---|---|
| **Update multiplier** | Large (e.g., 1.15×) | Small (e.g., 0.85×) | Controls effective step size |
| **Trust ratio mix** | Full layerwise trust (1.0) | No trust correction (0.0) | Adapts per-layer scale |
| **Clip threshold** | Loose (e.g., 2.0) | Tight (e.g., 0.5) | Bounds maximum update magnitude |
| **Second-moment β₂** | Low (e.g., 0.90) | High (e.g., 0.985) | Fast vs. stable variance estimation |
| **Orthogonal mix** | Full orthogonalization (1.0) | Reduced (e.g., 0.35) | Controls Newton-Schulz polar projection intensity |

This means evolution doesn't need to invent annealing behavior from nothing. It only needs to discover that:
- certain sensor correlations predict "we're in the finicky regime"
- and then shift the actuators from aggressive toward conservative

That is a very natural search surface for evolutionary optimization.

---

## What Evolution Would Discover

The project expects evolution to discover something like:

> **"When loss improvement slows (loss_improvement_ema drops), gradient alignment increases (grads become consistent), and step fraction grows (late in training): reduce update magnitude, tighten clipping, strengthen second-moment adaptation, and soften orthogonalization."**

In the DSL, this might manifest as gate weights like:

```
loss_improvement_ema: -0.8  (gate closes as improvement drops)
grad_alignment_ema:    0.7  (gate opens when grads are consistent ← aggressive)
step_fraction:        -1.0  (gate closes as training progresses)
grad_norm_ema:         0.5  (gate opens when gradients are strong)
bias:                 -0.2  (slightly conservative default)
```

This is **state-dependent annealing** — not a fixed schedule, but a learned reaction function. It would be more adaptive than any hand-designed LR schedule because it reacts to the actual loss landscape, not just a clock.

### More Plausible Than Universal Optimizer Discovery

This framing is deliberately modest in the right way:

- We are **not** trying to replace Adam or Muon with something fundamentally different
- We **are** trying to discover the best *policy for when and how aggressively* to apply known optimizer techniques
- The base optimizer math (momentum, orthogonalization, trust ratios, clipping) stays fixed
- Only the **regime-dependent blending** is evolved

This is much more likely to produce transferable results than attempting to evolve entirely novel optimizer algebra.

---

## Architecture: How It All Fits Together

```
┌──────────────────────────────────────────────────────┐
│                   AdamOpt (Control Plane)             │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Spec DSL │→ │ Mutation │→ │ Candidate Specs  │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
│       ↑                             │                │
│       │                             ↓                │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Archive  │← │ Scoring  │← │ Evaluation       │   │
│  │ & Lineage│  │ & Ranking│  │ Harness          │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────────┐
│               NanoChat (Execution Plane)              │
│                                                      │
│  Real GPT model · Real training data · Real loss     │
│  Matrix params → Candidate optimizer (evolved DSL)   │
│  Non-matrix params → Fixed AdamW                     │
└──────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Constrained search space**: The DSL has hard bounds on all evolvable parameters. No gate coefficient exceeds ±8.0. All actuator ranges are bounded. This prevents pathological candidates and makes evaluation cheap

2. **Matrix-only evolution**: Only the transformer's matrix parameters (attention, MLP projections) use the evolved optimizer. Embeddings, layer norms, and scalar parameters stay on standard AdamW. This mirrors the NanoChat/Muon split and focuses evolution where the leverage is highest

3. **Staged evaluation**: Candidates earn progressively more expensive evaluation. Short-horizon screening kills bad ideas cheaply; only survivors get full-length runs

4. **Composable mutations**: Each mutation operator changes exactly one aspect of the spec (toggle trust ratio, adjust gate weight, scale actuator range, etc.), enabling clean lineage tracking and interpretable diffs between generations

---

## Implementation Status

### ✅ Completed

| Component | Key files |
|---|---|
| Bounded optimizer DSL | [spec.py](adamopt/optim_search/spec.py) |
| Stateful control (6 sensors, smooth gate, 5 actuators) | [spec.py](adamopt/optim_search/spec.py), [candidate_optimizer.py](adamopt/optim_search/candidate_optimizer.py) |
| Spec-driven candidate optimizer runtime | [candidate_optimizer.py](adamopt/optim_search/candidate_optimizer.py) |
| NanoChat-style parameter split (matrix vs non-matrix) | [candidate_optimizer.py](adamopt/optim_search/candidate_optimizer.py) |
| Mutation system (13 operators) | [mutations.py](adamopt/optim_search/mutations.py) |
| Evaluation harness (toy backend) | [eval_candidate.py](adamopt/optim_search/eval_candidate.py) |
| Composite scoring with win hierarchy | [score.py](adamopt/optim_search/score.py) |
| Tournament loop and archive persistence | [tournament.py](adamopt/optim_search/tournament.py), [archive.py](adamopt/optim_search/archive.py) |
| Code mutation, validation, and deployment infra | [command_mutator.py](adamopt/optim_search/command_mutator.py), [validation.py](adamopt/optim_search/validation.py), [deployment.py](adamopt/optim_search/deployment.py) |
| Autonomous async patch/deploy/poll loop | [autonomous.py](adamopt/optim_search/autonomous.py) |
| CLI entrypoint | [search_optimizer.py](adamopt/scripts/search_optimizer.py) |
| 18 passing tests | [adamopt/tests/](adamopt/tests/) |

### 🔲 Remaining (Compute-Dependent)

| Component | What's needed |
|---|---|
| **Real NanoChat evaluator** | Replace toy backend with real NanoChat training runs |
| **Longer-horizon fitness** | Extend evaluation to reward annealing behavior |
| **Multi-seed confirmation** | Verify winners survive across random seeds |

All remaining work is integration and execution. The search infrastructure is built.

---

## How Much Compute Is Needed

This is the concrete question. Here is the answer.

### The Unit of Work: One NanoChat Training Run

One full NanoChat training run on a single machine takes **4–6 hours**. That is the atomic cost unit. Every number below is a multiple of that.

### The Search Process

Evolution works in generations. Each generation:
1. **Mutate**: Take surviving parent specs, produce children (cheap, instant, no GPU)
2. **Screen**: Run each child on a short NanoChat training run to get loss/stability metrics
3. **Select**: Kill bad candidates, promote survivors
4. **Confirm**: Rerun survivors with different seeds to verify they're not flukes

The screening runs don't need to be full 4–6 hour runs. Short-horizon screening at **~20% of a full run** (roughly 1 hour per candidate) is sufficient to eliminate clearly bad ideas. Only the final survivors get full-length evaluation.

### The Math

| Parameter | Conservative | Aggressive |
|---|---|---|
| Generations | 5 | 10 |
| Candidates per generation | 8 | 16 |
| Screening run length | ~1 hr (short horizon) | ~1 hr (short horizon) |
| Survivors promoted per generation | 2–3 | 3–4 |
| Confirmation seeds per survivor | 3 | 3 |
| Confirmation run length | 4–6 hrs (full run) | 4–6 hrs (full run) |

**Screening runs** (cheap, many):

| | Conservative | Aggressive |
|---|---|---|
| Total screening runs | 5 × 8 = 40 | 10 × 16 = 160 |
| Hours per run | ~1 hr | ~1 hr |
| **Screening total** | **~40 machine-hours** | **~160 machine-hours** |

**Confirmation runs** (expensive, few):

| | Conservative | Aggressive |
|---|---|---|
| Survivors needing confirmation | 5 × 3 = 15 | 10 × 4 = 40 |
| Seeds per survivor | 3 | 3 |
| Total confirmation runs | 45 | 120 |
| Hours per run | 5 hrs avg | 5 hrs avg |
| **Confirmation total** | **~225 machine-hours** | **~600 machine-hours** |

**Baseline runs** (one-time calibration):

| | |
|---|---|
| Baseline full runs (3 seeds) | 3 × 5 hrs = 15 machine-hours |
| Evaluator validation / debugging | ~20 machine-hours |
| **Baseline total** | **~35 machine-hours** |

### Total Compute Budget

| Scenario | Screening | Confirmation | Baseline | **Total** |
|---|---|---|---|---|
| **Conservative** (5 gens, pop 8) | 40 hrs | 225 hrs | 35 hrs | **~300 machine-hours** |
| **Aggressive** (10 gens, pop 16) | 160 hrs | 600 hrs | 35 hrs | **~800 machine-hours** |

**The answer: 300–800 machine-hours on a single-GPU cluster, running one NanoChat job at a time.**

With parallelism (e.g., 4 machines running candidates simultaneously), the wall-clock time compresses proportionally. With 4 machines, the conservative scenario finishes in ~75 hours of wall-clock. The aggressive scenario finishes in ~200 hours of wall-clock.

### Why This Is Bounded

The compute is bounded because the search is bounded:

- **Fixed population sizes**: Not open-ended generation; each generation has a hard cap on children
- **Aggressive pruning**: Only 2–4 survivors per generation reach confirmation
- **Staged evaluation**: Short screening kills bad ideas in 1 hour, not 5
- **Hard DSL bounds**: No candidate can produce pathological behavior that wastes the full run before failing; the bounds catch it
- **No code-level mutation in this phase**: Only spec-level search, which is cheap to generate and predictable to evaluate

The search does not grow unbounded. It is a fixed-depth, fixed-width tree with pruning at every level.

### What If We Want to Go Deeper?

If early results show strong signal and we want to push further:

| Extension | Additional compute |
|---|---|
| Double the generations (10 → 20) | +400–800 machine-hours |
| Add a longer-horizon final validation (10+ hrs per candidate, top 5 winners only) | +150 machine-hours |
| Multi-hyperparameter robustness sweep for final winner (5 LR × 3 seeds) | +75 machine-hours |

These are optional and conditional on results. The core search program is the 300–800 hours above.

---
## The baseline is strong

The starting point is NanoChat's Muon optimizer — a competitive, modern, orthogonalized optimizer with cautious weight decay and factored second moments. Improvements over this baseline are meaningful, not just improvements over vanilla SGD.

### The evaluation hierarchy prevents overfitting

Candidates earn progressively more expensive evaluation. Short-run wins don't count unless they survive multi-seed reruns. The win hierarchy explicitly penalizes:
- Speed regression
- Memory regression
- Instability
- Tuning sensitivity

### The DSL was designed for this

The DSL encodes exactly the structure needed for evolution to express phase-aware behavior: sensors that track training dynamics, a smooth gate that conditions on them, and actuator poles that shift optimizer aggressiveness. Evolution needs to discover the right coefficients for a well-designed reaction function.

---

## What We Are Building Toward

The target outcome is an evolved optimizer policy that achieves a **better quality-vs-compute tradeoff** than NanoChat's baseline Muon: same final validation loss in measurably less wall-clock time, or better final loss at the same compute budget. The policy is interpretable, robust across seeds and hyperparameter neighborhoods, and directly usable by anyone training transformers with the Muon/AdamW split.

That is the goal. The search infrastructure, the DSL, the evaluation harness — all of it exists to get there.

### Win Hierarchy

The project validates its trajectory through a hierarchy of wins, from foundational to full result. Each win is a confirmed checkpoint on the path to the target.

**Win 1 — The evaluator separates good from bad.** The search harness, running on real NanoChat training, reliably ranks optimizer variants. Known-good mutations score well. Known-bad mutations score badly. Same seed gives same result. This confirms the evaluation substrate is trustworthy and the search has signal.

**Win 2 — Stateful gating shows correlated behavior with training phase.** The gate output shifts measurably as training progresses. The sensors (loss EMA, grad norm, alignment) track real training dynamics and the actuators respond. This confirms the DSL is expressive enough and the optimizer is actually reacting to training state, not just running a fixed policy with extra overhead.

**Win 3 — An evolved stateful optimizer beats the fixed baseline.** An evolved gated optimizer reaches the same validation loss as baseline Muon in fewer steps or less wall-clock. The gate weights are interpretable: we can read them and explain what the optimizer learned to do differently in early vs. late training. This is the first concrete result.

**Win 4 — The winning policy is robust.** The improvement holds across 3+ seeds, modest LR variation, and different training horizons. It requires less tuning, not more. A training team can drop it in and benefit without recalibration. This confirms the result is real, not a lucky seed.

**Win 5 — The result is production-relevant.** The evolved policy delivers a meaningfully better quality-vs-compute tradeoff on NanoChat. The improvement, the lineage, the gate weights, the evaluation data — all released openly. Anyone using the Muon/AdamW split can adopt it directly. The sponsor is credited with enabling a result that changes how people train models.

---

## Technical Prerequisites

| Requirement | Status |
|---|---|
| PyTorch ≥ 2.0 | Available |
| NanoChat codebase (local fork) | Available at `nanochat/` |
| GPU access for evaluation runs | **Needed — compute sponsorship target** |
| CUDA-compatible hardware | Required for real NanoChat training |
| SSH-accessible remote GPU instances | Supported by existing deployment infrastructure |

---

## Why Open Source

This project is open source because it has to be.

### Optimizer improvements only matter if people use them

An optimizer policy discovered behind closed doors, published as a paper with no code, adopted by nobody — that is a wasted result. The entire value proposition of discovering a better optimizer is that it **changes how people train models**. That only happens if the result is public, reproducible, and easy to adopt.

NanoChat itself is open source. The optimizer split it uses (Muon for matrix params, AdamW for everything else) is becoming a community standard. If we discover a better policy for that split, releasing it openly means it can be adopted by anyone using the same pattern — which is increasingly everyone training competitive small-to-mid-scale transformers.

### Reproducibility is the credibility

The claim "evolution discovered a better optimizer" is only credible if:
- The complete evolutionary lineage is public (every generation, every mutation, every score)
- The evaluation harness is reproducible (same seeds, same data, same training code)
- Anyone can rerun the winner against the baseline and verify the result

Closed-source results in optimizer research have a long history of not reproducing. Open source is the only honest way to present this kind of work.

### The search infrastructure is the contribution, not just the result

Even if the first search run produces modest improvements, the infrastructure — the bounded DSL, the gated optimizer runtime, the mutation system, the staged evaluation pipeline — is independently valuable. Other researchers can use it to search for optimizer policies on their own substrates, with their own models, with their own compute. That multiplier effect only exists if the code is open.

### Evolution discovers things humans wouldn't design

The Li et al. (2026) results demonstrate this clearly: VAD-CFR's hard warm-start at iteration 500 and regret-magnitude weighting were described by the authors as "non-intuitive to human designers." When evolution finds something surprising, the scientific community needs to be able to inspect it, poke at it, ablate it, and understand *why* it works. That process requires open access to the full code, the full data, and the full evolutionary trace.

---

## Repository

All code is in the nanoevolve repository. The key entry points:

**Search control plane** (`adamopt/`):
- [spec.py](adamopt/optim_search/spec.py) — bounded optimizer DSL, stateful control config, baseline and variant specs
- [candidate_optimizer.py](adamopt/optim_search/candidate_optimizer.py) — spec-driven optimizer runtime, gating, actuators, NanoChat parameter split
- [mutations.py](adamopt/optim_search/mutations.py) — 13 composable mutation operators
- [eval_candidate.py](adamopt/optim_search/eval_candidate.py) — evaluation harness
- [score.py](adamopt/optim_search/score.py) — composite scoring and Pareto frontier
- [tournament.py](adamopt/optim_search/tournament.py) — generation loop, multi-seed promotion
- [archive.py](adamopt/optim_search/archive.py) — candidate archive and persistence
- [search_optimizer.py](adamopt/scripts/search_optimizer.py) — CLI entrypoint for all search operations

**Deployment infrastructure** (`adamopt/`):
- [command_mutator.py](adamopt/optim_search/command_mutator.py) — code mutation via codex/claude
- [validation.py](adamopt/optim_search/validation.py) — local preflight validation
- [deployment.py](adamopt/optim_search/deployment.py) — remote deployment and trace capture
- [autonomous.py](adamopt/optim_search/autonomous.py) — persistent async patch/deploy/poll controller

**Training substrate** (`nanochat/`):
- Real GPT model, real training data, real optimizer split
- Patch targets: `nanochat/gpt.py` (optimizer routing), `nanochat/optim.py` (Muon/AdamW math)

**Strategy documents:**
- [EVOLUTION_STRATEGY.md](adamopt/EVOLUTION_STRATEGY.md) — staged search plan (DSL first → real evaluation → long runs → code mutation)
- [WIN_HIERARCHY.md](adamopt/WIN_HIERARCHY.md) — what counts as a win and how wins are ranked
- [checkpoint.md](checkpoint.md) — current project state and what's verified
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to contribute (search runs, DSL extensions, analysis)

---

## Contact

For questions, collaboration, or compute sponsorship inquiries, please reach out to me at ankit@clioapp.ai
