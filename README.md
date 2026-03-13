# NanoEvolve

**Discovering training-phase-aware optimizer policies through evolutionary search.**

Modern optimizers use fixed schedules — learning rate warmup, cosine decay, beta annealing — applied uniformly regardless of what is actually happening during training. NanoEvolve evolves optimizer policies that **sense where training is and adapt their behavior accordingly**, using evolutionary search over a bounded optimizer DSL.

The substrate is [NanoChat](https://github.com/karpathy/nanochat) — a real, competitive GPT training codebase. Any optimizer policy that wins on NanoChat wins on real transformer training.

---

## The Core Idea

Rather than evolving arbitrary optimizer code, we constrain the search to a **bounded DSL** where the evolvable surface is a set of smooth gates conditioned on training-state signals.

**Six training-state sensors** feed into a smooth sigmoid gate:

| Sensor | What it captures |
|---|---|
| `loss_ema` | Smoothed current loss — where are we in training? |
| `loss_improvement_ema` | Rate of loss decrease — are we still making progress? |
| `grad_norm_ema` | Average gradient magnitude — is the signal strong or fading? |
| `update_ratio_ema` | Update size relative to parameter size |
| `grad_alignment_ema` | Cosine similarity between consecutive gradients — consistent or noisy? |
| `step_fraction` | Progress through total training budget |

The gate interpolates between **aggressive** and **conservative** behavior poles across five optimizer dimensions: update multiplier, trust ratio, clip threshold, second-moment beta, and orthogonal projection intensity.

Evolution discovers which sensor correlations predict "we're in the finicky regime" and shifts the actuators accordingly. This is **state-dependent annealing** — not a fixed schedule, but a learned reaction function.

---

## Why This Works

We are **not** trying to replace Adam or Muon with something fundamentally different. We **are** trying to discover the best *policy for when and how aggressively* to apply known optimizer techniques. The base math stays fixed. Only the regime-dependent blending is evolved.

This approach has strong precedent. [Li et al. (2026)](https://arxiv.org/abs/2602.16928) applied AlphaEvolve to multi-agent learning and discovered state-adaptive policies that outperform all hand-designed baselines — including volatility-adaptive discounting and dynamically annealed blending factors. We apply the same principle to neural network optimization.

---

## Architecture

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

**Key design decisions:**

- **Matrix-only evolution**: Only matrix params (attention, MLP projections) use the evolved optimizer. Embeddings, layer norms, scalars stay on AdamW — mirroring the NanoChat/Muon split.
- **Bounded DSL**: All evolvable parameters have hard min/max bounds. No pathological candidates.
- **Staged evaluation**: Short-horizon screening kills bad ideas cheaply. Only survivors get full-length runs.
- **Composable mutations**: 13 mutation operators, each changing exactly one aspect of the spec. Clean lineage tracking.

---

## Repository Structure

```
nanoevolve/
  adamopt/       # Optimizer search control plane
  nanochat/      # Real GPT training substrate
  alphaevolve/   # Prior evolutionary code and reference material
```

### Search Control Plane (`adamopt/`)

| Module | Role |
|---|---|
| [`spec.py`](adamopt/optim_search/spec.py) | Bounded optimizer DSL, stateful control config |
| [`candidate_optimizer.py`](adamopt/optim_search/candidate_optimizer.py) | Spec-driven optimizer runtime, gating, actuators |
| [`mutations.py`](adamopt/optim_search/mutations.py) | 13 composable mutation operators |
| [`eval_candidate.py`](adamopt/optim_search/eval_candidate.py) | Evaluation harness |
| [`score.py`](adamopt/optim_search/score.py) | Composite scoring and Pareto frontier |
| [`tournament.py`](adamopt/optim_search/tournament.py) | Generation loop, multi-seed promotion |
| [`archive.py`](adamopt/optim_search/archive.py) | Candidate persistence and lineage |
| [`search_optimizer.py`](adamopt/scripts/search_optimizer.py) | CLI entrypoint |

### Deployment Infrastructure (`adamopt/`)

| Module | Role |
|---|---|
| [`command_mutator.py`](adamopt/optim_search/command_mutator.py) | Code mutation via LLM |
| [`validation.py`](adamopt/optim_search/validation.py) | Local preflight validation |
| [`deployment.py`](adamopt/optim_search/deployment.py) | Remote deployment and trace capture |
| [`autonomous.py`](adamopt/optim_search/autonomous.py) | Async patch/deploy/poll controller |

### Training Substrate (`nanochat/`)

Real GPT model, real training data, real optimizer split. Patch targets: `nanochat/gpt.py`, `nanochat/optim.py`.

---

## Getting Started

```bash
# Python 3.10+, PyTorch >= 2.0 required

# Install both packages
pip install -e adamopt/
pip install -e nanochat/

# Run tests (18 passing)
python -m pytest adamopt/tests -q
```

---

## Compute Budget

| Scenario | Screening | Confirmation | Baseline | Total |
|---|---|---|---|---|
| Conservative (5 gens, pop 8) | 40 hrs | 225 hrs | 35 hrs | **~300 machine-hours** |
| Aggressive (10 gens, pop 16) | 160 hrs | 600 hrs | 35 hrs | **~800 machine-hours** |

With 4 machines in parallel, the conservative scenario finishes in ~75 hours wall-clock.

The compute is bounded because the search is bounded: fixed population sizes, aggressive pruning, staged evaluation, and hard DSL bounds prevent runaway costs.

---

## Win Hierarchy

The project validates through progressively harder wins:

1. **Evaluator separates good from bad** — the search harness reliably ranks optimizer variants on real NanoChat training
2. **Stateful gating correlates with training phase** — gate output shifts measurably as training progresses
3. **Evolved optimizer beats the fixed baseline** — same validation loss in fewer steps or less wall-clock
4. **Winning policy is robust** — holds across 3+ seeds, modest LR variation, different horizons
5. **Result is production-relevant** — meaningfully better quality-vs-compute tradeoff, openly released

---

## Current Status

The search infrastructure is fully built and tested. The main evaluator still uses a toy backend. The highest-priority next step is replacing it with a real NanoChat short-run evaluator to unlock real selection pressure. See [`checkpoint.md`](checkpoint.md) for detailed status.

---

## Strategy Documents

- [`EVOLUTION_STRATEGY.md`](adamopt/EVOLUTION_STRATEGY.md) — staged search plan
- [`WIN_HIERARCHY.md`](adamopt/WIN_HIERARCHY.md) — what counts as a win
- [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) — full research motivation and compute estimates
- [`checkpoint.md`](checkpoint.md) — current project state

---

## Contact

For questions, collaboration, or compute sponsorship inquiries: ankit@clioapp.ai
