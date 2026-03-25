# CLAUDE.md

## Project Overview

NanoEvolve is an evolutionary optimizer search project. It discovers training-phase-aware optimizer policies through evolutionary search over a bounded DSL, evaluated against real NanoChat (GPT) training runs.

## Repository Structure

```
nanoevolve/
  adamopt/       # Optimizer search control plane (DSL, mutations, scoring, tournament, deployment)
  nanochat/      # Real GPT training substrate (model, optimizer, training loop)
  alphaevolve/   # Prior evolutionary code and reference material
```

- `adamopt/` and `nanochat/` are tightly coupled by design — adamopt is the search plane, nanochat is the execution plane.
- Machine-readable layout: `workspace.toml`

## Key Entry Points

- **CLI**: `adamopt/scripts/search_optimizer.py` — all search operations
- **DSL**: `adamopt/optim_search/spec.py` — bounded optimizer spec, stateful control config
- **Runtime**: `adamopt/optim_search/candidate_optimizer.py` — spec-driven optimizer, gating, actuators
- **Mutations**: `adamopt/optim_search/mutations.py` — 13 composable mutation operators
- **Evaluation**: `adamopt/optim_search/eval_candidate.py` — evaluation harness
- **Scoring**: `adamopt/optim_search/score.py` — composite scoring and Pareto frontier
- **Tournament**: `adamopt/optim_search/tournament.py` — generation loop, multi-seed promotion
- **Archive**: `adamopt/optim_search/archive.py` — candidate persistence
- **NanoChat patch targets**: `nanochat/nanochat/gpt.py`, `nanochat/nanochat/optim.py`

## Development

```bash
# Virtualenv is in the repo root
source .venv/bin/activate

# Install
pip install -e adamopt/
pip install -e nanochat/

# Run tests (from repo root)
python -m pytest adamopt/tests -q
# Expected: 18 passed
```

Python 3.10+ required. PyTorch >= 2.0.

## Code Conventions

- Type hints on all function signatures
- Frozen dataclasses for specs and configs — immutability matters for reproducibility
- `dataclasses.replace()` for mutations, never mutate in place
- Every mutation returns `(new_spec, lineage_dict)` — lineage traces what evolution did
- Tests use deterministic seeds
- DSL extensions must: have bounded parameters, round-trip through `to_dict()`/`from_dict()`, have mutation operators, pass tests

## Architecture Decisions

- **Matrix-only evolution**: Only matrix params (attention, MLP projections) use evolved optimizer. Embeddings, layer norms, scalars stay on AdamW.
- **Bounded DSL**: All evolvable parameters have hard min/max bounds. No gate coefficient exceeds +/-8.0.
- **Staged evaluation**: Short-horizon screening kills bad ideas cheaply; only survivors get full-length runs.
- **Spec-first search**: DSL mutation first, real NanoChat eval second, long-run promotion third, code-level mutation only after that.

## Current State

The search infrastructure is fully built. The main evaluator still uses a toy backend. The highest-priority next step is replacing it with a real NanoChat short-run evaluator. See `checkpoint.md` for detailed status.

## Parameter Golf (pgolf/)

The `pgolf/parameter-golf/` subtree is for the OpenAI Parameter Golf challenge: train the best LM in 10 min on 8xH100, 16MB artifact limit, measured by val_bpb.

### Experiment Execution Rules

- **Never run full-length (10 min) runs for screening.** Use three horizons:
  - Sanity (90s): catch crashes, bad envs, illegal export size
  - Screen (150-180s): rank ideas cheaply by relative signal
  - Decision (600s): only for promoted survivors
- **Always parallelize independent hypotheses.** If you have N hypotheses and N GPUs, run N × 1-GPU screens simultaneously. Never run them sequentially on 8 GPUs each.
- **Pivot to the strongest known stack.** Don't mutate a weaker baseline hoping to catch up. Adopt the SOTA stack and improve from there.
- **One causal question per hypothesis.** Each slot tests exactly one env-var change (or one coherent set). No "let's try everything at once."
- **Use the orchestrator.** `stage2/h100_matrix_r2/orchestrate_stage2.py` handles parallel launch, metric parsing, auto-promotion, and summary generation. Don't hand-roll bash scripts for the same job.
- **Control repeat is mandatory.** R0A vs R0B gives noise calibration for free. Without it, small deltas are uninterpretable.
- **No code edits to train_gpt.py.** All mutations via environment variables only.

### Key Paths

- Orchestrator: `pgolf/parameter-golf/stage2/h100_matrix_r2/orchestrate_stage2.py`
- Strategy entrypoint: `pgolf/parameter-golf/stage2/sota_stack_strategy_r1/run_strategy.py`
- Run configs: `pgolf/parameter-golf/stage2/sota_stack_strategy_r1/run_configs.json`
- Training script: `pgolf/parameter-golf/train_gpt.py` (read-only, all config via env vars)
- Records: `pgolf/parameter-golf/records/track_10min_16mb/`

### H100 Box (Hyperbolic)

- SSH: `ubuntu@147.185.41.138`
- Venv: `/data/pgolf_venv/bin/python`
- uv: `~/.local/bin/uv`
- Repo: `/data/parameter-golf/`
- Dataset: `/data/parameter-golf/data/datasets/fineweb10B_sp1024/`
- Set `PGOLF_PYTHON=/data/pgolf_venv/bin/python` when launching the orchestrator
- Use `nohup ... &` + `disown` for detached runs — but launch exactly once, not twice

### Hypothesis Screening Doctrine

The final score decomposes into three independent components:

```
final BPB = training quality at 10 min + export/quant penalty - eval lift
```

Each component has its own cheapest predictive screen. Do not use one screening method for all hypotheses.

#### Lane A: Training-dynamics hypotheses

What they change: optimizer, schedule, geometry, init, architecture.

Screen method: 1-GPU parallel training runs against matched control.

Read: early validation trajectory (not just final point), ms/step, steps reached.

Kill if: slower and not learning faster, or same speed but worse validation.

Examples: Muon WD, momentum tuning, batch size, warmdown length, adaptive Muon.

#### Lane B: Export/quantization hypotheses

What they change: how the checkpoint is compressed into the 16MB artifact.

Screen method: train ONE baseline checkpoint, then run many export variants on it. No retraining.

Read: size, quantization gap, final post-quant BPB.

Kill if: bigger and not better, or smaller but quality collapses.

Examples: fp16 embeddings, clip percentile, row-scale dtype, int6 vs int8, keep-last-K fp16.

#### Lane C: Eval-policy hypotheses

What they change: how the artifact is evaluated at inference time.

Screen method: take the same checkpoint, run multiple eval policies on it. No retraining.

Read: eval BPB lift, eval wall time.

Kill if: gain is tiny, or eval time is operationally bad.

Examples: sliding window, stride, eval seq len, eval batch size.

#### The search tree

This factorization gives a natural search order:

1. Find the best training trunk (Lane A screens)
2. Find the best export policy on that trunk (Lane B bakeoff on the winning checkpoint)
3. Find the best eval policy on that trunk (Lane C bakeoff on the winning checkpoint)
4. Test the composite on 8xH100 (one full decision run)

This kills bad ideas early without wasting full-cluster runs. A Lane B or Lane C hypothesis never needs a training screen — it just needs the checkpoint.

#### GPU budget strategy

With 8 GPUs and N hypotheses in the same lane:

- 2 GPUs for replicated controls (noise calibration)
- 6 GPUs for candidates
- The control repeat (R0A vs R0B) tells you whether a +0.005 delta is signal or noise

#### What is predictive on 1-GPU screens

Good signals: relative learning curve vs control, delta ms/step, delta steps reached, quant gap on same checkpoint, eval lift on same checkpoint.

Bad signals: absolute leaderboard closeness, mixing training + export + eval in one tiny run, late-regime schedule claims from ~300 steps.

#### Decision rule for training screens

Do not just look at final post-quant BPB after 180 seconds. Instead:

- delta val_bpb at matched wallclock
- delta val_bpb at matched step count
- delta step_avg
- combined: keep only if it wins on one axis without losing badly on the others

### Submission Records

- Do not include author name or github_id in submission.json
- Do not add co-author lines to commits
- Follow existing record format in `records/track_10min_16mb/`

## Strategy Documents

- `adamopt/EVOLUTION_STRATEGY.md` — staged search plan
- `adamopt/WIN_HIERARCHY.md` — what counts as a win
- `RESEARCH_PLAN.md` — full research motivation and compute estimates
- `checkpoint.md` — current project state and verified properties
