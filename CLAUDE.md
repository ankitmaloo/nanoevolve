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

## Strategy Documents

- `adamopt/EVOLUTION_STRATEGY.md` — staged search plan
- `adamopt/WIN_HIERARCHY.md` — what counts as a win
- `RESEARCH_PLAN.md` — full research motivation and compute estimates
- `checkpoint.md` — current project state and verified properties
