# NanoEvolve

This folder defines the composite project that combines:

- [`adamopt`](./adamopt)
- [`nanochat`](./nanochat)
- [`alphaevolve`](./alphaevolve)

The intended structure is:

```text
nanoevolve/
  adamopt/
  alphaevolve/
  nanochat/
```

Where:

- `adamopt/` owns optimizer search, scoring, lineage, promotion logic, validation, and remote orchestration
- `nanochat/` owns the real model, optimizer split, and training/evaluation substrate

## Responsibilities

### AdamOpt

Owned in [`adamopt`](./adamopt):

- bounded optimizer DSL
- spec mutation
- scoring and win hierarchy
- tournament logic
- archive persistence
- code mutation infrastructure
- validation and deployment orchestration

### NanoChat

Owned in [`nanochat`](./nanochat):

- real optimizer split
- real model implementation
- real training loop
- real evaluation behavior

### Evolution Code

Owned in [`alphaevolve`](./alphaevolve):

- prior evolution-agent experiments
- MVP evolutionary scaffolding
- reference material for search-loop design

## Project Rule

This project is not intended to make `adamopt/` independent of `nanochat/`.

The correct mental model is:

- `adamopt/` is the optimizer-search control plane
- `nanochat/` is the training execution plane

## Canonical Entry Points

Spec-search and evaluation entrypoints live in:

- [`adamopt/scripts/search_optimizer.py`](./adamopt/scripts/search_optimizer.py)

Training substrate and optimizer patch targets live in:

- [`nanochat/nanochat/gpt.py`](./nanochat/nanochat/gpt.py)
- [`nanochat/nanochat/optim.py`](./nanochat/nanochat/optim.py)

Strategy documents live in:

- [`adamopt/EVOLUTION_STRATEGY.md`](./adamopt/EVOLUTION_STRATEGY.md)
- [`adamopt/WIN_HIERARCHY.md`](./adamopt/WIN_HIERARCHY.md)

## Composite Workflow

The intended default workflow is:

1. mutate optimizer specs in `adamopt/`
2. evaluate candidates against NanoChat behavior
3. rank and promote winners in `adamopt/`
4. only later allow code-level optimizer mutation against `nanochat/`

## Manifest

The machine-readable composite layout is stored in:

- [`workspace.toml`](./workspace.toml)
