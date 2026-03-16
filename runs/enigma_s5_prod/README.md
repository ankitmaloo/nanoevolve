# Enigma Stage 5 Production Run

This folder is the home for the **Stage 5 production-faithful search loop**.

## Stage Definition

- Horizon: **20000 steps**
- Primary promoted parent baseline: `H60_solo`
- Reference baseline: pure production optimizer
- Optional secondary parent: `H73_solo`

## Required Context

Read:

- `runs/enigma_s5_prod/stage5_context.md`
- `runs/enigma_stage4_postmortem.json`
- `runs/enigma_stage4ext_postmortem.json`

## Intended Artifacts

- `results/` — result JSON files
- `diffs/` — candidate diffs
- `logs/` — slurm stdout/stderr and analysis notes

## Search Rule

Stage 5 should compare candidates against:

1. promoted parent baseline (`H60_solo`)
2. pure production baseline

This stage is for real 20k production runs, not another short-horizon extension batch.
