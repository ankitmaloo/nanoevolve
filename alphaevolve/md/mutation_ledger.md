# Mutation Ledger

This file enforces non-overlap within a generation and controlled revisit across generations.

## Purpose
Track mutation neighborhoods, not just candidate ids.

A mutation neighborhood is the combination of:
- target region,
- operator family,
- bottleneck attacked,
- hypothesis family,
- benchmark slice or regime being stressed.

Operator family should distinguish:
- structural rewrite,
- policy rewrite,
- latent-state introduction,
- schedule / annealing invention,
- low-value knob retune baseline.

If two candidates share almost all of those, they overlap.

Two knob retunes of the same rule usually overlap completely and should not occupy separate slots unless used as an explicit baseline and ablation.

## Rules
- In one generation, no two active portfolio slots should occupy the same neighborhood unless one is an explicit ablation pair.
- Across generations, do not revisit a neighborhood unless you can state what changed:
  - new evidence,
  - new parent code,
  - new constraint,
  - new operator family,
  - new benchmark slice,
  - failed assumption corrected.
- If a neighborhood fails repeatedly, retire it here.
- If a retired neighborhood is reopened, record the reason explicitly.

## Entry Template
### MXX: Neighborhood name
- loop:
- hypothesis:
- target region:
- operator family:
- bottleneck attacked:
- benchmark slice:
- parent candidate:
- relation to prior neighborhoods:
- overlap status:
- why allowed:
- outcome:
- retire status:
- reopen condition:
- notes:

## Overlap Test
Treat a candidate as overlapping if it matches an earlier one on:
- the same target region,
- the same main mechanism,
- the same bottleneck attacked,
- and the same evaluation slice focus.

If overlap is partial, record whether it is:
- acceptable diversification,
- intentional ablation,
- accidental duplication.

## Active Neighborhoods
Add current-loop neighborhoods below.
