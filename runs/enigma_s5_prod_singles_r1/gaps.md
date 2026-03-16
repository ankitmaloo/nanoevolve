# Stage 5 Gaps — Singles Rerun

These are the set-level gaps the previous batch left uncovered.

## G1. Muon update semantics gap

We have wins on Muon schedules, but almost no exploration of the actual Muon update rule. `H531` targets this directly by changing the cautious weight decay gate source.

## G2. Regime-boundary state gap

The 20k H60 curve implies the warmdown boundary matters. No prior run tested a hard state intervention exactly at that boundary. `H532` and `H537` target this.

## G3. Shape-specialization gap

Muon groups are already shape-bucketed in production, but all prior logic treated them uniformly. `H533` is the first explicit shape-aware Muon policy in production.

## G4. Compute-depth scheduling gap

`ns_steps` is a real algorithmic knob, not just a hyperparameter, because it changes the orthogonalization approximation itself. `H534` samples this without collapsing to a permanently shallow Muon.

## G5. AdamW subgroup gap

Previous AdamW wins were global. The code has stable subgroups with very different roles. `H535`, `H536`, and `H538` target subgroup-specific behavior instead of one rule for the entire AdamW path.
