# AdamOpt Win Hierarchy

The final outcome is not "a better AdamW" in the abstract.

It is a training policy that wins under chosen operational constraints.

That means the search objective is not:

- prettier update equations
- surprising optimizer algebra
- one lucky short-run result

The search objective is:

- better training economics
- better training reliability
- better scaling behavior

This file defines what counts as a win and how wins should be ranked.

## What A Winner Actually Is

A winner is an optimizer policy that improves one or more of:

- time to target quality
- quality at fixed compute
- stability margin
- hyperparameter robustness
- throughput-adjusted training efficiency

In practice, the most meaningful winner is usually:

- same or better validation quality in less wall-clock

That metric compresses several things at once:

- optimizer quality
- speed overhead
- stability

If an optimizer gets slightly lower validation bpb but adds large per-step overhead, that is not automatically a win.

## The Main Win Types

### 1. Faster Training

Definition:

- same final quality
- fewer steps or less wall-clock

Example:

- baseline reaches val bpb 1.20 in 8 hours
- evolved optimizer reaches val bpb 1.20 in 6.5 hours

Why it matters:

- cheaper training
- faster iteration
- less cluster occupancy

This is one of the cleanest possible wins.

### 2. Better Sample Efficiency

Definition:

- same token budget
- better validation loss or bpb

Example:

- both candidates train on 10B tokens
- evolved optimizer ends at lower validation bpb

Why it matters:

- more learning per token
- better use of fixed-data or fixed-budget runs

This is especially valuable when token budget is the hard constraint.

### 3. Better Stability

Definition:

- fewer divergence events
- fewer NaN/Inf failures
- tolerates larger learning rates or larger batches

Examples:

- survives a higher LR that kills the baseline
- maintains stable grad/update norms in runs where baseline spikes
- requires less defensive clipping or manual babysitting

Why it matters:

- fewer wasted runs
- easier production use
- safer scaling

A stable optimizer is often worth more than a fragile one with a tiny peak advantage.

### 4. Better Scaling

Definition:

- advantage becomes clearer as model size, batch size, or training horizon grows

Example:

- only marginally better on small runs
- clearly better on larger runs with the same training recipe

Why it matters:

- small-run signal translates into large-run value
- same optimizer recipe remains useful as the training program grows

This is the highest-value technical win.

### 5. Better Robustness

Definition:

- improvement survives across seeds, nearby hyperparameters, and nearby architectures

Examples:

- wins across 2-3 seeds
- remains competitive under modest LR variation
- still helps on a slightly different model width/depth or batch setting

Why it matters:

- less benchmark overfitting
- more trustworthy result
- lower retuning cost

If it only wins under one narrow setup, it is not a strong winner.

## What We Actually Want To Optimize

The most operationally useful target is usually:

- time to reach a target validation loss or target validation bpb

Why this should be treated as a primary metric:

- it captures learning quality
- it captures optimizer overhead
- it captures instability penalties indirectly

This should sit above purely aesthetic or equation-level judgments.

## What We Should Hope To Discover

The most realistic high-value result is:

- same or better model quality
- with fewer tokens or less wall-clock
- while staying at least as stable as the baseline

That is much more valuable than:

- a mathematically interesting optimizer that is slow
- a one-off short-run win
- a candidate that requires constant retuning

## The Win Hierarchy

From weakest to strongest:

1. wins one benchmark
2. wins short-horizon training
3. wins across seeds
4. wins across scales
5. wins in wall-clock
6. wins while requiring less tuning

This is the project's win hierarchy.

It should guide:

- candidate scoring
- promotion criteria
- experiment review
- final claims about discovered optimizers

## Why "Less Tuning" Is A Real Win

An optimizer that is less sensitive to:

- learning rate
- weight decay
- warmup
- batch size

can improve research speed even when raw final loss improvements are modest.

That matters because:

- fewer runs are wasted on tuning
- fewer knobs need hand-calibration
- scaling to a new setup gets easier

This can be as important as pure training-speed gains.

## How This Should Affect Search Policy

The search loop should not promote candidates only because they:

- beat the baseline once
- look good on a short horizon
- have a high raw quality score with large runtime overhead

Promotion should increasingly demand:

### Early stage

- fixed-seed short-run improvement
- no unacceptable speed or memory regression
- no stability failure

### Mid stage

- multi-seed consistency
- acceptable wall-clock efficiency
- acceptable stability margin

### Late stage

- longer-run advantage
- scale sensitivity checks
- robustness under nearby hyperparameters

## How This Maps To Project Stages

### DSL Stage

At the DSL stage, the search should mainly optimize for:

- short-horizon quality at fixed compute
- early stability signal
- cheap wall-clock-aware screening

The DSL stage is not where final victory is declared.

It is where candidates earn the right to receive more expensive evaluation.

### Full Training Stage

At the full training stage, the search should mainly optimize for:

- time to target quality
- quality at fixed large compute
- scale behavior
- robustness

This is where true winners are confirmed.

## Practical Rule For Declaring A Winner

Do not declare a candidate a real winner unless it satisfies all of:

- beats the baseline on the primary metric
- stays within acceptable step-time overhead
- stays within acceptable memory overhead
- survives multi-seed reruns
- does not show worse stability behavior

For stronger claims, also require:

- evidence at a longer training horizon
- evidence at one or more larger scales
- evidence that it is not unusually sensitive to tuning

## Final Summary

The end state is not "we found new optimizer math."

The end state is:

- a discovered optimizer recipe
- that improves training economics
- by being faster, more stable, more sample-efficient, more robust, or some Pareto tradeoff of those

That is the standard AdamOpt should use when deciding what counts as a win.
