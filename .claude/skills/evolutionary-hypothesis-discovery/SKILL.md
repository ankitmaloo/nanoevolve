---
name: evolutionary-hypothesis-discovery
description: Generate strong, distinct, mechanism-level hypotheses for evolutionary search, optimizer search, benchmark attacks, and constrained optimization. Use when Codex needs to uncover a good hypothesis before running experiments, avoid local-mode idea collapse, transfer ideas across domains, derive mutation operators from first principles, or produce a broad but defensible set of candidate directions for exploration.
---

# Evolutionary Hypothesis Discovery

Use this skill before pack design.

Its job is not to choose the final experiments. Its job is to discover candidate mechanisms that are worth testing at all.

A good hypothesis is:

- causal, not descriptive
- distinct from nearby variants
- falsifiable
- observable at some horizon
- plausible under the task constraints
- composable if it wins

The failure mode is local sampling: generating five versions of the same idea because they are all easy to imagine.

## Core Principle

Search over transformations of the solution space, not just direct guesses about the answer.

When discovering hypotheses, ask:

- what bottleneck could dominate the score
- what hidden constraint is shaping the frontier
- what variable is under-used or over-used
- what analogous problem has the same abstract structure
- what if the dominant story is wrong

The aim is to sample from the full distribution of plausible mechanisms, then compress that set into distinct families.

## Workflow

### 1. Write the abstract problem

Express the task without domain jargon.

Examples:

- fixed wallclock, maximize model quality
- fixed artifact budget, minimize deployment loss
- bounded optimizer DSL, maximize improvement over baseline
- limited evaluations, identify a winner with minimal regret

Then identify the hidden trade:

- quality vs speed
- capacity vs compressibility
- exploration vs certainty
- train-time gain vs eval-time gain

### 2. Identify bottleneck classes

List the major bottleneck types before generating ideas.

Common classes:

- compute bottleneck
- memory bottleneck
- optimization bottleneck
- representation bottleneck
- compression bottleneck
- evaluation bottleneck
- search bottleneck
- coordination bottleneck

Do not assume only one bottleneck matters.

### 3. Generate hypotheses by operator, not by inspiration

Use operator families to force broad sampling.

Required operators:

- `reallocate`
  - move budget from one subsystem to another
- `factorize`
  - replace a monolith with shared or low-rank structure
- `specialize`
  - give different tensor families or stages different rules
- `stage`
  - change behavior by phase, depth, or time
- `relax`
  - remove unnecessary precision, coupling, or control
- `tighten`
  - add discipline, clipping, decay, or normalization
- `externalize`
  - move work from training to eval, export, or preprocessing
- `internalize`
  - pull a late penalty into train-time optimization
- `borrow`
  - import an idea from an abstractly similar domain
- `invert`
  - test the opposite of the current dominant assumption

Generate at least one candidate from each operator family before pruning.

### 4. Use analogical transfer

Ask which other domains solved an abstractly similar problem.

Transfer from:

- compression
- control systems
- numerical linear algebra
- architecture search
- caching or systems design
- Bayesian optimization
- curriculum learning
- error-correcting codes

Do not transfer surface details. Transfer the mechanism.

Format the analogy as:

- source domain
- abstract mechanism
- translated mutation in the current domain
- why the mapping is valid

### 5. Probe invariants and anti-invariants

Write what probably must stay true:

- invariant examples:
  - final score must survive quantization
  - code growth must stay under budget
  - candidate must preserve numerical stability

Then ask what can be broken safely:

- homogeneity across layers
- one-optimizer-for-all-tensors
- one-eval-policy-for-all-runs
- one-precision-for-all-weights

Hypotheses often emerge from breaking a false invariant.

### 6. Generate anti-dominant hypotheses

Explicitly generate ideas that challenge the current leading narrative.

If the current story is:

- “training is the bottleneck”

generate:

- eval-only win
- export-only win
- systems-throughput win

If the current story is:

- “bigger context is better”

generate:

- shorter context but more updates
- asymmetric train/eval context
- selective long context only where it pays

Always include at least one anti-dominant family unless the task is fully settled.

### 7. Convert raw ideas into mechanism families

For each raw idea, rewrite it as:

- mechanism
- expected first-order effect
- dominant metric it should move
- cheapest observable signal
- likely failure mode

Then merge nearby ideas into one family.

Do not keep:

- tiny parameter sweeps
- multiple ideas with the same causal story
- ideas that only differ in magnitude

### 8. Score each family for search value

Score each family on:

- distinctness
- causal clarity
- expected upside
- cheap observability
- compatibility with constraints
- stackability if positive

A high-upside idea with no observable signal may still be worth keeping, but mark it as late-stage.

### 9. Produce mutation operators

When the user wants evolutionary search, output mutation operators, not just hypotheses.

For each family, define:

- mutation target
- allowable direction
- guardrails or bounds
- what should be held fixed
- what a child mutation is allowed to change next

This makes the hypothesis usable by an evolutionary loop.

### 10. Write negative knowledge

Record what should not be regenerated.

Examples:

- ideas already falsified
- ideas dominated by another family
- ideas unobservable in the current horizon
- ideas that only win through metric leakage

Negative knowledge is part of discovery. It prevents the search from revisiting dead neighborhoods.

## Output Format

When using this skill, produce:

1. abstract problem statement
2. major bottleneck classes
3. operator-generated raw ideas
4. analogical-transfer ideas
5. anti-dominant ideas
6. merged mechanism families
7. family scores:
   - distinctness
   - upside
   - observability
   - stackability
8. suggested mutation operators
9. negative knowledge
10. shortlist for pack design

If another skill will handle pack construction, end with `shortlist for pack design` rather than building the run matrix directly.

## Heuristics

Use these defaults:

- prefer breadth first, then prune
- generate by operator family before judging quality
- preserve one wildcard from a distant region
- treat abstraction transfer as mandatory, not optional
- write one anti-dominant hypothesis on purpose
- collapse local variants into one family
- separate discovery from evaluation

If the user asks for “new ideas,” do not answer with a flat list. Re-run the operator families and analogical-transfer pass.
