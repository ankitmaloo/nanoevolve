# Contributing to NanoEvolve

This is an evolutionary optimizer search project. The most valuable contributions are not code fixes — they are **search runs, DSL extensions, and analysis of what evolution discovers**.

## Ways to Contribute

### 1. Run Search and Share Results (highest impact)

If you have GPU access, run tournaments and share the results. This is the single most valuable thing you can do.

**What to do:**

```bash
# Run a tournament with the toy backend to verify setup
python adamopt/scripts/search_optimizer.py tournament \
  --backend toy \
  --generations 3 \
  --population 6 \
  --promotion-seeds 7,13,29 \
  --out-dir adamopt/runs/your_run_name
```

**What to share:**

- The full `out-dir` contents: archive, lineage, metrics, summary
- Your hardware setup (GPU type, memory, machine count)
- Any observations about which mutations survived and which didn't

Submit results as a PR adding your run directory under `adamopt/runs/community/`, or open an issue with a link to your results.

Every search run adds selection pressure from a different starting point. Even runs that don't find a winner produce useful signal about which regions of the search space are dead and which are alive.

### 2. Extend the DSL

The optimizer DSL defines what evolution can express. Extending it directly expands the search space.

**New sensors** — Add a new training-state signal to `ALLOWED_STATE_SENSORS` in [spec.py](adamopt/optim_search/spec.py) and wire it through [candidate_optimizer.py](adamopt/optim_search/candidate_optimizer.py). Examples of sensors that could be useful:

- Curvature proxy (gradient-of-gradient norm)
- Parameter norm growth rate
- Loss oscillation frequency
- Batch-to-batch loss variance

**New actuators** — Add a new actuator dimension to `AdaptiveActuatorConfig` in [spec.py](adamopt/optim_search/spec.py). This gives evolution a new behavioral axis to modulate. Examples:

- Momentum coefficient (currently fixed)
- Nesterov on/off blending
- Newton-Schulz step count (currently discrete, could be continuous-blended)
- Weight decay strength

**New gate architectures** — The current gate is a single sigmoid over a linear combination of sensors. More expressive alternatives:

- Multi-gate (separate gates for different actuator groups)
- Piecewise-linear gates
- Two-layer gating (sensor → hidden → gate)

Any DSL extension must:
- Have bounded parameters (hard min/max on all evolvable values)
- Round-trip through `to_dict()` / `from_dict()`
- Have at least one mutation operator in [mutations.py](adamopt/optim_search/mutations.py)
- Pass existing tests plus new tests for the extension

### 3. Add Mutation Operators

More mutation operators give evolution more moves. Add new operators to [mutations.py](adamopt/optim_search/mutations.py).

Good mutation operators:
- Change one thing at a time
- Stay within DSL bounds
- Return a clean lineage dict explaining what changed
- Are composable with existing operators

Examples of operators that don't exist yet:
- Swap two sensor weights
- Reset gate to neutral (bias=0, all weights=0)
- Clone actuator ranges from one dimension to another
- Interpolate between two parent specs (crossover)

### 4. Wire a New Training Substrate

NanoChat is the primary substrate, but the evaluator contract in [eval_candidate.py](adamopt/optim_search/eval_candidate.py) is substrate-agnostic. If you have a different training codebase and want to run optimizer search on it:

- Implement the evaluator interface (takes a spec, returns metrics JSON)
- Register it as a backend option
- Share results so we can compare cross-substrate

### 5. Analyze Evolved Policies

When search runs produce winners, the most interesting question is **why** they work. Contributions that analyze evolved policies are valuable:

- Gate weight interpretation (what sensors matter, what signs mean)
- Actuator curve plots over training (when does behavior shift?)
- Ablation studies (disable one actuator, re-evaluate — what breaks?)
- Comparison to hand-designed schedules (does the evolved policy resemble cosine decay? warmup? something else entirely?)

If you find a clean explanation for why an evolved policy works, that is a publishable insight and you will be credited.

### 6. Improve Infrastructure

Lower-priority but still useful:
- Provider integrations for cloud GPU scheduling (AWS, GCP, CoreWeave, Lambda, etc.)
- Better monitoring / dashboards for long-running tournaments
- Evaluation caching (avoid duplicate baseline runs across tournaments)
- Richer failure taxonomies (OOM, divergence, slowdown)

## Development Setup

```bash
# Clone the repo
git clone <repo-url>
cd nanoevolve

# Create a virtualenv (Python 3.10+)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e adamopt/
pip install -e nanochat/

# Run tests
python -m pytest adamopt/tests -q
```

All 18 tests should pass. If they don't, open an issue.

## Code Conventions

- Type hints on all function signatures
- Frozen dataclasses for specs and configs (immutability matters for reproducibility)
- `dataclasses.replace()` for mutations, never mutate in place
- Every mutation returns `(new_spec, lineage_dict)` — lineage is how we trace what evolution did
- Tests use deterministic seeds

## Project Structure

Read these before diving in:

- [RESEARCH_PLAN.md](RESEARCH_PLAN.md) — what the project is, why it exists, what we're building toward
- [EVOLUTION_STRATEGY.md](adamopt/EVOLUTION_STRATEGY.md) — staged search plan
- [WIN_HIERARCHY.md](adamopt/WIN_HIERARCHY.md) — what counts as a win
- [checkpoint.md](checkpoint.md) — current project state

## Submitting Contributions

- **Search results**: PR adding your run under `adamopt/runs/community/` or an issue with a link
- **DSL extensions / mutation operators**: PR with tests. Must not break existing tests
- **Analysis**: Issue or PR adding a writeup under `docs/analysis/`
- **Infrastructure**: PR with tests

## Discussion

Open an issue for:
- Proposing new sensors, actuators, or gate architectures
- Sharing observations from search runs
- Asking questions about the DSL or search process
- Coordinating parallel search efforts to avoid duplicate work

## Credit

Contributors who run search, extend the DSL, or analyze results will be credited in publications and in the repository. If evolution discovers something interesting using your sensor, your mutation operator, or your compute — that is your contribution and it will be acknowledged.

## Contact

Reach out to ankit@clioapp.ai for coordination on larger contributions or compute-intensive search campaigns.
