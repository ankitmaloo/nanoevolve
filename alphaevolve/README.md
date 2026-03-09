# AlphaEvolve MVP

This folder contains a setup-first MVP of an AlphaEvolve-style loop.

## Goals
- Evolve multiple program variants.
- Evaluate each variant on a harder popular algorithm target (A* routing heuristic).
- Use a harder A* benchmark with weighted terrain and holdout scenarios to reduce one-shot wins.
- Feed metrics and failure reasons back into next-generation mutation prompts.
- Co-evolve mutation prompt strategies based on reward.
- Keep only survivable candidates.
- Run LLM mutation calls with `asyncio` + semaphore-gated concurrency.

## Constraints
- Everything is self-contained in `alphaevolve/`.
- Default model is `gemini-3-flash-lite` for Gemini mode.
- OpenAI mode uses an ensemble schedule: base/slow `gpt-5.2` plus fast `gpt-5.2-mini` (slow used every 4 calls by default).
- Gemini usage follows local SDK patterns from `gemini_examples/`.
- OpenAI usage follows local SDK patterns from `openai_examples/` (Responses API).
- MVP prioritizes loop integrity and observability over outcome quality.

## Setup
```bash
cd alphaevolve
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python mvp/cli.py setup-check
```

## Commands
```bash
# Build/update PAPER_STUDY.md from alphaevolve.pdf
python mvp/cli.py study-paper

# Dry run before live API usage (full loop, mock diffs)
python mvp/cli.py dry-run --generations 5

# Watch a live terminal dashboard while it runs
python mvp/cli.py dry-run --generations 5 --parallel-candidates 4 --llm-concurrency 4 --tui

# One-command A* demo run + HTML report
python mvp/cli.py demo-astar --mode mock --generations 10 --run-name demo_astar

# Deterministic progression demo across 4 generations (best for presentations)
python mvp/cli.py demo-astar-progression --tui

# Run with Gemini API once GEMINI_API_KEY is set
python mvp/cli.py run --mode gemini --generations 10

# Hardened A* demo profile on gemini-3-flash-latest
python mvp/cli.py demo-astar-gemini-latest --tui

# Run with OpenAI API once OPENAI_API_KEY is set
python mvp/cli.py run --mode openai --model gpt-5.2 --fast-model gpt-5.2-mini --slow-every 4 --generations 10

# Increase mutation throughput with async parallel calls
python mvp/cli.py run --mode openai --parallel-candidates 6 --llm-concurrency 6 --generations 10

# Generate HTML report from an existing run
python mvp/cli.py render-report --run-dir runs/demo_astar
```

## Output Artifacts
Each run is stored under `alphaevolve/runs/<timestamp>/`:
- `events.jsonl`: generation-by-generation records
- `prompts/`: prompts sent to mutator
- `diffs/`: returned diff blocks
- `evaluations/`: metrics and stage results
- `seed_program.py`: baseline implementation used in run
- `best_program.py`: best candidate program source
- `summary.json`: final run summary
- `demo_report.html`: human-readable demo page (baseline -> evolution -> final)
- live TUI: optional real-time dashboard with slot status and score progression (`--tui`)

## File Guide
- `mvp/controller.py`: orchestrates evolutionary loop
- `mvp/evaluator.py`: runs toy evaluator and stage-gated scoring
- `mvp/population_db.py`: parent sampling, inspirations, survivor pruning
- `mvp/mutator_mock.py`: mock mutation replay
- `mvp/mutator_gemini.py`: Gemini mutation path
- `mvp/mutator_openai.py`: OpenAI Responses API mutation path
- `mvp/diff_engine.py`: SEARCH/REPLACE parser and applier
- `mvp/tasks/astar_routing_target.py`: recommended harder evolvable target program
- `mvp/prompt_evolver.py`: co-evolves prompt strategies and tracks prompt reward
- `mvp/tui.py`: real-time terminal dashboard for generation and slot-level updates
- `WALKTHROUGH.md`: end-to-end code reading guide
