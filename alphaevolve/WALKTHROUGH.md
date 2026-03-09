# AlphaEvolve MVP Walkthrough

## Purpose
This MVP proves the loop mechanics of AlphaEvolve-style evolution:
1. mutate code,
2. evaluate code,
3. feed metrics/failures back,
4. co-evolve mutation prompts,
5. keep survivors.

## Read Order
1. `mvp/tasks/astar_routing_target.py`: recommended harder seed program with evolve blocks.
2. `mvp/controller.py`: main generation loop.
3. `mvp/diff_engine.py`: SEARCH/REPLACE parser and applier.
4. `mvp/evaluator.py`: toy evaluation and stage gating.
   A* evaluator now uses train + holdout scenario suites with weighted terrain costs.
5. `mvp/prompt_evolver.py`: prompt strategy sampling, reward updates, and mutation.
6. `mvp/population_db.py`: parent sampling, inspirations, survivor pruning.
7. `mvp/mutator_mock.py`, `mvp/mutator_gemini.py`, and `mvp/mutator_openai.py`: mutation sources.
`mutator_openai.py` uses both slow/base and fast models in a scheduled ensemble.
8. `mvp/tui.py`: optional live terminal dashboard (`--tui`) for real-time visibility.
9. `mvp/cli.py`: user entrypoints.

## Core Flow
- Seed program is evaluated and inserted into population.
- For each generation:
  - Build a batch of parent+inspiration+feedback contexts.
  - Run mutation calls in parallel using `asyncio` and a semaphore (`--llm-concurrency`).
  - Apply SEARCH/REPLACE diffs and evaluate candidates.
  - Reward prompt strategies based on candidate outcomes.
  - Occasionally create new prompt variants from top-performing prompts.
  - Prune population with top-k + one diversity slot.
- Persist artifacts per generation under `runs/<run_name>/`.
- Use `python mvp/cli.py render-report --run-dir runs/<run_name>` to produce `demo_report.html`
  with:
  - baseline implementation section,
  - evolution timeline section,
  - final best implementation section.

## Where Feedback Is Captured
- Candidate-level fields:
  - `metrics`
  - `failure_reasons`
  - `stage_results`
  - `lineage` (`parent_id`, mutator info)
- Prompt fields include:
  - best metrics so far,
  - recent weak-candidate failure reasons,
  - drop notes from pruning,
  - current prompt strategy text.

## How to Run
```bash
cd alphaevolve
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python mvp/cli.py setup-check
python mvp/cli.py dry-run --generations 5
python mvp/cli.py dry-run --generations 5 --parallel-candidates 4 --llm-concurrency 4 --tui
python mvp/cli.py demo-astar --mode mock --generations 10 --run-name demo_astar
python mvp/cli.py demo-astar-progression --tui
python mvp/cli.py demo-astar-gemini-latest --tui
```

Add `GEMINI_API_KEY` and use `--mode gemini`, or add `OPENAI_API_KEY` and use `--mode openai`.
