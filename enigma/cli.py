#!/usr/bin/env python3
"""Enigma CLI — evolve any program through disciplined evolutionary search.

Usage:
    # Run evolution on a target directory
    python -m enigma evolve /path/to/target --provider anthropic --model claude-sonnet-4-20250514

    # Resume from a previous run
    python -m enigma evolve /path/to/target --resume /path/to/run_dir

    # Initialize a new target directory with task.json template
    python -m enigma init /path/to/new_target

    # Inspect search memory from a completed or in-progress run
    python -m enigma inspect /path/to/run_dir

    # Cold-start handoff: summarize what another LLM needs to know
    python -m enigma handoff /path/to/run_dir
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from enigma.config import ProviderConfig, SearchConfig, TaskConfig
from enigma.context import generate_task_context
from enigma.controller import EnigmaController
from enigma.memory import SearchMemory
from enigma.mutator import LLMMutator
from enigma.prompts import build_cold_start_prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="enigma",
        description="Enigma: Universal program evolution engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── evolve ──────────────────────────────────────────────────
    evolve_parser = subparsers.add_parser("evolve", help="Run evolutionary search")
    evolve_parser.add_argument("task_dir", type=Path, help="Path to target directory with task.json")
    evolve_parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "gemini"])
    evolve_parser.add_argument("--model", default="claude-sonnet-4-20250514")
    evolve_parser.add_argument("--fast-model", default=None, help="Fast model for screening")
    evolve_parser.add_argument("--loops", type=int, default=10, help="Max doctrine loops")
    evolve_parser.add_argument("--portfolio-size", type=int, default=5)
    evolve_parser.add_argument("--hypotheses", type=int, default=10, help="Hypotheses per loop")
    evolve_parser.add_argument("--top-k", type=int, default=6, help="Survivor pool size")
    evolve_parser.add_argument("--diversity-slots", type=int, default=2)
    evolve_parser.add_argument("--seed", type=int, default=42)
    evolve_parser.add_argument("--parallel", type=int, default=4, help="Parallel candidates")
    evolve_parser.add_argument("--temperature", type=float, default=0.7)
    evolve_parser.add_argument("--max-tokens", type=int, default=4096)
    evolve_parser.add_argument("--timeout", type=int, default=120, help="LLM request timeout (s)")
    evolve_parser.add_argument("--run-dir", type=Path, default=None, help="Custom run directory")
    evolve_parser.add_argument("--resume", type=Path, default=None, help="Resume from run dir")
    evolve_parser.add_argument("--verbose", "-v", action="store_true")

    # ── init ────────────────────────────────────────────────────
    init_parser = subparsers.add_parser("init", help="Initialize a target directory")
    init_parser.add_argument("target_dir", type=Path)
    init_parser.add_argument("--name", default="my_target", help="Task name")
    init_parser.add_argument("--seed-file", default="program.py", help="Seed file name")
    init_parser.add_argument("--eval-command", default=None, help="Evaluation command")

    # ── inspect ─────────────────────────────────────────────────
    inspect_parser = subparsers.add_parser("inspect", help="Inspect search memory")
    inspect_parser.add_argument("run_dir", type=Path)
    inspect_parser.add_argument("--full", action="store_true", help="Show full memory dump")

    # ── handoff ─────────────────────────────────────────────────
    handoff_parser = subparsers.add_parser("handoff", help="Generate cold-start handoff summary")
    handoff_parser.add_argument("run_dir", type=Path)
    handoff_parser.add_argument("--task-dir", type=Path, default=None)

    args = parser.parse_args()

    if args.command == "evolve":
        _cmd_evolve(args)
    elif args.command == "init":
        _cmd_init(args)
    elif args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "handoff":
        _cmd_handoff(args)


def _cmd_evolve(args: argparse.Namespace) -> None:
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    search_config = SearchConfig(
        max_loops=args.loops,
        portfolio_size=args.portfolio_size,
        hypotheses_per_loop=args.hypotheses,
        survivor_top_k=args.top_k,
        diversity_slots=args.diversity_slots,
        seed=args.seed,
        parallel_candidates=args.parallel,
        provider=args.provider,
        model_name=args.model,
        fast_model_name=args.fast_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_timeout_s=args.timeout,
    )

    provider_config = ProviderConfig(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_timeout_s=args.timeout,
    )

    run_dir = args.resume or args.run_dir

    def event_callback(event: dict) -> None:
        et = event.get("event_type", "")
        if et == "loop_start":
            print(f"\n{'='*60}")
            print(f"  Loop {event.get('loop')}")
            print(f"{'='*60}")
        elif et == "stage":
            print(f"  >> Stage: {event.get('stage')}")
        elif et == "slot_evaluated":
            valid = event.get("valid")
            score = event.get("score", -1)
            marker = "OK" if valid else "FAIL"
            print(f"  [{marker}] {event.get('slot')}: score={score:.4f} (cand={event.get('candidate')})")
        elif et == "slot_failed":
            print(f"  [ERR] {event.get('slot')}: {event.get('error', '')[:80]}")
        elif et == "loop_error":
            print(f"  [LOOP ERROR] {event.get('error', '')[:120]}")
        elif et == "run_end":
            print(f"\n{'='*60}")
            print(f"  DONE — Best: {event.get('best_candidate_id')} "
                  f"score={event.get('best_score', 0):.4f}")
            print(f"  Loops: {event.get('loops_completed')}, "
                  f"Candidates: {event.get('total_candidates')}")
            print(f"  Run dir: {event.get('run_dir')}")
            print(f"{'='*60}")

    controller = EnigmaController(
        task_dir=args.task_dir,
        search_config=search_config,
        provider_config=provider_config,
        run_dir=run_dir,
        event_callback=event_callback,
    )

    summary = controller.run()
    print(f"\nSummary: {json.dumps(summary, indent=2)}")


def _cmd_init(args: argparse.Namespace) -> None:
    target_dir = args.target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create task.json
    task = {
        "name": args.name,
        "description": f"Optimize {args.name} for better performance.",
        "seed_file": args.seed_file,
        "mutable_files": [args.seed_file],
        "context_files": ["CONTEXT.md"],
        "eval_command": args.eval_command or f"python eval.py {{candidate_file}}",
        "eval_mode": "command",
        "metric_keys": ["aggregate_score"],
        "maximize": True,
        "primary_metric": "aggregate_score",
        "eval_timeout": 120,
        "language": "auto",
        "hardware": "",
        "hard_constraints": [],
        "benchmark_slices": ["typical_case", "edge_case"],
    }
    (target_dir / "task.json").write_text(json.dumps(task, indent=2))

    # Create seed program template
    seed_path = target_dir / args.seed_file
    if not seed_path.exists():
        seed_path.write_text(f"""# {args.name} — Seed Program
# Add EVOLVE-BLOCK markers around the code you want Enigma to optimize.

def main():
    # EVOLVE-BLOCK-START
    # Your optimizable code goes here.
    # Enigma will mutate only the code between these markers.
    result = naive_implementation()
    # EVOLVE-BLOCK-END
    return result


def naive_implementation():
    return 0


if __name__ == "__main__":
    print(main())
""")

    # Create eval.py template
    eval_path = target_dir / "eval.py"
    if not eval_path.exists():
        eval_path.write_text("""#!/usr/bin/env python3
\"\"\"Evaluator script — Enigma calls this to score each candidate.

Usage: python eval.py <candidate_file>

Must print a JSON object to stdout:
{
    "valid": true,
    "aggregate_score": 42.0,
    "metrics": {"throughput": 42.0, "correctness": 1.0},
    "failure_reasons": []
}
\"\"\"
import json
import sys
import importlib.util
from pathlib import Path


def evaluate(candidate_path: str) -> dict:
    # Load the candidate module
    spec = importlib.util.spec_from_file_location("candidate", candidate_path)
    module = importlib.util.module_from_spec(spec)

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return {
            "valid": False,
            "aggregate_score": -1.0,
            "metrics": {},
            "failure_reasons": [f"Import error: {e}"],
        }

    try:
        result = module.main()
        # TODO: Replace with your actual scoring logic
        score = float(result) if result is not None else 0.0
        return {
            "valid": True,
            "aggregate_score": score,
            "metrics": {"score": score},
            "failure_reasons": [],
        }
    except Exception as e:
        return {
            "valid": False,
            "aggregate_score": -1.0,
            "metrics": {},
            "failure_reasons": [f"Runtime error: {e}"],
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"valid": False, "aggregate_score": -1.0, "failure_reasons": ["No file"]}))
        sys.exit(1)

    result = evaluate(sys.argv[1])
    print(json.dumps(result))
""")

    # Create CONTEXT.md template
    ctx_path = target_dir / "CONTEXT.md"
    if not ctx_path.exists():
        ctx_path.write_text(f"""# {args.name} — Optimization Context

## What This Program Does
Describe the program's purpose and how it works.

## Performance Bottlenecks
- List known or suspected bottlenecks.

## Optimization Strategies To Try
- Strategy 1
- Strategy 2

## Constraints
- List hard constraints the optimizer must respect.
""")

    print(f"Initialized target at: {target_dir}")
    print(f"  task.json  - Edit this to configure your evolution target")
    print(f"  {args.seed_file} - Add your code with EVOLVE-BLOCK markers")
    print(f"  eval.py    - Implement your evaluation/scoring logic")
    print(f"  CONTEXT.md - Add domain knowledge for the LLM")
    print(f"\nRun: python -m enigma evolve {target_dir}")


def _cmd_inspect(args: argparse.Namespace) -> None:
    memory = SearchMemory(args.run_dir / "memory")

    print(f"Loop: {memory.current_loop}")
    print(f"Hypotheses: {len(memory.hypotheses)} "
          f"(active: {len(memory.active_hypotheses())}, "
          f"killed: {len(memory.killed_hypotheses())})")
    print(f"Surfaces: {len(memory.surfaces)} "
          f"(unexplored: {len(memory.unexplored_surfaces())})")
    print(f"Portfolio slots: {len(memory.portfolio)}")
    print(f"Negative knowledge: {len(memory.negative_knowledge)}")
    print(f"Neighborhoods: {len(memory.mutation_ledger)} "
          f"(active: {len(memory.active_neighborhoods())}, "
          f"retired: {len(memory.retired_neighborhoods())})")
    print(f"Loops logged: {len(memory.loop_log)}")

    if args.full:
        print(f"\n{'='*60}")
        print(memory.export_for_prompt(max_tokens=10000))

    last = memory.last_loop()
    if last:
        print(f"\nLast loop ({last.loop}):")
        print(f"  Best: {last.best_candidate}")
        print(f"  Score movement: {last.score_movement}")
        print(f"  Next focus: {last.next_loop_focus}")


def _cmd_handoff(args: argparse.Namespace) -> None:
    memory = SearchMemory(args.run_dir / "memory")
    memory_export = memory.export_for_prompt(max_tokens=8000)

    task_dir = args.task_dir
    if task_dir is None:
        # Try to find task_dir from the run_dir parent
        task_dir = args.run_dir.parent.parent
        if not (task_dir / "task.json").exists():
            print("Cannot find task.json. Use --task-dir to specify the target directory.")
            sys.exit(1)

    task_config = TaskConfig.load(task_dir)
    code_context = generate_task_context(task_config, task_dir)

    prompt = build_cold_start_prompt(memory_export, code_context)
    print(prompt)


if __name__ == "__main__":
    main()
