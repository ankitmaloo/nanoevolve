from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

if sys.path and os.path.realpath(sys.path[0]) == SCRIPT_DIR:
    # Prevent mvp/types.py from shadowing stdlib types when cli.py is run as a script.
    sys.path.pop(0)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse
import json
from pathlib import Path

from mvp.config import RunConfig
from mvp.controller import EvolutionController
from mvp.paper_study import write_paper_study
from mvp.report_html import generate_html_report


def _base_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_with_optional_tui(
    base: Path, cfg: RunConfig, enable_tui: bool, task_dir: Path | None = None
) -> dict[str, object]:
    tui = None
    event_callback = None
    if enable_tui:
        try:
            from mvp.tui import EvolutionLiveTUI
        except Exception as exc:
            raise RuntimeError(f"TUI requested but unavailable: {exc}") from exc
        tui = EvolutionLiveTUI(
            mode=cfg.mode,
            model_name=cfg.model_name,
            total_generations=cfg.generations,
            parallel_candidates=cfg.parallel_candidates,
        )
        event_callback = tui.handle_event
        tui.start()

    try:
        controller = EvolutionController(
            base_dir=base, config=cfg, event_callback=event_callback, task_dir=task_dir
        )
        return controller.run()
    finally:
        if tui is not None:
            tui.stop()


def setup_check() -> int:
    base = _base_dir()
    checks: dict[str, bool] = {}

    checks["alphaevolve.pdf exists"] = (base / "alphaevolve.pdf").exists()
    checks["default seed target exists"] = (base / "mvp/tasks/astar_routing_target.py").exists()
    checks["default mock diffs exist"] = (base / "mvp/mock_diffs/astar_routing_diffs.json").exists()

    try:
        import pypdf  # noqa: F401

        checks["pypdf import"] = True
    except Exception:
        checks["pypdf import"] = False

    try:
        from google import genai  # noqa: F401

        checks["google-genai import"] = True
    except Exception:
        checks["google-genai import"] = False

    try:
        from openai import OpenAI  # noqa: F401

        checks["openai import"] = True
    except Exception:
        checks["openai import"] = False

    try:
        import rich  # noqa: F401

        checks["rich import"] = True
    except Exception:
        checks["rich import"] = False

    all_ok = all(checks.values())
    print(json.dumps({"ok": all_ok, "checks": checks}, indent=2))
    return 0 if all_ok else 1


def run_study_paper(pdf_path: str | None = None, out_path: str | None = None) -> int:
    base = _base_dir()
    pdf = Path(pdf_path).resolve() if pdf_path else (base / "alphaevolve.pdf")
    out = Path(out_path).resolve() if out_path else (base / "PAPER_STUDY.md")
    write_paper_study(pdf, out)
    print(f"wrote {out}")
    return 0


def run_loop(args: argparse.Namespace) -> int:
    base = _base_dir()
    if args.model:
        model_name = args.model
    elif args.mode == "openai":
        model_name = "gpt-5.2"
    else:
        model_name = "gemini-3-flash-lite"

    cfg = RunConfig(
        mode=args.mode,
        model_name=model_name,
        openai_fast_model_name=args.fast_model,
        openai_slow_every=args.slow_every,
        openai_request_timeout_s=args.openai_timeout,
        openai_max_retries=args.openai_retries,
        openai_max_output_tokens=args.openai_max_output_tokens,
        parallel_candidates=args.parallel_candidates,
        llm_concurrency=args.llm_concurrency,
        generations=args.generations,
        inspirations_k=args.inspirations,
        survivor_top_k=args.top_k,
        diversity_slots=args.diversity_slots,
        seed=args.seed,
        run_name=args.run_name,
        seed_program_path=args.seed_program,
        mock_diff_path=args.mock_diffs,
    )
    task_dir = Path(args.task_dir).resolve() if args.task_dir else None
    try:
        summary = _run_with_optional_tui(base, cfg, args.tui, task_dir=task_dir)
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


def dry_run(args: argparse.Namespace) -> int:
    base = _base_dir()
    model_name = args.model or "gpt-5.2"
    cfg = RunConfig(
        mode="mock",
        model_name=model_name,
        openai_fast_model_name=args.fast_model,
        openai_slow_every=args.slow_every,
        openai_request_timeout_s=args.openai_timeout,
        openai_max_retries=args.openai_retries,
        openai_max_output_tokens=args.openai_max_output_tokens,
        parallel_candidates=args.parallel_candidates,
        llm_concurrency=args.llm_concurrency,
        generations=args.generations,
        inspirations_k=args.inspirations,
        survivor_top_k=args.top_k,
        diversity_slots=args.diversity_slots,
        seed=args.seed,
        run_name=args.run_name,
        seed_program_path=args.seed_program,
        mock_diff_path=args.mock_diffs,
    )
    summary = _run_with_optional_tui(base, cfg, args.tui)
    summary["dry_run"] = True
    print(json.dumps(summary, indent=2))
    return 0


def render_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else None
    report_path = generate_html_report(run_dir=run_dir, out_path=out_path)
    print(json.dumps({"ok": True, "run_dir": str(run_dir), "report_path": str(report_path)}, indent=2))
    return 0


def _read_progression_scores(run_dir: Path) -> list[float]:
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return []
    scores: list[float] = []
    for line in events_path.read_text().splitlines():
        payload = json.loads(line)
        if int(payload.get("generation", 0)) <= 0:
            continue
        eval_rel = payload.get("evaluation_file")
        if not eval_rel:
            continue
        eval_path = run_dir / str(eval_rel)
        if not eval_path.exists():
            continue
        evaluation = json.loads(eval_path.read_text())
        score = float(evaluation.get("aggregate_score", -1.0))
        scores.append(score)
    return scores


def demo_astar(args: argparse.Namespace) -> int:
    base = _base_dir()
    if args.model:
        model_name = args.model
    elif args.mode == "openai":
        model_name = "gpt-5.2"
    else:
        model_name = "gemini-3-flash-lite"

    cfg = RunConfig(
        mode=args.mode,
        model_name=model_name,
        openai_fast_model_name=args.fast_model,
        openai_slow_every=args.slow_every,
        openai_request_timeout_s=args.openai_timeout,
        openai_max_retries=args.openai_retries,
        openai_max_output_tokens=args.openai_max_output_tokens,
        parallel_candidates=args.parallel_candidates,
        llm_concurrency=args.llm_concurrency,
        generations=args.generations,
        inspirations_k=args.inspirations,
        survivor_top_k=args.top_k,
        diversity_slots=args.diversity_slots,
        seed=args.seed,
        run_name=args.run_name,
        seed_program_path=args.seed_program,
        mock_diff_path=args.mock_diffs,
    )
    try:
        summary = _run_with_optional_tui(base, cfg, args.tui)
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1

    run_dir = Path(summary["run_dir"]).resolve()
    report_path = generate_html_report(run_dir=run_dir, out_path=None)
    summary["report_path"] = str(report_path)
    summary["demo"] = "astar"
    print(json.dumps(summary, indent=2))
    return 0


def demo_astar_gemini_latest(args: argparse.Namespace) -> int:
    base = _base_dir()
    cfg = RunConfig(
        mode="gemini",
        model_name="gemini-3-flash-latest",
        generations=args.generations,
        inspirations_k=args.inspirations,
        survivor_top_k=args.top_k,
        diversity_slots=args.diversity_slots,
        parallel_candidates=args.parallel_candidates,
        llm_concurrency=args.llm_concurrency,
        seed=args.seed,
        run_name=args.run_name,
        seed_program_path=args.seed_program,
        mock_diff_path=args.mock_diffs,
    )
    try:
        summary = _run_with_optional_tui(base, cfg, args.tui)
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1

    run_dir = Path(summary["run_dir"]).resolve()
    report_path = generate_html_report(run_dir=run_dir, out_path=None)
    summary["report_path"] = str(report_path)
    summary["demo"] = "astar_gemini_latest"
    print(json.dumps(summary, indent=2))
    return 0


def demo_astar_progression(args: argparse.Namespace) -> int:
    base = _base_dir()
    cfg = RunConfig(
        mode="mock",
        model_name="gemini-3-flash-lite",
        generations=args.generations,
        inspirations_k=1,
        survivor_top_k=1,
        diversity_slots=0,
        parallel_candidates=1,
        llm_concurrency=1,
        seed=args.seed,
        run_name=args.run_name,
        seed_program_path=args.seed_program,
        mock_diff_path=args.mock_diffs,
    )

    summary = _run_with_optional_tui(base, cfg, args.tui)
    run_dir = Path(summary["run_dir"]).resolve()
    report_path = generate_html_report(run_dir=run_dir, out_path=None)
    progression_scores = _read_progression_scores(run_dir)
    summary["report_path"] = str(report_path)
    summary["demo"] = "astar_progression"
    summary["progression_scores"] = progression_scores
    summary["strictly_improving"] = all(
        next_score > prev_score for prev_score, next_score in zip(progression_scores, progression_scores[1:])
    )
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaEvolve MVP CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup-check", help="Validate local setup and required files")

    study = sub.add_parser("study-paper", help="Extract implementation notes from alphaevolve.pdf")
    study.add_argument("--pdf", default=None, help="Path to PDF (default: alphaevolve/alphaevolve.pdf)")
    study.add_argument("--out", default=None, help="Output markdown path (default: alphaevolve/PAPER_STUDY.md)")

    run = sub.add_parser("run", help="Run evolutionary MVP loop")
    run.add_argument("--mode", choices=["mock", "gemini", "openai"], default="mock")
    run.add_argument("--model", default=None)
    run.add_argument("--fast-model", default="gpt-5.2-mini")
    run.add_argument("--slow-every", type=int, default=4, help="For OpenAI mode: use slow/base model every N calls")
    run.add_argument("--openai-timeout", type=float, default=45.0)
    run.add_argument("--openai-retries", type=int, default=1)
    run.add_argument("--openai-max-output-tokens", type=int, default=2500)
    run.add_argument("--parallel-candidates", type=int, default=3)
    run.add_argument("--llm-concurrency", type=int, default=3)
    run.add_argument("--generations", type=int, default=10)
    run.add_argument("--inspirations", type=int, default=2)
    run.add_argument("--top-k", type=int, default=4)
    run.add_argument("--diversity-slots", type=int, default=1)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--run-name", default=None)
    run.add_argument("--seed-program", default="mvp/tasks/astar_routing_target.py")
    run.add_argument("--mock-diffs", default="mvp/mock_diffs/astar_routing_diffs.json")
    run.add_argument("--task-dir", default=None, help="Path to a folder with task.json (generic target evolution)")
    run.add_argument("--tui", action="store_true", help="Show a live terminal dashboard")

    dry = sub.add_parser("dry-run", help="Run full loop in mock mode to validate setup before live API runs")
    dry.add_argument("--model", default=None)
    dry.add_argument("--fast-model", default="gpt-5.2-mini")
    dry.add_argument("--slow-every", type=int, default=4)
    dry.add_argument("--openai-timeout", type=float, default=45.0)
    dry.add_argument("--openai-retries", type=int, default=1)
    dry.add_argument("--openai-max-output-tokens", type=int, default=2500)
    dry.add_argument("--parallel-candidates", type=int, default=3)
    dry.add_argument("--llm-concurrency", type=int, default=3)
    dry.add_argument("--generations", type=int, default=5)
    dry.add_argument("--inspirations", type=int, default=2)
    dry.add_argument("--top-k", type=int, default=4)
    dry.add_argument("--diversity-slots", type=int, default=1)
    dry.add_argument("--seed", type=int, default=7)
    dry.add_argument("--run-name", default=None)
    dry.add_argument("--seed-program", default="mvp/tasks/astar_routing_target.py")
    dry.add_argument("--mock-diffs", default="mvp/mock_diffs/astar_routing_diffs.json")
    dry.add_argument("--tui", action="store_true", help="Show a live terminal dashboard")

    report = sub.add_parser("render-report", help="Generate HTML report from an existing run directory")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--out", default=None, help="Output HTML file (default: <run-dir>/demo_report.html)")

    demo = sub.add_parser("demo-astar", help="Run A* evolution and emit an HTML demo report")
    demo.add_argument("--mode", choices=["mock", "gemini", "openai"], default="mock")
    demo.add_argument("--model", default=None)
    demo.add_argument("--fast-model", default="gpt-5.2-mini")
    demo.add_argument("--slow-every", type=int, default=4)
    demo.add_argument("--openai-timeout", type=float, default=45.0)
    demo.add_argument("--openai-retries", type=int, default=1)
    demo.add_argument("--openai-max-output-tokens", type=int, default=2500)
    demo.add_argument("--parallel-candidates", type=int, default=3)
    demo.add_argument("--llm-concurrency", type=int, default=3)
    demo.add_argument("--generations", type=int, default=10)
    demo.add_argument("--inspirations", type=int, default=2)
    demo.add_argument("--top-k", type=int, default=4)
    demo.add_argument("--diversity-slots", type=int, default=1)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--run-name", default=None)
    demo.add_argument("--seed-program", default="mvp/tasks/astar_routing_target.py")
    demo.add_argument("--mock-diffs", default="mvp/mock_diffs/astar_routing_diffs.json")
    demo.add_argument("--tui", action="store_true", help="Show a live terminal dashboard")

    prog = sub.add_parser(
        "demo-astar-progression",
        help="Deterministic 4-step A* progression demo (mock diffs + live TUI optional)",
    )
    prog.add_argument("--generations", type=int, default=4)
    prog.add_argument("--seed", type=int, default=7)
    prog.add_argument("--run-name", default="demo_astar_progression")
    prog.add_argument("--seed-program", default="mvp/tasks/astar_routing_target.py")
    prog.add_argument("--mock-diffs", default="mvp/mock_diffs/astar_progression_diffs.json")
    prog.add_argument("--tui", action="store_true", help="Show a live terminal dashboard")

    gem = sub.add_parser(
        "demo-astar-gemini-latest",
        help="Run hardened 4-gen A* demo on gemini-3-flash-latest",
    )
    gem.add_argument("--generations", type=int, default=4)
    gem.add_argument("--parallel-candidates", type=int, default=2)
    gem.add_argument("--llm-concurrency", type=int, default=2)
    gem.add_argument("--inspirations", type=int, default=2)
    gem.add_argument("--top-k", type=int, default=1)
    gem.add_argument("--diversity-slots", type=int, default=0)
    gem.add_argument("--seed", type=int, default=17)
    gem.add_argument("--run-name", default="gemini_flash_latest_harder_g4_seed17")
    gem.add_argument("--seed-program", default="mvp/tasks/astar_routing_target.py")
    gem.add_argument("--mock-diffs", default="mvp/mock_diffs/astar_routing_diffs.json")
    gem.add_argument("--tui", action="store_true", help="Show a live terminal dashboard")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "setup-check":
        return setup_check()
    if args.command == "study-paper":
        return run_study_paper(args.pdf, args.out)
    if args.command == "run":
        return run_loop(args)
    if args.command == "dry-run":
        return dry_run(args)
    if args.command == "render-report":
        return render_report(args)
    if args.command == "demo-astar":
        return demo_astar(args)
    if args.command == "demo-astar-progression":
        return demo_astar_progression(args)
    if args.command == "demo-astar-gemini-latest":
        return demo_astar_gemini_latest(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
