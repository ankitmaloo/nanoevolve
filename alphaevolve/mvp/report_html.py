from __future__ import annotations

import html
import json
from pathlib import Path


def _load_events(events_path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def _safe_read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text()


def _build_svg(scores: list[float], best_scores: list[float], width: int = 920, height: int = 260) -> str:
    if not scores:
        return ""

    valid_scores = [s for s in scores if isinstance(s, (int, float))]
    if not valid_scores:
        valid_scores = [0.0]

    min_s = min(valid_scores)
    max_s = max(valid_scores)
    if max_s - min_s < 1e-9:
        max_s = min_s + 1.0

    pad = 30
    n = len(scores)

    def to_xy(i: int, s: float) -> tuple[float, float]:
        x = pad + (i / max(1, n - 1)) * (width - 2 * pad)
        y = pad + (1.0 - (s - min_s) / (max_s - min_s)) * (height - 2 * pad)
        return x, y

    line_pts = []
    best_pts = []
    for i, (s, b) in enumerate(zip(scores, best_scores)):
        x1, y1 = to_xy(i, s)
        x2, y2 = to_xy(i, b)
        line_pts.append(f"{x1:.1f},{y1:.1f}")
        best_pts.append(f"{x2:.1f},{y2:.1f}")

    return f"""
<svg viewBox=\"0 0 {width} {height}\" width=\"100%\" height=\"{height}\" role=\"img\" aria-label=\"Evolution score chart\">
  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#0f172a\" rx=\"8\"/>
  <polyline fill=\"none\" stroke=\"#60a5fa\" stroke-width=\"2\" points=\"{' '.join(line_pts)}\"/>
  <polyline fill=\"none\" stroke=\"#34d399\" stroke-width=\"2\" points=\"{' '.join(best_pts)}\"/>
  <text x=\"{pad}\" y=\"20\" fill=\"#cbd5e1\" font-size=\"12\">candidate score (blue), best-so-far (green)</text>
  <text x=\"{pad}\" y=\"{height-10}\" fill=\"#94a3b8\" font-size=\"11\">gen 0..{n-1} | score range {min_s:.4f}..{max_s:.4f}</text>
</svg>
"""


def _build_rows(run_dir: Path, events: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[float], list[float]]:
    rows: list[dict[str, object]] = []
    scores: list[float] = []
    best_scores: list[float] = []
    current_best = -10**9

    for event in events:
        gen = int(event.get("generation", 0))
        if gen == 0:
            score = float(event.get("aggregate_score", -1.0))
            valid = bool(event.get("valid", False))
            cid = str(event.get("candidate_id", "seed"))
            metric_key = "seed"
            prompt_id = "seed"
            prompt_reward = score
            failure = ""
        else:
            cid = str(event.get("child_id", "unknown"))
            metric_key = str(event.get("metric_key", "aggregate_score"))
            prompt_id = str(event.get("prompt_id", "n/a"))
            prompt_reward = float(event.get("prompt_reward", -1.0))
            eval_rel = str(event.get("evaluation_file", ""))
            eval_path = run_dir / eval_rel
            eval_payload = json.loads(_safe_read(eval_path) or "{}")
            score = float(eval_payload.get("aggregate_score", -1.0))
            valid = bool(eval_payload.get("valid", False))
            failure_reasons = eval_payload.get("failure_reasons", [])
            failure = "; ".join(failure_reasons[:1]) if failure_reasons else ""

        current_best = max(current_best, score)
        scores.append(score)
        best_scores.append(current_best)

        rows.append(
            {
                "generation": gen,
                "candidate_id": cid,
                "metric_key": metric_key,
                "aggregate_score": score,
                "valid": valid,
                "prompt_id": prompt_id,
                "prompt_reward": prompt_reward,
                "failure": failure,
            }
        )

    return rows, scores, best_scores


def generate_html_report(run_dir: Path, out_path: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    if out_path is None:
        out_path = run_dir / "demo_report.html"

    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"
    seed_path = run_dir / "seed_program.py"
    best_path = run_dir / "best_program.py"

    events = _load_events(events_path)
    summary = json.loads(_safe_read(summary_path) or "{}")
    rows, scores, best_scores = _build_rows(run_dir, events)

    seed_code = _safe_read(seed_path)
    best_code = _safe_read(best_path)
    svg = _build_svg(scores, best_scores)

    top_prompts = summary.get("top_prompts", [])
    top_prompts_html = "".join(
        f"<tr><td>{html.escape(str(p.get('id')))}</td><td>{float(p.get('mean_reward', 0.0)):.4f}</td><td>{int(p.get('uses', 0))}</td><td>{html.escape(str(p.get('text', '')))}</td></tr>"
        for p in top_prompts
    )
    if not top_prompts_html:
        top_prompts_html = "<tr><td colspan='4'>No prompt stats available</td></tr>"

    row_html = "".join(
        f"<tr><td>{r['generation']}</td><td>{html.escape(str(r['candidate_id']))}</td><td>{html.escape(str(r['metric_key']))}</td><td>{float(r['aggregate_score']):.4f}</td><td>{'yes' if r['valid'] else 'no'}</td><td>{html.escape(str(r['prompt_id']))}</td><td>{float(r['prompt_reward']):.4f}</td><td>{html.escape(str(r['failure']))}</td></tr>"
        for r in rows
    )

    html_doc = f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>A* Evolution Demo Report</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #111827; }}
section {{ margin: 22px 0; }}
pre {{ background: #0b1020; color: #e5e7eb; padding: 14px; border-radius: 8px; overflow-x: auto; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #d1d5db; padding: 6px; text-align: left; vertical-align: top; }}
th {{ background: #f3f4f6; }}
.small {{ color: #4b5563; font-size: 13px; }}
.kpi {{ display: inline-block; margin-right: 14px; padding: 8px 10px; background: #eef2ff; border-radius: 8px; }}
</style>
</head>
<body>
<h1>A* Evolution Demo</h1>
<p class=\"small\">Run dir: {html.escape(str(run_dir))}</p>

<section>
<h2>1) Basic Implementation (Baseline)</h2>
<p class=\"small\">This is the starting A* scoring program before evolution.</p>
<pre><code>{html.escape(seed_code)}</code></pre>
</section>

<section>
<h2>2) Evolution Over Generations</h2>
<div class=\"kpi\">Mode: <b>{html.escape(str(summary.get('mode', 'unknown')))}</b></div>
<div class=\"kpi\">Model: <b>{html.escape(str(summary.get('model_name', 'unknown')))}</b></div>
<div class=\"kpi\">Generations: <b>{int(summary.get('generations', 0))}</b></div>
<div class=\"kpi\">Best Score: <b>{float(summary.get('best_aggregate_score', -1.0)):.4f}</b></div>
{svg}
<h3>Generation Log</h3>
<table>
<thead><tr><th>gen</th><th>candidate</th><th>selection metric</th><th>score</th><th>valid</th><th>prompt id</th><th>prompt reward</th><th>failure sample</th></tr></thead>
<tbody>
{row_html}
</tbody>
</table>

<h3>Top Prompt Strategies</h3>
<table>
<thead><tr><th>prompt id</th><th>mean reward</th><th>uses</th><th>text</th></tr></thead>
<tbody>
{top_prompts_html}
</tbody>
</table>
</section>

<section>
<h2>3) Final Implementation (Best Candidate)</h2>
<p class=\"small\">Best candidate id: <b>{html.escape(str(summary.get('best_candidate_id', 'unknown')))}</b></p>
<pre><code>{html.escape(best_code)}</code></pre>
</section>
</body>
</html>
"""

    out_path.write_text(html_doc)
    return out_path
