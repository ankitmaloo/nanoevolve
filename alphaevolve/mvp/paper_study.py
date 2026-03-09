from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def _snippet(full_text: str, keyword: str, window: int = 550) -> str:
    idx = full_text.lower().find(keyword.lower())
    if idx == -1:
        return f"Keyword not found: {keyword}"
    start = max(0, idx - 120)
    end = min(len(full_text), idx + window)
    chunk = full_text[start:end].replace("\n", " ")
    return " ".join(chunk.split())


def write_paper_study(pdf_path: Path, out_path: Path) -> Path:
    reader = PdfReader(str(pdf_path))
    full_text = "\n\n".join((page.extract_text() or "") for page in reader.pages)

    sections = {
        "EVOLVE Blocks and API": _snippet(full_text, "EVOLVE-BLOCK-START"),
        "Prompt Sampling": _snippet(full_text, "2.2. Prompt sampling"),
        "SEARCH/REPLACE Format": _snippet(full_text, "Output format"),
        "Evaluation Cascade": _snippet(full_text, "Evaluation cascade"),
        "Multi-Metric Optimization": _snippet(full_text, "Multiple scores"),
        "Evolutionary Database": _snippet(full_text, "2.5. Evolution"),
        "Controller Loop": _snippet(full_text, "parent_program, inspirations = database.sample()"),
    }

    lines: list[str] = []
    lines.append("# AlphaEvolve Paper Study (MVP-Oriented)\n")
    lines.append("This note was generated from `alphaevolve/alphaevolve.pdf` for implementation grounding.\n")
    lines.append("## MVP Extraction Summary")
    lines.append("- Evolvable code is marked with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`.")
    lines.append("- Mutations are emitted as `SEARCH/REPLACE` diffs.")
    lines.append("- Evaluators return scalar metrics and can use cascades.")
    lines.append("- Evolution reuses prior high-quality ideas with exploration/exploitation balance.")
    lines.append("- Main loop: sample parent + inspirations -> mutate -> evaluate -> add back to DB.\n")

    lines.append("## Evidence Snippets")
    for title, snippet in sections.items():
        lines.append(f"### {title}")
        lines.append(snippet)
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path
