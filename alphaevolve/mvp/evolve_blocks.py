from __future__ import annotations

START_MARKER = "# EVOLVE-BLOCK-START"
END_MARKER = "# EVOLVE-BLOCK-END"


class EvolveBlockError(ValueError):
    pass


def find_evolve_blocks(source: str) -> list[tuple[int, int, str]]:
    """Return (start_line, end_line, block_text) for every evolve block."""
    lines = source.splitlines()
    blocks: list[tuple[int, int, str]] = []
    current_start: int | None = None
    current_lines: list[str] = []

    for idx, line in enumerate(lines, start=1):
        if START_MARKER in line:
            if current_start is not None:
                raise EvolveBlockError("Nested EVOLVE-BLOCK-START markers are not allowed.")
            current_start = idx
            current_lines = []
            continue
        if END_MARKER in line:
            if current_start is None:
                raise EvolveBlockError("Found EVOLVE-BLOCK-END without a matching start.")
            blocks.append((current_start, idx, "\n".join(current_lines)))
            current_start = None
            current_lines = []
            continue
        if current_start is not None:
            current_lines.append(line)

    if current_start is not None:
        raise EvolveBlockError("Unclosed EVOLVE-BLOCK-START marker.")
    return blocks


def assert_has_evolve_blocks(source: str) -> None:
    blocks = find_evolve_blocks(source)
    if not blocks:
        raise EvolveBlockError("No evolve blocks found. Add # EVOLVE-BLOCK-START/END markers.")
