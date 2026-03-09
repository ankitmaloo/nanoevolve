from __future__ import annotations

import re
from dataclasses import dataclass


class DiffFormatError(ValueError):
    pass


class SearchBlockNotFoundError(ValueError):
    pass


BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)


@dataclass
class ParsedBlock:
    search: str
    replace: str


def parse_search_replace_blocks(diff_text: str) -> list[ParsedBlock]:
    blocks = [ParsedBlock(search=m.group(1), replace=m.group(2)) for m in BLOCK_RE.finditer(diff_text)]
    if not blocks:
        raise DiffFormatError(
            "No valid SEARCH/REPLACE blocks found. Expected <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE"
        )
    return blocks


def apply_search_replace_blocks(source: str, diff_text: str) -> tuple[str, list[dict[str, int]]]:
    updated = source
    blocks = parse_search_replace_blocks(diff_text)
    stats: list[dict[str, int]] = []

    for block in blocks:
        matches = updated.count(block.search)
        if matches == 0:
            snippet = block.search.splitlines()[0] if block.search.splitlines() else "<empty>"
            raise SearchBlockNotFoundError(f"SEARCH block not found in source: {snippet!r}")
        updated = updated.replace(block.search, block.replace)
        stats.append({"matches": matches, "search_len": len(block.search), "replace_len": len(block.replace)})

    return updated, stats
