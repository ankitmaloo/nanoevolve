from __future__ import annotations

import pytest

from mvp.diff_engine import (
    DiffFormatError,
    SearchBlockNotFoundError,
    apply_search_replace_blocks,
)


def test_apply_search_replace_blocks_success() -> None:
    source = "def f():\n    return 1\n"
    diff = """<<<<<<< SEARCH
    return 1
=======
    return 2
>>>>>>> REPLACE"""

    updated, stats = apply_search_replace_blocks(source, diff)

    assert "return 2" in updated
    assert stats[0]["matches"] == 1


def test_apply_search_replace_blocks_no_blocks() -> None:
    source = "x = 1\n"
    with pytest.raises(DiffFormatError):
        apply_search_replace_blocks(source, "not a diff")


def test_apply_search_replace_blocks_unmatched_search() -> None:
    source = "def g():\n    return 3\n"
    diff = """<<<<<<< SEARCH
    return 999
=======
    return 1
>>>>>>> REPLACE"""

    with pytest.raises(SearchBlockNotFoundError):
        apply_search_replace_blocks(source, diff)
