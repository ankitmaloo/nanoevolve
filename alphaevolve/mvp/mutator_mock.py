from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mvp.types import Candidate, DiffProposal, FeedbackBundle


class MockMutator:
    """Replay predefined diff proposals for setup/testing without API access."""

    def __init__(self, diff_path: Path) -> None:
        data = json.loads(diff_path.read_text())
        if not isinstance(data, list) or not data:
            raise ValueError("Mock diff file must be a non-empty JSON list.")
        self._entries = data
        self._idx = 0

    async def mutate(
        self,
        parent: Candidate,
        inspirations: list[Candidate],
        feedback: FeedbackBundle,
        prompt: str,
    ) -> DiffProposal:
        entry = self._entries[self._idx % len(self._entries)]
        self._idx += 1

        if isinstance(entry, str):
            raw_diff = entry
            name = f"mock_{self._idx}"
        elif isinstance(entry, dict):
            raw_diff = str(entry["diff"])
            name = str(entry.get("name", f"mock_{self._idx}"))
        else:
            raise ValueError("Mock diff entries must be strings or objects with a 'diff' key.")

        return DiffProposal(
            raw_diff=raw_diff,
            model="mock-replay",
            metadata={
                "entry_name": name,
                "entry_index": self._idx - 1,
                "parent_id": parent.id,
                "inspiration_ids": [c.id for c in inspirations],
                "feedback_failure_count": len(feedback.weak_failure_reasons),
                "prompt_chars": len(prompt),
            },
        )
