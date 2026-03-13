from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .types import CandidateRecord


class SearchArchive:
    def __init__(self) -> None:
        self._records: list[CandidateRecord] = []
        self._by_id: dict[str, CandidateRecord] = {}

    @property
    def records(self) -> list[CandidateRecord]:
        return list(self._records)

    def add(self, record: CandidateRecord) -> None:
        self._records.append(record)
        self._by_id[record.id] = record

    def get(self, candidate_id: str) -> CandidateRecord:
        return self._by_id[candidate_id]

    def get_by_id(self, candidate_id: str | None) -> CandidateRecord | None:
        if candidate_id is None:
            return None
        return self._by_id.get(candidate_id)

    def survivors(self) -> list[CandidateRecord]:
        return [record for record in self._records if record.status in {"baseline", "survivor", "winner"}]

    def best(self) -> CandidateRecord:
        ranked = [record for record in self._records if record.primary_outcome.valid]
        if not ranked:
            raise ValueError("No valid candidates in archive.")
        return max(ranked, key=lambda item: item.score)

    def prune(self, survivor_ids: set[str]) -> None:
        for record in self._records:
            if record.id not in survivor_ids and record.status not in {"baseline", "winner"}:
                record.status = "rejected"

    def persist(self, run_dir: Path) -> None:
        archive_path = run_dir / "archive.jsonl"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(asdict(record)) + "\n")

