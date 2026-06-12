from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class JobPosting:
    id: str
    lang: str
    title: str
    description: str


@dataclass(slots=True)
class GazetteerEntry:
    canonical_id: str
    entity_type: str
    term: str
    source: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def variants(self) -> tuple[str, ...]:
        values = [self.term, *self.aliases]
        return tuple(value for value in values if value)


@dataclass(slots=True)
class EntityMention:
    text: str
    start: int
    end: int
    entity_type: str
    canonical_id: str
    source: str


@dataclass(slots=True)
class NerExample:
    id: str
    lang: str
    text: str
    tokens: list[str]
    labels: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def write_jsonl(records: Iterable[dict[str, Any] | JobPosting | GazetteerEntry | NerExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "__dataclass_fields__"):
                payload = asdict(record)
            else:
                payload = record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
