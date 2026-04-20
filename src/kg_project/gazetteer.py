from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import typer

from kg_project.data_types import EntityMention, GazetteerEntry, read_jsonl, write_jsonl


def _split_aliases(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    values = [item.strip() for item in re.split(r"[|;]", raw) if item.strip()]
    return tuple(dict.fromkeys(values))


@dataclass(slots=True)
class Gazetteer:
    entries: list[GazetteerEntry]

    def find_mentions(self, text: str) -> list[EntityMention]:
        candidates: list[EntityMention] = []
        for entry in self.entries:
            for variant in entry.variants():
                pattern = re.escape(variant)
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    candidates.append(
                        EntityMention(
                            text=text[match.start() : match.end()],
                            start=match.start(),
                            end=match.end(),
                            entity_type=entry.entity_type,
                            canonical_id=entry.canonical_id,
                            source=entry.source,
                        )
                    )

        selected: list[EntityMention] = []
        occupied: list[tuple[int, int]] = []
        for mention in sorted(candidates, key=lambda item: (-(item.end - item.start), item.start)):
            if any(not (mention.end <= start or mention.start >= end) for start, end in occupied):
                continue
            occupied.append((mention.start, mention.end))
            selected.append(mention)
        return sorted(selected, key=lambda item: item.start)


def load_esco_entries(path: Path) -> list[GazetteerEntry]:
    rows: list[GazetteerEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                GazetteerEntry(
                    canonical_id=row["uri"],
                    entity_type=row["entityType"],
                    term=row["preferredLabel"],
                    source="ESCO",
                    description=row.get("description", ""),
                    aliases=_split_aliases(row.get("altLabels")),
                )
            )
    return rows


def load_onet_entries(path: Path) -> list[GazetteerEntry]:
    rows: list[GazetteerEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append(
                GazetteerEntry(
                    canonical_id=f"onet:{row['category']}:{row['code']}",
                    entity_type=row["category"],
                    term=row["label"],
                    source="O*NET",
                    description=row.get("description", ""),
                    aliases=_split_aliases(row.get("aliases")),
                )
            )
    return rows


def build_gazetteer(esco_path: Path, onet_path: Path) -> Gazetteer:
    merged: dict[tuple[str, str], GazetteerEntry] = {}
    for entry in [*load_esco_entries(esco_path), *load_onet_entries(onet_path)]:
        key = (entry.entity_type, entry.term.casefold())
        if key not in merged:
            merged[key] = entry
            continue
        aliases = tuple(dict.fromkeys([*merged[key].aliases, *entry.aliases]))
        merged[key] = GazetteerEntry(
            canonical_id=merged[key].canonical_id,
            entity_type=merged[key].entity_type,
            term=merged[key].term,
            source=merged[key].source,
            description=merged[key].description or entry.description,
            aliases=aliases,
        )
    return Gazetteer(entries=sorted(merged.values(), key=lambda item: (item.entity_type, item.term)))


def write_gazetteer_jsonl(gazetteer: Gazetteer, output_path: Path) -> None:
    write_jsonl(gazetteer.entries, output_path)


def read_gazetteer_jsonl(path: Path) -> Gazetteer:
    entries = [GazetteerEntry(**row) for row in read_jsonl(path)]
    return Gazetteer(entries=entries)


def build_gazetteer_command(
    esco: Path = typer.Option(..., help="Path to the ESCO CSV file."),
    onet: Path = typer.Option(..., help="Path to the O*NET TSV file."),
    output: Path = typer.Option(..., help="Output path for the merged gazetteer JSONL."),
) -> None:
    gazetteer = build_gazetteer(esco_path=esco, onet_path=onet)
    write_gazetteer_jsonl(gazetteer, output)
