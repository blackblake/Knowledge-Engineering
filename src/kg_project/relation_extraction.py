from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from kg_project.data_types import NerExample


_RELATION_MAP = {
    "SKL": "REQUIRES_SKILL",
    "KNW": "REQUIRES_KNOWLEDGE",
    "ABL": "REQUIRES_ABILITY",
    "TSK": "HAS_TASK",
    "TOL": "USES_TOOL",
    "QLF": "REQUIRES_QUALIFICATION",
}

_PROFICIENCY_RULES: list[tuple[str, str]] = [
    ("expert", "expert"),
    ("proficient", "proficient"),
    ("experienced", "proficient"),
    ("familiar", "familiar"),
    ("basic", "basic"),
    ("advanced", "expert"),
    ("精通", "expert"),
    ("熟练", "proficient"),
    ("熟悉", "familiar"),
    ("了解", "basic"),
]

_REQUIRED_HINTS = ["must", "required", "need", "needs", "require", "要求", "必须", "需", "需要"]
_OPTIONAL_HINTS = ["preferred", "nice to have", "plus", "加分", "优先", "可选"]

_YEAR_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(years?|年)")


@dataclass(frozen=True, slots=True)
class EntitySpan:
    text: str
    entity_type: str
    start_token: int
    end_token: int
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class RelationRecord:
    relation: str
    head: EntitySpan
    tail: EntitySpan
    essential: bool
    evidence_count: int
    proficiency_level: str | None
    experience_years: float | None
    sentence_id: str

    def to_dict(self) -> dict:
        return {
            "relation": self.relation,
            "head": {
                "text": self.head.text,
                "type": self.head.entity_type,
                "start_char": self.head.start_char,
                "end_char": self.head.end_char,
            },
            "tail": {
                "text": self.tail.text,
                "type": self.tail.entity_type,
                "start_char": self.tail.start_char,
                "end_char": self.tail.end_char,
            },
            "attributes": {
                "essential": self.essential,
                "evidence_count": self.evidence_count,
                "proficiency_level": self.proficiency_level,
                "experience_years": self.experience_years,
            },
            "sentence_id": self.sentence_id,
        }


def _token_offsets(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start == -1:
            start = cursor
        end = start + len(token)
        offsets.append((start, end))
        cursor = end
    return offsets


def extract_entity_spans(example: NerExample) -> list[EntitySpan]:
    offsets = _token_offsets(example.text, example.tokens)
    spans: list[EntitySpan] = []
    current_type = ""
    current_start = -1
    current_end = -1

    for idx, label in enumerate(example.labels):
        if label == "O":
            if current_type:
                start_char = offsets[current_start][0]
                end_char = offsets[current_end][1]
                spans.append(
                    EntitySpan(
                        text=example.text[start_char:end_char],
                        entity_type=current_type,
                        start_token=current_start,
                        end_token=current_end,
                        start_char=start_char,
                        end_char=end_char,
                    )
                )
                current_type = ""
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or entity_type != current_type:
            if current_type:
                start_char = offsets[current_start][0]
                end_char = offsets[current_end][1]
                spans.append(
                    EntitySpan(
                        text=example.text[start_char:end_char],
                        entity_type=current_type,
                        start_token=current_start,
                        end_token=current_end,
                        start_char=start_char,
                        end_char=end_char,
                    )
                )
            current_type = entity_type
            current_start = idx
            current_end = idx
        else:
            current_end = idx

    if current_type:
        start_char = offsets[current_start][0]
        end_char = offsets[current_end][1]
        spans.append(
            EntitySpan(
                text=example.text[start_char:end_char],
                entity_type=current_type,
                start_token=current_start,
                end_token=current_end,
                start_char=start_char,
                end_char=end_char,
            )
        )
    return spans


def _context_window(text: str, start: int, end: int, window: int = 30) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right]


def _infer_proficiency(context: str) -> str | None:
    lowered = context.casefold()
    for key, value in _PROFICIENCY_RULES:
        if key in lowered:
            return value
    return None


def _infer_essential(context: str) -> bool:
    lowered = context.casefold()
    if any(term in lowered for term in _OPTIONAL_HINTS):
        return False
    if any(term in lowered for term in _REQUIRED_HINTS):
        return True
    return True


def _infer_experience_years(context: str) -> float | None:
    match = _YEAR_PATTERN.search(context.casefold())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_relations(example: NerExample) -> list[RelationRecord]:
    spans = extract_entity_spans(example)
    occupations = [span for span in spans if span.entity_type == "OCC"]
    if not occupations:
        return []

    relations: list[RelationRecord] = []
    for span in spans:
        if span.entity_type == "OCC":
            continue
        relation = _RELATION_MAP.get(span.entity_type)
        if not relation:
            continue
        closest = min(occupations, key=lambda occ: abs(occ.start_token - span.start_token))
        context = _context_window(example.text, span.start_char, span.end_char)
        relations.append(
            RelationRecord(
                relation=relation,
                head=closest,
                tail=span,
                essential=_infer_essential(context),
                evidence_count=1,
                proficiency_level=_infer_proficiency(context),
                experience_years=_infer_experience_years(context),
                sentence_id=example.id,
            )
        )
    return relations


def extract_relations_from_dataset(examples: Iterable[NerExample]) -> list[RelationRecord]:
    records: list[RelationRecord] = []
    for example in examples:
        records.extend(extract_relations(example))
    return records
