from __future__ import annotations

import re
from typing import Iterable

from kg_project.data_types import EntityMention


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+.#/\-]*|[\u4e00-\u9fff]|[^\w\s]", re.UNICODE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.?!])\s+")


def split_sentences(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in SENTENCE_BOUNDARY.split(text) if chunk.strip()]
    return chunks or [text.strip()]


def normalize_lookup(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_with_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in TOKEN_PATTERN.finditer(text):
        tokens.append(match.group(0))
        spans.append(match.span())
    return tokens, spans


def spans_to_bio_labels(
    token_spans: list[tuple[int, int]],
    mentions: Iterable[EntityMention],
) -> list[str]:
    labels = ["O"] * len(token_spans)
    for mention in sorted(mentions, key=lambda item: (item.start, item.end)):
        covered_indices = [
            idx
            for idx, (start, end) in enumerate(token_spans)
            if not (end <= mention.start or start >= mention.end)
        ]
        if not covered_indices:
            continue
        labels[covered_indices[0]] = f"B-{mention.entity_type}"
        for idx in covered_indices[1:]:
            labels[idx] = f"I-{mention.entity_type}"
    return labels
