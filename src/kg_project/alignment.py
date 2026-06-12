from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
from rapidfuzz import fuzz, process

from kg_project.data_types import GazetteerEntry
from kg_project.relation_extraction import RelationRecord

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    canonical_id: str
    source: str
    score: float


@dataclass(frozen=True, slots=True)
class SemanticIndex:
    model_name: str
    embeddings: dict[str, np.ndarray]
    entries: dict[str, list[GazetteerEntry]]
    texts: dict[str, list[str]]


def _normalize_key(text: str) -> str:
    lowered = text.casefold()
    cleaned = re.sub(r"[^a-z0-9\u4e00-\u9fff+.#]+", "", lowered)
    return cleaned


def build_gazetteer_index(entries: Iterable[GazetteerEntry]) -> dict[str, list[tuple[str, GazetteerEntry]]]:
    index: dict[str, list[tuple[str, GazetteerEntry]]] = {}
    for entry in entries:
        variants = [entry.term, *entry.aliases]
        for variant in variants:
            key = _normalize_key(variant)
            if not key:
                continue
            index.setdefault(entry.entity_type, []).append((key, entry))
    return index


def build_exact_index(
    index: dict[str, list[tuple[str, GazetteerEntry]]],
) -> dict[str, dict[str, GazetteerEntry]]:
    exact: dict[str, dict[str, GazetteerEntry]] = {}
    for entity_type, rows in index.items():
        bucket = exact.setdefault(entity_type, {})
        for key, entry in rows:
            bucket.setdefault(key, entry)
    return exact


def build_semantic_index(
    entries: Iterable[GazetteerEntry],
    model_name: str = "all-MiniLM-L6-v2",
) -> SemanticIndex:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")

    model = SentenceTransformer(model_name)
    texts_by_type: dict[str, list[str]] = {}
    entries_by_type: dict[str, list[GazetteerEntry]] = {}

    for entry in entries:
        for variant in entry.variants():
            text = f"{variant} {entry.description}".strip()
            texts_by_type.setdefault(entry.entity_type, []).append(text)
            entries_by_type.setdefault(entry.entity_type, []).append(entry)

    embeddings: dict[str, np.ndarray] = {}
    for entity_type, texts in texts_by_type.items():
        vectors = model.encode(texts, normalize_embeddings=True)
        embeddings[entity_type] = np.asarray(vectors)

    return SemanticIndex(
        model_name=model_name,
        embeddings=embeddings,
        entries=entries_by_type,
        texts=texts_by_type,
    )


def candidate_alignments(
    text: str,
    entity_type: str,
    index: dict[str, list[tuple[str, GazetteerEntry]]],
    top_k: int = 5,
) -> list[AlignmentResult]:
    if entity_type not in index:
        return []
    key = _normalize_key(text)
    if not key:
        return []

    choices = [choice for choice, _ in index[entity_type]]
    matches = process.extract(key, choices, scorer=fuzz.ratio, limit=top_k)
    results: list[AlignmentResult] = []
    for _, score, match_index in matches:
        entry = index[entity_type][match_index][1]
        results.append(AlignmentResult(canonical_id=entry.canonical_id, source=entry.source, score=float(score)))
    return results


def align_entity(
    text: str,
    entity_type: str,
    index: dict[str, list[tuple[str, GazetteerEntry]]],
    min_score: float = 85.0,
    exact_index: dict[str, dict[str, GazetteerEntry]] | None = None,
) -> AlignmentResult | None:
    key = _normalize_key(text)
    if exact_index and key in exact_index.get(entity_type, {}):
        entry = exact_index[entity_type][key]
        return AlignmentResult(canonical_id=entry.canonical_id, source=entry.source, score=100.0)
    cands = candidate_alignments(text, entity_type, index, top_k=1)
    if not cands:
        return None
    best = cands[0]
    if best.score < min_score:
        return None
    return best


def align_entity_semantic(
    text: str,
    entity_type: str,
    index: SemanticIndex,
    min_score: float = 80.0,
) -> AlignmentResult | None:
    if entity_type not in index.embeddings:
        return None
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")

    model = SentenceTransformer(index.model_name)
    query = model.encode([text], normalize_embeddings=True)
    scores = np.dot(index.embeddings[entity_type], query[0])
    best_idx = int(np.argmax(scores))
    score = float(scores[best_idx] * 100)
    if score < min_score:
        return None
    entry = index.entries[entity_type][best_idx]
    return AlignmentResult(canonical_id=entry.canonical_id, source=entry.source, score=score)


def _llm_choose_alignment(
    mention: str,
    entity_type: str,
    candidates: list[AlignmentResult],
    api_key: str,
    api_base: str,
    model: str,
) -> AlignmentResult | None:
    if not candidates:
        return None
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Select the best canonical concept ID for a mention. Return strict JSON: {\"best\":\"<id or null>\"}.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "mention": mention,
                        "entity_type": entity_type,
                        "candidates": [
                            {"canonical_id": c.canonical_id, "source": c.source, "score": c.score} for c in candidates
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": 0,
    }
    req = Request(
        f"{api_base.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urlopen(req, timeout=45) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    text = raw["choices"][0]["message"]["content"]
    parsed = json.loads(text)
    best = parsed.get("best")
    if not best:
        return None
    for cand in candidates:
        if cand.canonical_id == best:
            return cand
    return None


def align_relations(
    relations: Iterable[RelationRecord],
    index: dict[str, list[tuple[str, GazetteerEntry]]],
    min_score: float = 85.0,
    semantic_index: SemanticIndex | None = None,
    semantic_min_score: float = 80.0,
    llm_rerank: bool = False,
    llm_api_key: str | None = None,
    llm_api_base: str = "https://api.openai.com",
    llm_model: str = "gpt-4o-mini",
) -> list[dict]:
    aligned: list[dict] = []
    exact_index = build_exact_index(index)
    for record in relations:
        base = record.to_dict()

        if semantic_index:
            head_alignment = align_entity_semantic(
                record.head.text,
                record.head.entity_type,
                semantic_index,
                min_score=semantic_min_score,
            )
            tail_alignment = align_entity_semantic(
                record.tail.text,
                record.tail.entity_type,
                semantic_index,
                min_score=semantic_min_score,
            )
        else:
            head_alignment = align_entity(
                record.head.text,
                record.head.entity_type,
                index,
                min_score=min_score,
                exact_index=exact_index,
            )
            tail_alignment = align_entity(
                record.tail.text,
                record.tail.entity_type,
                index,
                min_score=min_score,
                exact_index=exact_index,
            )

        if llm_rerank and llm_api_key and not semantic_index:
            if head_alignment is None:
                cands = candidate_alignments(record.head.text, record.head.entity_type, index, top_k=5)
                chosen = _llm_choose_alignment(
                    mention=record.head.text,
                    entity_type=record.head.entity_type,
                    candidates=cands,
                    api_key=llm_api_key,
                    api_base=llm_api_base,
                    model=llm_model,
                )
                if chosen and chosen.score >= min_score - 10:
                    head_alignment = chosen
            if tail_alignment is None:
                cands = candidate_alignments(record.tail.text, record.tail.entity_type, index, top_k=5)
                chosen = _llm_choose_alignment(
                    mention=record.tail.text,
                    entity_type=record.tail.entity_type,
                    candidates=cands,
                    api_key=llm_api_key,
                    api_base=llm_api_base,
                    model=llm_model,
                )
                if chosen and chosen.score >= min_score - 10:
                    tail_alignment = chosen

        aligned.append(
            {
                **base,
                "head": {
                    **base["head"],
                    "canonical_id": head_alignment.canonical_id if head_alignment else None,
                    "source": head_alignment.source if head_alignment else None,
                    "score": head_alignment.score if head_alignment else None,
                },
                "tail": {
                    **base["tail"],
                    "canonical_id": tail_alignment.canonical_id if tail_alignment else None,
                    "source": tail_alignment.source if tail_alignment else None,
                    "score": tail_alignment.score if tail_alignment else None,
                },
            }
        )
    return aligned
