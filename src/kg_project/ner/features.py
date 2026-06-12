from __future__ import annotations

import re
from typing import TypeAlias

from kg_project.data_types import GazetteerEntry


WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+.#/\-]*|[^\w\s]")
GazetteerMatcher: TypeAlias = dict[str, list[tuple[tuple[str, ...], str]]]


def _term_to_tokens(term: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in WORD_PATTERN.findall(term))


def compile_gazetteer_entries(
    gazetteer_entries: list[GazetteerEntry] | None,
) -> GazetteerMatcher:
    matcher: GazetteerMatcher = {}
    if not gazetteer_entries:
        return matcher

    for entry in gazetteer_entries:
        for variant in entry.variants():
            term_tokens = _term_to_tokens(variant)
            if not term_tokens:
                continue
            bucket = matcher.setdefault(term_tokens[0], [])
            bucket.append((term_tokens, entry.entity_type))

    for bucket in matcher.values():
        bucket.sort(key=lambda item: len(item[0]), reverse=True)
    return matcher


def build_gazetteer_lookup(
    tokens: list[str],
    gazetteer_entries: list[GazetteerEntry] | None = None,
    matcher: GazetteerMatcher | None = None,
) -> dict[int, dict[str, object]]:
    if matcher is None:
        matcher = compile_gazetteer_entries(gazetteer_entries)
    if not matcher:
        return {}

    lowered_tokens = [token.lower() for token in tokens]
    lookup: dict[int, dict[str, object]] = {}
    for start_idx in range(len(tokens)):
        for term_tokens, entity_type in matcher.get(lowered_tokens[start_idx], []):
            end_idx = start_idx + len(term_tokens)
            if end_idx > len(tokens):
                continue
            if tuple(lowered_tokens[start_idx:end_idx]) != term_tokens:
                continue
            for idx in range(start_idx, end_idx):
                slot = lookup.setdefault(idx, {"hit": True, "types": set(), "span_len": 0, "begin": False})
                slot["types"].add(entity_type)
                slot["span_len"] = max(slot["span_len"], len(term_tokens))
                slot["begin"] = slot["begin"] or idx == start_idx

    for slot in lookup.values():
        slot["types"] = sorted(slot["types"])
        slot["type"] = slot["types"][0]
    return lookup


def token2features(
    tokens: list[str],
    index: int,
    gazetteer_lookup: dict[int, dict[str, object]] | None = None,
    use_gazetteer: bool = False,
) -> dict[str, object]:
    token = tokens[index]
    features: dict[str, object] = {
        "bias": 1.0,
        "token.lower": token.lower(),
        "token.isupper": token.isupper(),
        "token.istitle": token.istitle(),
        "token.isdigit": token.isdigit(),
        "token.prefix1": token[:1],
        "token.prefix2": token[:2],
        "token.suffix1": token[-1:],
        "token.suffix2": token[-2:],
        "token.length": len(token),
    }
    if index == 0:
        features["BOS"] = True
    else:
        prev = tokens[index - 1]
        features["-1:token.lower"] = prev.lower()
        features["-1:token.istitle"] = prev.istitle()
    if index == len(tokens) - 1:
        features["EOS"] = True
    else:
        nxt = tokens[index + 1]
        features["+1:token.lower"] = nxt.lower()
        features["+1:token.istitle"] = nxt.istitle()
    if use_gazetteer and gazetteer_lookup:
        match = gazetteer_lookup.get(index)
        features["gazetteer.hit"] = bool(match)
        if match:
            features["gazetteer.type"] = match["type"]
            features["gazetteer.span_len"] = match["span_len"]
            features["gazetteer.begin"] = match["begin"]
    return features


def sent2features(
    tokens: list[str],
    gazetteer_entries: list[GazetteerEntry] | None = None,
    matcher: GazetteerMatcher | None = None,
    use_gazetteer: bool = False,
) -> list[dict[str, object]]:
    lookup = (
        build_gazetteer_lookup(tokens, gazetteer_entries=gazetteer_entries, matcher=matcher)
        if use_gazetteer
        else {}
    )
    return [token2features(tokens, idx, lookup, use_gazetteer=use_gazetteer) for idx in range(len(tokens))]
