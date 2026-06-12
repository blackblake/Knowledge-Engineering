from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Iterable


SOURCE_PRIORITY = {
    "human": 4,
    "ESCO": 3,
    "O*NET": 2,
    "distant": 1,
    None: 0,
}


@dataclass(frozen=True, slots=True)
class FusionStats:
    merged_entities: int
    merged_relations: int
    merged_alt_names: int


def _priority(source: str | None) -> int:
    return SOURCE_PRIORITY.get(source, 1)


def _node_id(entity: dict) -> str:
    cid = entity.get("canonical_id")
    if cid:
        return str(cid)
    return f"local:{entity['type']}:{entity['text']}"


def fuse_relation_payload(relations: Iterable[dict]) -> tuple[list[dict], FusionStats]:
    grouped_nodes: dict[str, dict] = {}
    alt_names: dict[str, set[str]] = defaultdict(set)

    for rel in relations:
        for side in ("head", "tail"):
            ent = rel[side]
            nid = _node_id(ent)
            alt_names[nid].add(ent.get("text", ""))
            prev = grouped_nodes.get(nid)
            cand_source = ent.get("source")
            if prev is None:
                grouped_nodes[nid] = {
                    "canonical_id": ent.get("canonical_id"),
                    "source": cand_source,
                    "name": ent.get("text", ""),
                    "type": ent.get("type"),
                    "score": ent.get("score") or 0,
                }
            else:
                # conflict resolution: human > ESCO > O*NET > distant
                if _priority(cand_source) > _priority(prev.get("source")):
                    prev["source"] = cand_source
                    prev["name"] = ent.get("text", prev["name"])
                if (ent.get("score") or 0) > (prev.get("score") or 0):
                    prev["score"] = ent.get("score") or prev.get("score")

    merged: dict[tuple[str, str, str], dict] = {}
    proficiency_rank = {"expert": 3, "proficient": 2, "familiar": 1, "basic": 0}

    for rel in relations:
        head_id = _node_id(rel["head"])
        tail_id = _node_id(rel["tail"])
        key = (head_id, rel["relation"], tail_id)

        payload = merged.get(key)
        if not payload:
            payload = {
                "relation": rel["relation"],
                "head": {
                    **rel["head"],
                    "canonical_id": grouped_nodes[head_id]["canonical_id"],
                    "source": grouped_nodes[head_id]["source"],
                    "alt_names": sorted(name for name in alt_names[head_id] if name),
                },
                "tail": {
                    **rel["tail"],
                    "canonical_id": grouped_nodes[tail_id]["canonical_id"],
                    "source": grouped_nodes[tail_id]["source"],
                    "alt_names": sorted(name for name in alt_names[tail_id] if name),
                },
                "attributes": {
                    "essential": rel["attributes"].get("essential", True),
                    "evidence_count": int(rel["attributes"].get("evidence_count") or 1),
                    "proficiency_level": rel["attributes"].get("proficiency_level"),
                    "experience_years": rel["attributes"].get("experience_years"),
                },
                "sentence_ids": [rel.get("sentence_id")],
            }
            merged[key] = payload
            continue

        attrs = payload["attributes"]
        attrs["evidence_count"] += int(rel["attributes"].get("evidence_count") or 1)
        attrs["essential"] = attrs["essential"] or bool(rel["attributes"].get("essential", True))

        cand_prof = rel["attributes"].get("proficiency_level")
        cur_prof = attrs.get("proficiency_level")
        if cand_prof and (not cur_prof or proficiency_rank.get(cand_prof, -1) > proficiency_rank.get(cur_prof, -1)):
            attrs["proficiency_level"] = cand_prof

        cand_years = rel["attributes"].get("experience_years")
        cur_years = attrs.get("experience_years")
        if cand_years is not None and (cur_years is None or float(cand_years) > float(cur_years)):
            attrs["experience_years"] = cand_years

        payload["sentence_ids"].append(rel.get("sentence_id"))

    stats = FusionStats(
        merged_entities=len(grouped_nodes),
        merged_relations=len(merged),
        merged_alt_names=sum(len(v) for v in alt_names.values()),
    )
    return list(merged.values()), stats
