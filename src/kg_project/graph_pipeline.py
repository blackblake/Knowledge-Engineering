from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import typer

from kg_project.alignment import build_gazetteer_index, build_semantic_index, align_relations
from kg_project.data_types import GazetteerEntry, NerExample, read_jsonl
from kg_project.knowledge_fusion import fuse_relation_payload
from kg_project.relation_extraction import extract_relations_from_dataset


def _load_ner_examples(path: Path) -> list[NerExample]:
    return [NerExample(**row) for row in read_jsonl(path)]


def _load_gazetteer_entries(path: Path | None) -> list[GazetteerEntry]:
    if not path:
        return []
    if not path.exists():
        return []
    return [GazetteerEntry(**row) for row in read_jsonl(path)]


def _gazetteer_map(entries: list[GazetteerEntry]) -> dict[str, GazetteerEntry]:
    return {entry.canonical_id: entry for entry in entries}


def _prefer_english_name(primary: str, candidates: set[str]) -> str:
    for value in [primary, *sorted(candidates)]:
        if any("a" <= ch.lower() <= "z" for ch in value):
            return value
    return primary


def _node_id(entity: dict) -> str:
    if entity.get("canonical_id"):
        return str(entity["canonical_id"])
    return f"local:{entity['type']}:{entity['text']}"


def _aggregate_relations(relations: Iterable[dict]) -> list[dict]:
    aggregated: dict[tuple[str, str, str], dict] = {}
    proficiency_rank = {"expert": 3, "proficient": 2, "familiar": 1, "basic": 0}

    for relation in relations:
        head = relation["head"]
        tail = relation["tail"]
        key = (_node_id(head), relation["relation"], _node_id(tail))
        payload = aggregated.get(key)
        if not payload:
            aggregated[key] = {
                "source": _node_id(head),
                "target": _node_id(tail),
                "relation": relation["relation"],
                "essential": relation["attributes"]["essential"],
                "evidence_count": relation["attributes"]["evidence_count"],
                "proficiency_level": relation["attributes"].get("proficiency_level"),
                "experience_years": relation["attributes"].get("experience_years"),
            }
            continue
        payload["evidence_count"] += relation["attributes"]["evidence_count"]
        if payload["essential"] is False:
            payload["essential"] = relation["attributes"]["essential"]
        candidate = relation["attributes"].get("proficiency_level")
        if candidate:
            current = payload.get("proficiency_level")
            if not current or proficiency_rank.get(candidate, -1) > proficiency_rank.get(current, -1):
                payload["proficiency_level"] = candidate
        if payload.get("experience_years") is None and relation["attributes"].get("experience_years") is not None:
            payload["experience_years"] = relation["attributes"].get("experience_years")

    return list(aggregated.values())


def _build_nodes(relations: Iterable[dict], gazetteer_by_id: dict[str, GazetteerEntry] | None = None) -> list[dict]:
    nodes: dict[str, dict] = {}
    for relation in relations:
        for entity in (relation["head"], relation["tail"]):
            node_id = _node_id(entity)
            if node_id in nodes:
                continue
            canonical_id = entity.get("canonical_id")
            g_entry = gazetteer_by_id.get(canonical_id) if (gazetteer_by_id and canonical_id) else None
            base_name = g_entry.term if g_entry else entity["text"]
            alt_name_set: set[str] = set(entity.get("alt_names", []))
            alt_name_set.add(entity.get("text", ""))
            if g_entry:
                alt_name_set.add(g_entry.term)
                alt_name_set.update(g_entry.aliases)
            if entity["type"] == "OCC":
                base_name = _prefer_english_name(base_name, alt_name_set)
            nodes[node_id] = {
                "id": node_id,
                "label": entity["type"],
                "name": base_name,
                "entity_type": entity["type"],
                "source": entity.get("source"),
                "alt_names": "|".join(sorted(name for name in alt_name_set if name)),
            }
    return list(nodes.values())


def _build_similarity_and_prerequisite_edges(nodes: list[dict], edges: list[dict]) -> list[dict]:
    occ_to_skills: dict[str, set[str]] = defaultdict(set)
    skill_to_degree: dict[str, int] = defaultdict(int)
    for edge in edges:
        if edge["relation"] == "REQUIRES_SKILL":
            occ_to_skills[edge["source"]].add(edge["target"])
            skill_to_degree[edge["target"]] += int(edge.get("evidence_count") or 1)

    extra: list[dict] = []
    occ_ids = list(occ_to_skills.keys())
    for i in range(len(occ_ids)):
        for j in range(i + 1, len(occ_ids)):
            a, b = occ_ids[i], occ_ids[j]
            sa, sb = occ_to_skills[a], occ_to_skills[b]
            union = sa | sb
            if not union:
                continue
            jac = len(sa & sb) / len(union)
            if jac >= 0.3:
                score = max(1, int(jac * 100))
                extra.append({"source": a, "target": b, "relation": "SIMILAR_TO", "essential": True, "evidence_count": score, "proficiency_level": None, "experience_years": None})
                extra.append({"source": b, "target": a, "relation": "SIMILAR_TO", "essential": True, "evidence_count": score, "proficiency_level": None, "experience_years": None})

    # heuristic prerequisite chain by global rarity -> commonness
    skill_nodes = [n for n in nodes if n.get("entity_type") == "SKL"]
    skill_nodes = sorted(skill_nodes, key=lambda n: skill_to_degree.get(n["id"], 0))
    for idx in range(len(skill_nodes) - 1):
        s1 = skill_nodes[idx]["id"]
        s2 = skill_nodes[idx + 1]["id"]
        extra.append({"source": s1, "target": s2, "relation": "PREREQUISITE_OF", "essential": True, "evidence_count": 1, "proficiency_level": None, "experience_years": None})

    return extra


def _write_nodes_csv(nodes: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "name", "entity_type", "source", "alt_names"])
        writer.writeheader()
        writer.writerows(nodes)


def _write_edges_csv(edges: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "target",
                "relation",
                "essential",
                "evidence_count",
                "proficiency_level",
                "experience_years",
            ],
        )
        writer.writeheader()
        writer.writerows(edges)


def build_graph_from_ner(
    ner_path: Path,
    gazetteer_path: Path | None,
    nodes_path: Path,
    edges_path: Path,
    relations_path: Path | None = None,
    report_path: Path | None = None,
    min_score: float = 85.0,
    alignment_method: str = "fuzzy",
    semantic_model: str = "all-MiniLM-L6-v2",
    semantic_min_score: float = 80.0,
    llm_rerank: bool = False,
    llm_api_key: str | None = None,
    llm_api_base: str = "https://api.openai.com",
    llm_model: str = "gpt-4o-mini",
) -> dict:
    examples = _load_ner_examples(ner_path)
    relations = extract_relations_from_dataset(examples)

    gazetteer_entries = _load_gazetteer_entries(gazetteer_path)
    gazetteer_by_id: dict[str, GazetteerEntry] = {}
    if gazetteer_entries:
        gazetteer_by_id = _gazetteer_map(gazetteer_entries)
        index = build_gazetteer_index(gazetteer_entries)
        semantic_index = None
        if alignment_method == "sbert":
            semantic_index = build_semantic_index(gazetteer_entries, model_name=semantic_model)
        relation_payload = align_relations(
            relations,
            index,
            min_score=min_score,
            semantic_index=semantic_index,
            semantic_min_score=semantic_min_score,
            llm_rerank=llm_rerank,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            llm_model=llm_model,
        )
    else:
        relation_payload = [record.to_dict() for record in relations]

    fused_relations, fusion_stats = fuse_relation_payload(relation_payload)

    nodes = _build_nodes(fused_relations, gazetteer_by_id=gazetteer_by_id)
    edges = _aggregate_relations(fused_relations)
    edges.extend(_build_similarity_and_prerequisite_edges(nodes, edges))

    _write_nodes_csv(nodes, nodes_path)
    _write_edges_csv(edges, edges_path)

    if relations_path:
        relations_path.parent.mkdir(parents=True, exist_ok=True)
        with relations_path.open("w", encoding="utf-8") as handle:
            for relation in fused_relations:
                handle.write(json.dumps(relation, ensure_ascii=False) + "\n")

    report = {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "num_relations": len(fused_relations),
        "relation_counts": {},
        "alignment": {
            "aligned_heads": 0,
            "aligned_tails": 0,
            "total_relations": len(fused_relations),
        },
        "fusion": {
            "merged_entities": fusion_stats.merged_entities,
            "merged_relations": fusion_stats.merged_relations,
            "merged_alt_names": fusion_stats.merged_alt_names,
        },
    }
    for edge in edges:
        report["relation_counts"].setdefault(edge["relation"], 0)
        report["relation_counts"][edge["relation"]] += 1

    if fused_relations:
        aligned_heads = sum(1 for row in fused_relations if row["head"].get("canonical_id"))
        aligned_tails = sum(1 for row in fused_relations if row["tail"].get("canonical_id"))
        report["alignment"]["aligned_heads"] = aligned_heads
        report["alignment"]["aligned_tails"] = aligned_tails
        report["alignment"]["head_coverage"] = round(aligned_heads / len(fused_relations), 4)
        report["alignment"]["tail_coverage"] = round(aligned_tails / len(fused_relations), 4)

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_graph_command(
    ner: Path = typer.Option(..., help="NER dataset JSONL with tokens and BIO labels."),
    gazetteer: Path | None = typer.Option(None, help="Gazetteer JSONL for entity alignment."),
    nodes: Path = typer.Option(..., help="Output CSV path for graph nodes."),
    edges: Path = typer.Option(..., help="Output CSV path for graph edges."),
    relations: Path | None = typer.Option(None, help="Optional output JSONL path for relation records."),
    report: Path | None = typer.Option(None, help="Optional output report JSON path."),
    min_score: float = typer.Option(85.0, help="Minimum fuzzy match score for alignment."),
    alignment_method: str = typer.Option(
        "fuzzy",
        help="Alignment method: fuzzy (rapidfuzz) or sbert (sentence-transformers).",
    ),
    semantic_model: str = typer.Option(
        "all-MiniLM-L6-v2",
        help="SentenceTransformer model name for sbert alignment.",
    ),
    semantic_min_score: float = typer.Option(
        80.0,
        help="Minimum semantic similarity score (0-100) for sbert alignment.",
    ),
    llm_rerank: bool = typer.Option(False, help="Enable LLM reranking for unresolved alignment mentions."),
    llm_api_key: str | None = typer.Option(None, help="OpenAI-compatible API key for LLM alignment rerank."),
    llm_api_base: str = typer.Option("https://api.openai.com", help="OpenAI-compatible API base URL."),
    llm_model: str = typer.Option("gpt-4o-mini", help="LLM model for alignment rerank."),
) -> None:
    build_graph_from_ner(
        ner_path=ner,
        gazetteer_path=gazetteer,
        nodes_path=nodes,
        edges_path=edges,
        relations_path=relations,
        report_path=report,
        min_score=min_score,
        alignment_method=alignment_method,
        semantic_model=semantic_model,
        semantic_min_score=semantic_min_score,
        llm_rerank=llm_rerank,
        llm_api_key=llm_api_key,
        llm_api_base=llm_api_base,
        llm_model=llm_model,
    )
