from __future__ import annotations

import csv
import json
from pathlib import Path

import typer


def _load_nodes(path: Path) -> dict[str, dict]:
    nodes: dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            nodes[row["id"]] = row
    return nodes


def _load_edges(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_graph(nodes_path: Path, edges_path: Path) -> dict:
    nodes = _load_nodes(nodes_path)
    edges = _load_edges(edges_path)

    degree: dict[str, int] = {node_id: 0 for node_id in nodes}
    relation_counts: dict[str, int] = {}
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        relation = edge["relation"]
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
        if source in degree:
            degree[source] += 1
        if target in degree:
            degree[target] += 1
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)

    if degree:
        degrees = list(degree.values())
        avg_degree = sum(degrees) / len(degrees)
        max_degree = max(degrees)
        min_degree = min(degrees)
    else:
        avg_degree = 0.0
        max_degree = 0
        min_degree = 0

    visited: set[str] = set()
    components = 0
    for node_id in nodes:
        if node_id in visited:
            continue
        components += 1
        queue = [node_id]
        visited.add(node_id)
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "relation_counts": relation_counts,
        "avg_degree": round(avg_degree, 4),
        "max_degree": max_degree,
        "min_degree": min_degree,
        "components": components,
    }


def evaluate_recommendations(
    nodes_path: Path,
    edges_path: Path,
    k: int = 5,
    holdout: int = 1,
) -> dict:
    nodes = _load_nodes(nodes_path)
    edges = _load_edges(edges_path)

    occ_skills: dict[str, list[str]] = {}
    for edge in edges:
        if edge["relation"] != "REQUIRES_SKILL":
            continue
        occ_skills.setdefault(edge["source"], []).append(edge["target"])

    hits = 0
    mrr_total = 0.0
    total = 0

    for occ_id, skills in occ_skills.items():
        if len(skills) <= holdout:
            continue
        total += 1
        held_out = set(skills[:holdout])
        user_skills = set(skills[holdout:])

        scores: dict[str, int] = {}
        for edge in edges:
            if edge["relation"] != "REQUIRES_SKILL":
                continue
            if edge["target"] not in user_skills:
                continue
            scores[edge["source"]] = scores.get(edge["source"], 0) + int(edge.get("evidence_count") or 0)

        ranked = sorted(scores.items(), key=lambda row: row[1], reverse=True)
        ranked_ids = [row[0] for row in ranked[:k]]
        if occ_id in ranked_ids:
            hits += 1
            rank = ranked_ids.index(occ_id) + 1
            mrr_total += 1.0 / rank

    if total == 0:
        return {"hit_at_k": 0.0, "mrr": 0.0, "num_cases": 0}

    return {
        "hit_at_k": round(hits / total, 4),
        "mrr": round(mrr_total / total, 4),
        "num_cases": total,
        "k": k,
    }


def evaluate_graph_command(
    nodes: Path = typer.Option(..., help="Graph nodes CSV."),
    edges: Path = typer.Option(..., help="Graph edges CSV."),
    output: Path = typer.Option(..., help="Output JSON path."),
) -> None:
    report = evaluate_graph(nodes, edges)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_recommendations_command(
    nodes: Path = typer.Option(..., help="Graph nodes CSV."),
    edges: Path = typer.Option(..., help="Graph edges CSV."),
    output: Path = typer.Option(..., help="Output JSON path."),
    k: int = typer.Option(5, help="Top-K value for Hit@K."),
    holdout: int = typer.Option(1, help="Number of skills held out for evaluation."),
) -> None:
    report = evaluate_recommendations(nodes, edges, k=k, holdout=holdout)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
