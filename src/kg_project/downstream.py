from __future__ import annotations

import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import typer


def _normalize(text: str) -> str:
    return "".join(text.casefold().split())


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


def _find_occupation_id(nodes: dict[str, dict], name: str) -> str | None:
    target = _normalize(name)
    for node_id, node in nodes.items():
        if node.get("entity_type") != "OCC":
            continue
        if _normalize(node.get("name", "")) == target:
            return node_id
    return None


def _build_occ_skill_matrix(nodes: dict[str, dict], edges: list[dict]) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
    occ_ids = [nid for nid, n in nodes.items() if n.get("entity_type") == "OCC"]
    skill_ids = [nid for nid, n in nodes.items() if n.get("entity_type") == "SKL"]
    matrix: dict[str, dict[str, float]] = {occ: defaultdict(float) for occ in occ_ids}
    for edge in edges:
        if edge["relation"] != "REQUIRES_SKILL":
            continue
        if edge["source"] not in matrix:
            continue
        matrix[edge["source"]][edge["target"]] += math.log1p(float(edge.get("evidence_count") or 0))
    return occ_ids, skill_ids, matrix


def recommend_roles(nodes_path: Path, edges_path: Path, skills: list[str], top_k: int = 5) -> dict:
    nodes = _load_nodes(nodes_path)
    edges = _load_edges(edges_path)
    skill_set = {_normalize(skill) for skill in skills}

    occ_ids, _, matrix = _build_occ_skill_matrix(nodes, edges)

    scores: list[dict] = []
    for occ in occ_ids:
        numer = 0.0
        denom = 0.0
        hits = 0
        for sid, weight in matrix[occ].items():
            denom += weight
            sname = _normalize(nodes.get(sid, {}).get("name", ""))
            if sname in skill_set:
                numer += weight
                hits += 1
        if denom == 0:
            continue
        scores.append({"occupation": nodes[occ]["name"], "score": round(numer / denom, 6), "hit_count": hits})

    scores.sort(key=lambda x: (x["score"], x["hit_count"]), reverse=True)
    return {"recommendations": scores[:top_k], "method": "weighted_overlap_log_evidence"}


def skill_gap(nodes_path: Path, edges_path: Path, occupation: str, skills: list[str]) -> dict:
    nodes = _load_nodes(nodes_path)
    edges = _load_edges(edges_path)
    occ_id = _find_occupation_id(nodes, occupation)
    if not occ_id:
        return {"occupation": occupation, "missing_skills": [], "reason": "occupation_not_found"}

    skill_set = {_normalize(skill) for skill in skills}
    missing: list[dict] = []
    for edge in edges:
        if edge["relation"] != "REQUIRES_SKILL":
            continue
        if edge["source"] != occ_id:
            continue
        target = edge["target"]
        target_name = nodes.get(target, {}).get("name", "")
        if _normalize(target_name) in skill_set:
            continue
        missing.append(
            {
                "skill": target_name,
                "essential": str(edge.get("essential", "true")).lower() == "true",
                "evidence_count": int(edge.get("evidence_count") or 0),
                "proficiency_level": edge.get("proficiency_level"),
            }
        )

    missing.sort(key=lambda row: (row["essential"], row["evidence_count"]), reverse=True)
    return {"occupation": nodes[occ_id]["name"], "missing_skills": missing}


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    u = a | b
    if not u:
        return 1.0
    return 1.0 - len(a & b) / len(u)


def growth_path(nodes_path: Path, edges_path: Path, source: str, target: str, top_k: int = 3) -> dict:
    nodes = _load_nodes(nodes_path)
    edges = _load_edges(edges_path)
    source_id = _find_occupation_id(nodes, source)
    target_id = _find_occupation_id(nodes, target)
    if not source_id or not target_id:
        return {"path": [], "reason": "occupation_not_found"}
    if source_id == target_id:
        return {"path": [nodes[source_id]["name"]], "reason": "same_role"}

    occ_skills: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge["relation"] == "REQUIRES_SKILL":
            occ_skills[edge["source"]].add(edge["target"])

    occ_ids = [nid for nid, n in nodes.items() if n.get("entity_type") == "OCC"]
    graph: dict[str, dict[str, float]] = defaultdict(dict)

    # use explicit SIMILAR_TO if present
    for edge in edges:
        if edge["relation"] == "SIMILAR_TO":
            graph[edge["source"]][edge["target"]] = 1.0 / max(float(edge.get("evidence_count") or 1), 1.0)

    # fallback: dense graph via skill distance
    if not graph:
        for i in range(len(occ_ids)):
            for j in range(i + 1, len(occ_ids)):
                a, b = occ_ids[i], occ_ids[j]
                d = _jaccard_distance(occ_skills[a], occ_skills[b])
                if d <= 0.85:
                    graph[a][b] = d + 1e-4
                    graph[b][a] = d + 1e-4

    # Dijkstra
    dist = {source_id: 0.0}
    prev: dict[str, str] = {}
    visited: set[str] = set()
    while True:
        cur = None
        cur_dist = float("inf")
        for node, d in dist.items():
            if node in visited:
                continue
            if d < cur_dist:
                cur, cur_dist = node, d
        if cur is None:
            break
        if cur == target_id:
            break
        visited.add(cur)
        for nxt, w in graph.get(cur, {}).items():
            nd = cur_dist + w
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                prev[nxt] = cur

    if target_id not in dist:
        return {"path": [nodes[source_id]["name"]], "reason": "path_not_found"}

    path_ids = [target_id]
    while path_ids[-1] != source_id:
        path_ids.append(prev[path_ids[-1]])
    path_ids.reverse()

    prereq = defaultdict(set)
    for edge in edges:
        if edge["relation"] == "PREREQUISITE_OF":
            prereq[edge["target"]].add(edge["source"])

    steps = []
    for i in range(len(path_ids) - 1):
        cur, nxt = path_ids[i], path_ids[i + 1]
        missing = list(occ_skills[nxt] - occ_skills[cur])
        missing_names = [nodes[mid]["name"] for mid in missing if mid in nodes]
        ordered = sorted(missing_names, key=lambda n: len(prereq.get(next((sid for sid, sn in ((k, nodes[k]["name"]) for k in nodes if nodes[k].get("entity_type") == "SKL") if sn == n), ""), set())))
        steps.append({"from": nodes[cur]["name"], "to": nodes[nxt]["name"], "new_skills": ordered[:5]})

    return {"path": [nodes[n]["name"] for n in path_ids], "steps": steps, "reason": "weighted_shortest_path"}


def recommend_command(
    nodes: Path = typer.Option(..., help="Graph nodes CSV."),
    edges: Path = typer.Option(..., help="Graph edges CSV."),
    skills: list[str] = typer.Option(..., help="User skill list."),
    top_k: int = typer.Option(5, help="Top-K occupations."),
    output: Path = typer.Option(..., help="Output JSON path."),
) -> None:
    payload = recommend_roles(nodes, edges, skills, top_k=top_k)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def skill_gap_command(
    nodes: Path = typer.Option(..., help="Graph nodes CSV."),
    edges: Path = typer.Option(..., help="Graph edges CSV."),
    occupation: str = typer.Option(..., help="Occupation name."),
    skills: list[str] = typer.Option(..., help="User skill list."),
    output: Path = typer.Option(..., help="Output JSON path."),
) -> None:
    payload = skill_gap(nodes, edges, occupation, skills)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def growth_path_command(
    nodes: Path = typer.Option(..., help="Graph nodes CSV."),
    edges: Path = typer.Option(..., help="Graph edges CSV."),
    source: str = typer.Option(..., help="Source occupation."),
    target: str = typer.Option(..., help="Target occupation."),
    output: Path = typer.Option(..., help="Output JSON path."),
) -> None:
    payload = growth_path(nodes, edges, source, target)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
