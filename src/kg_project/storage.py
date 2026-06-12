from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class GraphStore(Protocol):
    def summary(self) -> dict: ...
    def graph_raw(self, limit: int = 1200, include_similarity: bool = False) -> dict: ...
    def occupation_profile(self, name: str) -> dict: ...
    def recommend(self, skills: list[str], top_k: int) -> dict: ...
    def skill_gap(self, occupation: str, skills: list[str]) -> dict: ...
    def growth_path(self, source: str, target: str) -> dict: ...


def _normalize(text: str) -> str:
    return "".join(str(text).casefold().split())


def _safe_int(value: str | int | float | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


ROLE_KEYWORDS = (
    "administrator",
    "analyst",
    "architect",
    "consultant",
    "developer",
    "designer",
    "devops",
    "engineer",
    "manager",
    "qa",
    "scientist",
    "specialist",
    "sre",
    "tester",
)


def _looks_like_occupation(name: str) -> bool:
    lowered = name.casefold()
    return any(keyword in lowered for keyword in ROLE_KEYWORDS)


@dataclass
class CsvGraphStore:
    nodes_path: Path
    edges_path: Path

    def __post_init__(self) -> None:
        self.nodes = self._load_nodes(self.nodes_path)
        self.edges = self._load_edges(self.edges_path)

    @staticmethod
    def _is_readable(text: str) -> bool:
        if not text:
            return False
        bad_chars = {"�", "鈥", "鏈", "宸", "鏁", "绉"}
        return not any(ch in text for ch in bad_chars)

    def _display_name(self, row: dict) -> str:
        primary = row.get("name", "")
        if self._is_readable(primary):
            return primary
        for alt in str(row.get("alt_names", "")).split("|"):
            if self._is_readable(alt):
                return alt
        return primary

    def _load_nodes(self, path: Path) -> dict[str, dict]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = {row["id"]: row for row in csv.DictReader(handle)}
        for row in rows.values():
            row["display_name"] = self._display_name(row)
        return rows

    @staticmethod
    def _load_edges(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def summary(self) -> dict:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "relations": sorted({e["relation"] for e in self.edges}),
            "backend": "csv",
        }

    def graph_raw(self, limit: int = 1200, include_similarity: bool = False) -> dict:
        candidate_edges = [
            edge for edge in self.edges if include_similarity or edge.get("relation") != "SIMILAR_TO"
        ]
        candidate_edges.sort(key=lambda edge: _safe_int(edge.get("evidence_count")), reverse=True)
        selected_edges = candidate_edges[: max(0, int(limit))]
        selected_node_ids = {edge["source"] for edge in selected_edges} | {edge["target"] for edge in selected_edges}
        return {
            "nodes": [
                {
                    "id": node_id,
                    "name": payload.get("display_name") or payload.get("name"),
                    "entity_type": payload.get("entity_type"),
                }
                for node_id, payload in self.nodes.items()
                if node_id in selected_node_ids
            ],
            "edges": selected_edges,
            "backend": "csv",
            "is_limited": len(selected_edges) < len(self.edges),
            "visualized_edges": len(selected_edges),
            "total_edges": len(self.edges),
        }

    def _find_occupation_id(self, name: str) -> str | None:
        needle = _normalize(name)
        for nid, n in self.nodes.items():
            if n.get("entity_type") != "OCC":
                continue
            if _normalize(n.get("name", "")) == needle:
                return nid
            if _normalize(n.get("display_name", "")) == needle:
                return nid
            if needle and needle in _normalize(n.get("alt_names", "")):
                return nid
        return None

    def occupation_profile(self, name: str) -> dict:
        oid = self._find_occupation_id(name)
        if not oid:
            return {"occupation": name, "relations": [], "backend": "csv"}
        rels = []
        for e in self.edges:
            if e["source"] != oid:
                continue
            rels.append(
                {
                    "relation": e["relation"],
                    "target": self.nodes.get(e["target"], {}).get("display_name")
                    or self.nodes.get(e["target"], {}).get("name"),
                    "target_type": self.nodes.get(e["target"], {}).get("entity_type"),
                    "evidence_count": _safe_int(e.get("evidence_count")),
                }
            )
        rels.sort(key=lambda x: (x["relation"], -x["evidence_count"]))
        return {
            "occupation": self.nodes[oid].get("display_name") or self.nodes[oid].get("name"),
            "relations": rels,
            "backend": "csv",
        }

    def recommend(self, skills: list[str], top_k: int) -> dict:
        skill_set = {_normalize(skill) for skill in skills}
        known_skill_names = {
            _normalize(node.get("display_name") or node.get("name", ""))
            for node in self.nodes.values()
            if node.get("entity_type") in {"SKL", "TOL", "KNW", "ABL", "TSK"}
        }
        matched_inputs = [skill for skill in skills if _normalize(skill) in known_skill_names]
        unmatched_inputs = [skill for skill in skills if _normalize(skill) not in known_skill_names]
        scores: dict[str, dict[str, float | int]] = {}
        for edge in self.edges:
            if edge["relation"] not in {"REQUIRES_SKILL", "USES_TOOL", "REQUIRES_KNOWLEDGE"}:
                continue
            source = edge["source"]
            target = edge["target"]
            if self.nodes.get(source, {}).get("entity_type") != "OCC":
                continue
            target_name = _normalize(self.nodes.get(target, {}).get("display_name") or self.nodes.get(target, {}).get("name", ""))
            payload = scores.setdefault(source, {"num": 0.0, "den": 0.0, "hits": 0})
            w = math.log1p(_safe_int(edge.get("evidence_count"), 1))
            payload["den"] += w
            if target_name in skill_set:
                payload["num"] += w
                payload["hits"] += 1

        ranked = []
        for occ_id, p in scores.items():
            den = float(p["den"])
            hits = int(p["hits"])
            if den <= 0 or hits <= 0:
                continue
            occupation_name = self.nodes[occ_id].get("display_name") or self.nodes[occ_id].get("name")
            if not _looks_like_occupation(occupation_name):
                continue
            title_hits = sum(1 for skill in skill_set if skill and skill in _normalize(occupation_name))
            ranked.append(
                {
                    "occupation": occupation_name,
                    "hit_count": hits,
                    "score": round(float(p["num"]) / den, 6),
                    "match_weight": round(float(p["num"]), 6),
                    "title_hit_count": title_hits,
                }
            )
        ranked.sort(key=lambda x: (x["hit_count"], x["title_hit_count"], x["match_weight"], x["score"]), reverse=True)
        return {
            "recommendations": ranked[:top_k],
            "matched_inputs": matched_inputs,
            "unmatched_inputs": unmatched_inputs,
            "backend": "csv",
        }

    def skill_gap(self, occupation: str, skills: list[str]) -> dict:
        occ_id = self._find_occupation_id(occupation)
        if not occ_id:
            return {"occupation": occupation, "missing_skills": [], "reason": "occupation_not_found", "backend": "csv"}
        skill_set = {_normalize(skill) for skill in skills}
        missing = []
        for edge in self.edges:
            if edge["relation"] != "REQUIRES_SKILL" or edge["source"] != occ_id:
                continue
            target_name = self.nodes.get(edge["target"], {}).get("display_name") or self.nodes.get(edge["target"], {}).get("name", "")
            if _normalize(target_name) in skill_set:
                continue
            missing.append(
                {
                    "skill": target_name,
                    "essential": str(edge.get("essential", "true")).lower() == "true",
                    "evidence_count": _safe_int(edge.get("evidence_count")),
                    "proficiency_level": edge.get("proficiency_level") or None,
                }
            )
        missing.sort(key=lambda x: (x["essential"], x["evidence_count"]), reverse=True)
        return {
            "occupation": self.nodes[occ_id].get("display_name") or self.nodes[occ_id].get("name"),
            "missing_skills": missing,
            "backend": "csv",
        }

    def growth_path(self, source: str, target: str) -> dict:
        src = self._find_occupation_id(source)
        dst = self._find_occupation_id(target)
        if not src or not dst:
            return {"path": [], "reason": "occupation_not_found", "backend": "csv"}

        adj: dict[str, set[str]] = {}
        for edge in self.edges:
            if edge["relation"] != "SIMILAR_TO":
                continue
            adj.setdefault(edge["source"], set()).add(edge["target"])
            adj.setdefault(edge["target"], set()).add(edge["source"])

        queue: list[tuple[str, list[str]]] = [(src, [src])]
        seen = {src}
        while queue:
            nid, path = queue.pop(0)
            if nid == dst:
                return {
                    "path": [self.nodes[p].get("display_name") or self.nodes[p].get("name") for p in path],
                    "reason": "similarity_graph",
                    "backend": "csv",
                }
            for nxt in adj.get(nid, set()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, [*path, nxt]))

        return {
            "path": [self.nodes[src].get("display_name") or self.nodes[src].get("name")],
            "reason": "path_not_found",
            "backend": "csv",
        }


@dataclass
class Neo4jGraphStore:
    uri: str
    username: str
    password: str
    database: str = "neo4j"

    def __post_init__(self) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("neo4j python driver is required. install with pip install neo4j") from exc
        self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self) -> None:
        self._driver.close()

    def _query(self, cypher: str, **params):
        with self._driver.session(database=self.database) as session:
            return [record.data() for record in session.run(cypher, **params)]

    def summary(self) -> dict:
        rows = self._query(
            """
            MATCH (n:Entity)
            OPTIONAL MATCH ()-[r:RELATION]->()
            RETURN count(DISTINCT n) AS num_nodes, count(DISTINCT r) AS num_edges
            """
        )
        rels = self._query("MATCH ()-[r:RELATION]->() RETURN DISTINCT r.relation AS relation ORDER BY relation")
        if not rows:
            return {"num_nodes": 0, "num_edges": 0, "relations": [], "backend": "neo4j"}
        row = rows[0]
        return {
            "num_nodes": int(row["num_nodes"]),
            "num_edges": int(row["num_edges"]),
            "relations": [r["relation"] for r in rels],
            "backend": "neo4j",
        }

    def graph_raw(self, limit: int = 1200, include_similarity: bool = False) -> dict:
        edges = self._query(
            """
            MATCH (s:Entity)-[r:RELATION]->(t:Entity)
            WHERE $include_similarity OR r.relation <> 'SIMILAR_TO'
            RETURN s.id AS source, t.id AS target, r.relation AS relation,
                   r.evidence_count AS evidence_count, r.essential AS essential,
                   r.proficiency_level AS proficiency_level, r.experience_years AS experience_years
            ORDER BY coalesce(r.evidence_count,0) DESC
            LIMIT $limit
            """,
            include_similarity=include_similarity,
            limit=max(0, int(limit)),
        )
        node_ids = sorted({edge["source"] for edge in edges} | {edge["target"] for edge in edges})
        nodes = self._query(
            """
            MATCH (n:Entity)
            WHERE n.id IN $node_ids
            RETURN n.id AS id, n.name AS name, n.entity_type AS entity_type
            """,
            node_ids=node_ids,
        )
        totals = self._query("MATCH ()-[r:RELATION]->() RETURN count(r) AS total_edges")
        total_edges = int(totals[0]["total_edges"]) if totals else len(edges)
        return {
            "nodes": nodes,
            "edges": edges,
            "backend": "neo4j",
            "is_limited": len(edges) < total_edges,
            "visualized_edges": len(edges),
            "total_edges": total_edges,
        }

    def occupation_profile(self, name: str) -> dict:
        rows = self._query(
            """
            MATCH (o:Entity {entity_type:'OCC'})
            WHERE toLower(replace(o.name,' ','')) = toLower(replace($name,' ',''))
            MATCH (o)-[r:RELATION]->(e:Entity)
            RETURN o.name AS occupation, r.relation AS relation, e.name AS target,
                   e.entity_type AS target_type, coalesce(r.evidence_count,0) AS evidence_count
            ORDER BY relation, evidence_count DESC
            """,
            name=name,
        )
        if not rows:
            return {"occupation": name, "relations": [], "backend": "neo4j"}
        return {
            "occupation": rows[0]["occupation"],
            "relations": [
                {
                    "relation": r["relation"],
                    "target": r["target"],
                    "target_type": r["target_type"],
                    "evidence_count": int(r["evidence_count"]),
                }
                for r in rows
            ],
            "backend": "neo4j",
        }

    def recommend(self, skills: list[str], top_k: int) -> dict:
        normalized_skills = [_normalize(s) for s in skills]
        known_rows = self._query(
            """
            MATCH (s:Entity)
            WHERE s.entity_type IN ['SKL', 'TOL', 'KNW', 'ABL', 'TSK']
            RETURN toLower(replace(s.name,' ','')) AS skill
            """
        )
        known_skill_names = {row["skill"] for row in known_rows}
        matched_inputs = [skill for skill in skills if _normalize(skill) in known_skill_names]
        unmatched_inputs = [skill for skill in skills if _normalize(skill) not in known_skill_names]
        rows = self._query(
            """
            MATCH (o:Entity {entity_type:'OCC'})-[r:RELATION]->(s:Entity)
            WHERE r.relation IN ['REQUIRES_SKILL', 'USES_TOOL', 'REQUIRES_KNOWLEDGE']
            WITH o, collect(toLower(replace(s.name,' ',''))) AS all_skills,
                 collect(coalesce(r.evidence_count,1)) AS weights,
                 $skills AS user_skills
            WITH o, all_skills, weights,
                 reduce(num=0.0, i IN range(0, size(all_skills)-1) |
                    num + CASE WHEN all_skills[i] IN user_skills THEN log(1.0 + toFloat(weights[i])) ELSE 0.0 END) AS numer,
                 reduce(den=0.0, w IN weights | den + log(1.0 + toFloat(w))) AS denom,
                 reduce(h=0, sk IN all_skills | h + CASE WHEN sk IN user_skills THEN 1 ELSE 0 END) AS hit_count
            WHERE denom > 0 AND hit_count > 0
            RETURN o.name AS occupation, hit_count, numer/denom AS score, numer AS match_weight
            ORDER BY hit_count DESC, score DESC, match_weight DESC
            LIMIT $candidate_limit
            """,
            skills=normalized_skills,
            candidate_limit=max(int(top_k) * 50, 200),
        )
        recommendations = [
            {
                "occupation": r["occupation"],
                "hit_count": int(r["hit_count"]),
                "score": round(float(r["score"]), 6),
                "match_weight": round(float(r["match_weight"]), 6),
                "title_hit_count": sum(
                    1 for skill in normalized_skills if skill and skill in _normalize(r["occupation"])
                ),
            }
            for r in rows
            if _looks_like_occupation(r["occupation"])
        ]
        recommendations.sort(
            key=lambda x: (x["hit_count"], x["title_hit_count"], x["match_weight"], x["score"]),
            reverse=True,
        )
        return {
            "recommendations": recommendations[:top_k],
            "matched_inputs": matched_inputs,
            "unmatched_inputs": unmatched_inputs,
            "backend": "neo4j",
        }

    def skill_gap(self, occupation: str, skills: list[str]) -> dict:
        rows = self._query(
            """
            MATCH (o:Entity {entity_type:'OCC'})-[r:RELATION {relation:'REQUIRES_SKILL'}]->(s:Entity)
            WHERE toLower(replace(o.name,' ','')) = toLower(replace($occupation,' ',''))
              AND NOT toLower(replace(s.name,' ','')) IN $skills
            RETURN o.name AS occupation, s.name AS skill,
                   coalesce(r.essential,true) AS essential,
                   coalesce(r.evidence_count,0) AS evidence_count,
                   r.proficiency_level AS proficiency_level
            ORDER BY essential DESC, evidence_count DESC
            """,
            occupation=occupation,
            skills=[_normalize(s) for s in skills],
        )
        if not rows:
            return {"occupation": occupation, "missing_skills": [], "backend": "neo4j"}
        return {
            "occupation": rows[0]["occupation"],
            "missing_skills": [
                {
                    "skill": r["skill"],
                    "essential": bool(r["essential"]),
                    "evidence_count": int(r["evidence_count"]),
                    "proficiency_level": r.get("proficiency_level"),
                }
                for r in rows
            ],
            "backend": "neo4j",
        }

    def growth_path(self, source: str, target: str) -> dict:
        rows = self._query(
            """
            MATCH p = allShortestPaths(
              (a:Entity {entity_type:'OCC', name:$source})-[:RELATION {relation:'SIMILAR_TO'}*..6]-(b:Entity {entity_type:'OCC', name:$target})
            )
            RETURN [n IN nodes(p) | n.name] AS path
            LIMIT 1
            """,
            source=source,
            target=target,
        )
        if not rows:
            return {"path": [source], "reason": "path_not_found", "backend": "neo4j"}
        return {"path": rows[0]["path"], "reason": "similarity_graph", "backend": "neo4j"}


def build_store(
    nodes_path: Path,
    edges_path: Path,
    use_neo4j: bool = False,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "neo4j",
    neo4j_database: str = "neo4j",
) -> GraphStore:
    if use_neo4j:
        return Neo4jGraphStore(
            uri=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            database=neo4j_database,
        )
    return CsvGraphStore(nodes_path=nodes_path, edges_path=edges_path)
