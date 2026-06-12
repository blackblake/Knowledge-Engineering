from pathlib import Path

from kg_project.downstream import growth_path, recommend_roles, skill_gap
from kg_project.gazetteer import build_gazetteer, write_gazetteer_jsonl
from kg_project.graph_pipeline import build_graph_from_ner


def _build_graph(tmp_path: Path) -> tuple[Path, Path]:
    gazetteer = build_gazetteer(
        esco_path=Path("data/fixtures/esco_skills.csv"),
        onet_path=Path("data/fixtures/onet_reference.tsv"),
    )
    gazetteer_path = tmp_path / "gazetteer.jsonl"
    write_gazetteer_jsonl(gazetteer, gazetteer_path)

    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"

    build_graph_from_ner(
        ner_path=Path("data/fixtures/gold_ner.jsonl"),
        gazetteer_path=gazetteer_path,
        nodes_path=nodes_path,
        edges_path=edges_path,
    )
    return nodes_path, edges_path


def test_recommend_roles(tmp_path: Path) -> None:
    nodes_path, edges_path = _build_graph(tmp_path)
    payload = recommend_roles(nodes_path, edges_path, ["Python", "PyTorch"], top_k=3)
    assert payload["recommendations"]


def test_skill_gap(tmp_path: Path) -> None:
    nodes_path, edges_path = _build_graph(tmp_path)
    payload = skill_gap(nodes_path, edges_path, "Data Scientist", ["Python"])
    assert "missing_skills" in payload


def test_growth_path_fallback(tmp_path: Path) -> None:
    nodes_path, edges_path = _build_graph(tmp_path)
    payload = growth_path(nodes_path, edges_path, "Data Scientist", "Machine Learning Engineer")
    assert "path" in payload
