from pathlib import Path

from kg_project.evaluation import evaluate_graph
from kg_project.gazetteer import build_gazetteer, write_gazetteer_jsonl
from kg_project.graph_pipeline import build_graph_from_ner


def test_evaluate_graph_returns_stats(tmp_path: Path) -> None:
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

    report = evaluate_graph(nodes_path, edges_path)
    assert report["num_nodes"] > 0
    assert report["num_edges"] > 0
