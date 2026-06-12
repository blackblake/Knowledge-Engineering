from pathlib import Path

from typer.testing import CliRunner

from kg_project.cli import app
from kg_project.gazetteer import build_gazetteer, write_gazetteer_jsonl


runner = CliRunner()


def test_build_graph_command_creates_outputs(tmp_path: Path) -> None:
    gazetteer = build_gazetteer(
        esco_path=Path("data/fixtures/esco_skills.csv"),
        onet_path=Path("data/fixtures/onet_reference.tsv"),
    )
    gazetteer_path = tmp_path / "gazetteer.jsonl"
    write_gazetteer_jsonl(gazetteer, gazetteer_path)

    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"
    relations_path = tmp_path / "relations.jsonl"
    report_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "build-graph",
            "--ner",
            "data/fixtures/gold_ner.jsonl",
            "--gazetteer",
            str(gazetteer_path),
            "--nodes",
            str(nodes_path),
            "--edges",
            str(edges_path),
            "--relations",
            str(relations_path),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert nodes_path.exists()
    assert edges_path.exists()
    assert relations_path.exists()
    assert report_path.exists()
