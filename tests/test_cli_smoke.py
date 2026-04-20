from pathlib import Path

from typer.testing import CliRunner

from kg_project.cli import app


runner = CliRunner()


def test_build_gazetteer_cli_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "gazetteer.jsonl"
    result = runner.invoke(
        app,
        [
            "build-gazetteer",
            "--esco",
            "data/fixtures/esco_skills.csv",
            "--onet",
            "data/fixtures/onet_reference.tsv",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
