from typer.testing import CliRunner

from kg_project.cli import app


runner = CliRunner()


def test_formal_crf_commands_are_registered() -> None:
    for command in ["fetch-crf-data", "prepare-crf-data", "run-crf-experiment"]:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
