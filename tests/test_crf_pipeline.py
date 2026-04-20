from pathlib import Path

from kg_project.ner.experiments import run_crf_experiment


def test_crf_experiment_returns_metrics() -> None:
    report = run_crf_experiment(Path("data/fixtures/gold_ner.jsonl"))
    assert "crf" in report
    assert report["crf"]["num_examples"] == 6
    assert 0.0 <= report["crf"]["f1"] <= 1.0
