from __future__ import annotations

import json
from pathlib import Path

import typer

from kg_project.ner.crf_baseline import CRFTagger
from kg_project.ner.data import load_ner_examples
from kg_project.ner.neural import MissingTrainingDependencyError, ensure_training_dependencies


def run_crf_experiment(dataset_path: Path) -> dict[str, dict[str, float | int | str]]:
    examples = load_ner_examples(dataset_path)
    model = CRFTagger()
    model.fit(examples)
    report = model.evaluate(examples)
    return {
        "crf": {
            "precision": round(report.precision, 4),
            "recall": round(report.recall, 4),
            "f1": round(report.f1, 4),
            "num_examples": report.num_examples,
            "status": "ok",
        }
    }


def compare_models(dataset_path: Path) -> dict[str, dict[str, float | int | str]]:
    report = run_crf_experiment(dataset_path)
    try:
        ensure_training_dependencies()
    except MissingTrainingDependencyError as exc:
        message = str(exc)
        report["bilstm_crf"] = {"status": "skipped", "reason": message}
        report["bert_crf"] = {"status": "skipped", "reason": message}
        return report

    report["bilstm_crf"] = {"status": "implemented", "reason": "训练入口已提供，建议在真实数据上运行。"}
    report["bert_crf"] = {"status": "implemented", "reason": "训练入口已提供，首次运行需下载预训练权重。"}
    return report


def compare_ner_command(
    dataset: Path = typer.Option(..., help="NER dataset JSONL."),
    output: Path = typer.Option(..., help="Destination path for the JSON report."),
) -> None:
    report = compare_models(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
