from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from seqeval.metrics import classification_report
import typer

from kg_project.data_sources import default_crf_config_path, load_crf_experiment_config, resolve_crf_paths
from kg_project.data_types import GazetteerEntry, read_jsonl
from kg_project.ner.crf_baseline import CRFTagger
from kg_project.ner.data import load_ner_examples
from kg_project.ner.features import compile_gazetteer_entries
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


def _load_gazetteer_entries(path: Path) -> list[GazetteerEntry]:
    if not path.exists():
        return []
    return [GazetteerEntry(**row) for row in read_jsonl(path)]


def _jd_coverage(jd_sample_path: Path, gazetteer_entries: list[GazetteerEntry]) -> dict[str, float | int]:
    rows = read_jsonl(jd_sample_path) if jd_sample_path.exists() else []
    if not rows:
        return {"num_documents": 0, "documents_with_hits": 0, "coverage_ratio": 0.0}

    from kg_project.ner.features import build_gazetteer_lookup

    matcher = compile_gazetteer_entries(gazetteer_entries)
    documents_with_hits = 0
    for row in rows:
        tokens = str(row["description"]).split()
        if build_gazetteer_lookup(tokens, matcher=matcher):
            documents_with_hits += 1
    return {
        "num_documents": len(rows),
        "documents_with_hits": documents_with_hits,
        "coverage_ratio": round(documents_with_hits / len(rows), 4),
    }


def run_formal_crf_experiment(config_path: Path | None = None) -> dict:
    config = load_crf_experiment_config(config_path)
    paths = resolve_crf_paths(config)
    train_examples = load_ner_examples(paths["train_path"])
    dev_examples = load_ner_examples(paths["dev_path"])
    test_examples = load_ner_examples(paths["test_path"])
    gazetteer_entries = _load_gazetteer_entries(paths["gazetteer_path"])
    gazetteer_matcher = compile_gazetteer_entries(gazetteer_entries)

    report: dict[str, object] = {
        "dataset": {
            "train_examples": len(train_examples),
            "dev_examples": len(dev_examples),
            "test_examples": len(test_examples),
        },
        "jd_coverage": _jd_coverage(paths["jd_sample_path"], gazetteer_entries),
        "experiments": {},
    }

    for name, experiment_config in config["experiments"].items():
        model = CRFTagger(
            c1=experiment_config["c1"],
            c2=experiment_config["c2"],
            max_iterations=experiment_config["max_iterations"],
            use_gazetteer=experiment_config["use_gazetteer"],
        )
        model.fit(
            train_examples,
            gazetteer_entries=gazetteer_entries,
            gazetteer_matcher=gazetteer_matcher,
        )
        dev_predictions = model.predict(
            dev_examples,
            gazetteer_entries=gazetteer_entries,
            gazetteer_matcher=gazetteer_matcher,
        )
        test_predictions = model.predict(
            test_examples,
            gazetteer_entries=gazetteer_entries,
            gazetteer_matcher=gazetteer_matcher,
        )
        dev_report = model.score_predictions(
            [example.labels for example in dev_examples],
            dev_predictions,
        )
        test_report = model.score_predictions(
            [example.labels for example in test_examples],
            test_predictions,
        )
        per_label = classification_report(
            [example.labels for example in test_examples],
            test_predictions,
            output_dict=True,
            zero_division=0,
        )
        report["experiments"][name] = {
            "config": experiment_config,
            "dev": {
                "precision": round(dev_report.precision, 4),
                "recall": round(dev_report.recall, 4),
                "f1": round(dev_report.f1, 4),
                "num_examples": dev_report.num_examples,
            },
            "test": {
                "precision": round(test_report.precision, 4),
                "recall": round(test_report.recall, 4),
                "f1": round(test_report.f1, 4),
                "num_examples": test_report.num_examples,
            },
            "per_label": per_label,
            "sample_errors": [
                {
                    "id": example.id,
                    "tokens": example.tokens,
                    "gold": example.labels,
                    "pred": pred,
                }
                for example, pred in zip(test_examples, test_predictions)
                if example.labels != pred
            ][:10],
        }

    base = report["experiments"]["crf_base"]["test"]["f1"]
    enhanced = report["experiments"]["crf_gazetteer"]["test"]["f1"]
    report["comparison"] = {
        "base_test_f1": base,
        "gazetteer_test_f1": enhanced,
        "delta_f1": round(enhanced - base, 4),
    }
    return report


def _render_formal_crf_markdown(report: dict) -> str:
    lines = [
        "# CRF Formal Experiment Report",
        "",
        "## Dataset",
        "",
        f"- Train examples: {report['dataset']['train_examples']}",
        f"- Dev examples: {report['dataset']['dev_examples']}",
        f"- Test examples: {report['dataset']['test_examples']}",
        "",
        "## JD Coverage",
        "",
        f"- Documents: {report['jd_coverage']['num_documents']}",
        f"- Documents with gazetteer hits: {report['jd_coverage']['documents_with_hits']}",
        f"- Coverage ratio: {report['jd_coverage']['coverage_ratio']}",
        "",
        "## Results",
        "",
    ]
    for name, payload in report["experiments"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Dev F1: {payload['dev']['f1']}",
                f"- Test F1: {payload['test']['f1']}",
                f"- Test Precision: {payload['test']['precision']}",
                f"- Test Recall: {payload['test']['recall']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Comparison",
            "",
            f"- CRF-base test F1: {report['comparison']['base_test_f1']}",
            f"- CRF+gazetteer test F1: {report['comparison']['gazetteer_test_f1']}",
            f"- Delta F1: {report['comparison']['delta_f1']}",
            "",
        ]
    )
    return "\n".join(lines)


def run_formal_crf_experiment_command(
    config: Path = typer.Option(default_crf_config_path(), help="Path to the CRF experiment config."),
) -> None:
    report = run_formal_crf_experiment(config_path=config)
    config_payload = load_crf_experiment_config(config)
    paths = resolve_crf_paths(config_payload)
    def _json_default(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    paths["report_json_path"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    paths["report_md_path"].write_text(_render_formal_crf_markdown(report), encoding="utf-8")


def compare_ner_command(
    dataset: Path = typer.Option(..., help="NER dataset JSONL."),
    output: Path = typer.Option(..., help="Destination path for the JSON report."),
) -> None:
    report = compare_models(dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
