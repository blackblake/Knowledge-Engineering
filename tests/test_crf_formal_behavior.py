from pathlib import Path

import yaml

from kg_project.data_types import GazetteerEntry, JobPosting, write_jsonl
from kg_project.ner.experiments import run_formal_crf_experiment
from kg_project.ner.features import build_gazetteer_lookup, compile_gazetteer_entries


def test_build_gazetteer_lookup_marks_multi_token_hits() -> None:
    entries = [
        GazetteerEntry(canonical_id="1", entity_type="SKL", term="machine learning", source="test"),
    ]
    tokens = ["machine", "learning", "engineer"]

    lookup = build_gazetteer_lookup(tokens, entries)

    assert lookup[0]["type"] == "SKL"
    assert lookup[0]["begin"] is True
    assert lookup[1]["span_len"] == 2


def test_build_gazetteer_lookup_accepts_precompiled_matcher() -> None:
    entries = [
        GazetteerEntry(canonical_id="1", entity_type="SKL", term="machine learning", source="test"),
        GazetteerEntry(canonical_id="2", entity_type="TOOL", term="python", source="test"),
    ]
    tokens = ["Machine", "Learning", "with", "Python"]

    matcher = compile_gazetteer_entries(entries)
    lookup = build_gazetteer_lookup(tokens, matcher=matcher)

    assert lookup[0]["type"] == "SKL"
    assert lookup[1]["span_len"] == 2
    assert lookup[3]["type"] == "TOOL"


def test_run_formal_crf_experiment_returns_two_variants(tmp_path: Path) -> None:
    gazetteer_path = tmp_path / "gazetteer.jsonl"
    jd_path = tmp_path / "jd.jsonl"
    report_json_path = tmp_path / "report.json"
    report_md_path = tmp_path / "report.md"
    coverage_path = tmp_path / "coverage.json"

    write_jsonl(
        [GazetteerEntry(canonical_id="1", entity_type="SKL", term="Python", source="fixture")],
        gazetteer_path,
    )
    write_jsonl(
        [JobPosting(id="jd-1", lang="en", title="Data Scientist", description="Python is required for this role.")],
        jd_path,
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "external_dir": str(tmp_path / "external"),
                    "processed_dir": str(tmp_path / "processed"),
                    "reports_dir": str(tmp_path / "reports"),
                    "gazetteer_path": str(gazetteer_path),
                    "train_path": str(Path("data/fixtures/gold_ner.jsonl").resolve()),
                    "dev_path": str(Path("data/fixtures/gold_ner.jsonl").resolve()),
                    "test_path": str(Path("data/fixtures/gold_ner.jsonl").resolve()),
                    "jd_sample_path": str(jd_path),
                    "coverage_path": str(coverage_path),
                    "report_json_path": str(report_json_path),
                    "report_md_path": str(report_md_path),
                },
                "sources": {
                    "skillspan": {"split_urls": {}},
                    "onet": {"text_zip_url": ""},
                    "esco": {"required": False, "confirmation_url": ""},
                    "jd": {"parquet_url": "", "max_samples": 1, "min_chars": 1, "target_keywords": ["data"]},
                },
                "experiments": {
                    "crf_base": {"use_gazetteer": False, "c1": 0.1, "c2": 0.1, "max_iterations": 30},
                    "crf_gazetteer": {"use_gazetteer": True, "c1": 0.1, "c2": 0.1, "max_iterations": 30},
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    report = run_formal_crf_experiment(config_path)

    assert sorted(report["experiments"].keys()) == ["crf_base", "crf_gazetteer"]
    assert "comparison" in report
    assert report["jd_coverage"]["num_documents"] == 1
