from pathlib import Path

from kg_project.gazetteer import build_gazetteer, write_gazetteer_jsonl
from kg_project.labeling import label_job_corpus, sample_gold_candidates


def test_weak_label_pipeline_creates_bio_annotations(tmp_path: Path) -> None:
    gazetteer = build_gazetteer(
        esco_path=Path("data/fixtures/esco_skills.csv"),
        onet_path=Path("data/fixtures/onet_reference.tsv"),
    )
    gazetteer_path = tmp_path / "gazetteer.jsonl"
    weak_labels_path = tmp_path / "weak_labels.jsonl"
    gold_path = tmp_path / "gold_batch.jsonl"

    write_gazetteer_jsonl(gazetteer, gazetteer_path)
    examples = label_job_corpus(
        input_path=Path("data/fixtures/jd_corpus.jsonl"),
        gazetteer_path=gazetteer_path,
        output_path=weak_labels_path,
    )

    assert examples
    first = examples[0]
    assert len(first.tokens) == len(first.labels)
    assert any(label.startswith("B-") for label in first.labels)

    sampled = sample_gold_candidates(
        input_path=weak_labels_path,
        output_path=gold_path,
        sample_size=3,
        seed=7,
    )
    assert len(sampled) == 3
    assert gold_path.exists()
