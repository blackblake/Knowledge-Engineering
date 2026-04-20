from pathlib import Path

from kg_project.schema import build_bio_labels, load_schema


def test_schema_matches_proposal_counts() -> None:
    schema = load_schema(Path("config/ontology.yaml"))
    assert len(schema.entities) == 7
    assert len(schema.relations) == 8


def test_bio_label_space_has_15_tags() -> None:
    schema = load_schema(Path("config/ontology.yaml"))
    labels = build_bio_labels(entity.code for entity in schema.entities)
    assert len(labels) == 15
    assert labels[0] == "O"
    assert "B-OCC" in labels
    assert "I-QLF" in labels
