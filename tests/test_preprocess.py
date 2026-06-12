from pathlib import Path

from kg_project.preprocess import convert_skillspan_record


def test_convert_skillspan_record_maps_skill_and_knowledge_tags() -> None:
    record = {
        "idx": 7,
        "tokens": ["Python", "and", "statistics"],
        "tags_skill": ["B", "O", "O"],
        "tags_knowledge": ["O", "O", "B"],
        "source": "tech",
    }

    example = convert_skillspan_record(record)

    assert example.id == "7"
    assert example.labels == ["B-SKL", "O", "B-KNW"]
    assert example.metadata["source"] == "tech"
