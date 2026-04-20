from __future__ import annotations

from pathlib import Path

from kg_project.data_types import NerExample, read_jsonl


def load_ner_examples(path: Path) -> list[NerExample]:
    return [NerExample(**row) for row in read_jsonl(path)]
