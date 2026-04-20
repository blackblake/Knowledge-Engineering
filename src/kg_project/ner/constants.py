from __future__ import annotations

from kg_project.schema import build_bio_labels, load_schema


LABELS = build_bio_labels(entity.code for entity in load_schema().entities)
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
