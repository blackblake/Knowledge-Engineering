from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True, slots=True)
class EntityTypeDefinition:
    code: str
    name: str
    name_zh: str
    description: str


@dataclass(frozen=True, slots=True)
class RelationDefinition:
    code: str
    source: str
    target: str


@dataclass(frozen=True, slots=True)
class EdgeAttributeDefinition:
    name: str
    type: str


@dataclass(frozen=True, slots=True)
class SchemaDefinition:
    name: str
    version: int
    entities: tuple[EntityTypeDefinition, ...]
    relations: tuple[RelationDefinition, ...]
    edge_attributes: tuple[EdgeAttributeDefinition, ...]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_schema_path() -> Path:
    return project_root() / "config" / "ontology.yaml"


def load_schema(path: Path | None = None) -> SchemaDefinition:
    schema_path = path or default_schema_path()
    payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    entities = tuple(EntityTypeDefinition(**row) for row in payload["entities"])
    relations = tuple(RelationDefinition(**row) for row in payload["relations"])
    edge_attributes = tuple(EdgeAttributeDefinition(**row) for row in payload["edge_attributes"])
    return SchemaDefinition(
        name=payload["name"],
        version=payload["version"],
        entities=entities,
        relations=relations,
        edge_attributes=edge_attributes,
    )


def build_bio_labels(entity_codes: Iterable[str]) -> list[str]:
    labels = ["O"]
    for code in entity_codes:
        labels.append(f"B-{code}")
        labels.append(f"I-{code}")
    return labels
