from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable

from kg_project.data_types import NerExample, read_jsonl
from kg_project.relation_extraction import extract_entity_spans, extract_relations


@dataclass(frozen=True, slots=True)
class RelationExample:
    text: str
    head: str
    tail: str
    relation: str


def build_relation_examples(
    examples: Iterable[NerExample],
    negative_ratio: int = 1,
) -> list[RelationExample]:
    relation_examples: list[RelationExample] = []
    rng = random.Random(42)

    for example in examples:
        spans = extract_entity_spans(example)
        relations = extract_relations(example)
        positive_pairs = {(rel.head.text, rel.tail.text): rel.relation for rel in relations}

        for (head, tail), relation in positive_pairs.items():
            relation_examples.append(
                RelationExample(
                    text=example.text,
                    head=head,
                    tail=tail,
                    relation=relation,
                )
            )

        candidates = [(head.text, tail.text) for head in spans if head.entity_type == "OCC" for tail in spans]
        negatives = [pair for pair in candidates if pair not in positive_pairs and pair[0] != pair[1]]
        rng.shuffle(negatives)
        for head, tail in negatives[: negative_ratio * max(len(positive_pairs), 1)]:
            relation_examples.append(
                RelationExample(
                    text=example.text,
                    head=head,
                    tail=tail,
                    relation="NO_RELATION",
                )
            )
    return relation_examples


def write_relation_dataset(examples: Iterable[RelationExample], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


def load_relation_dataset(path: Path) -> list[RelationExample]:
    rows = read_jsonl(path)
    return [RelationExample(**row) for row in rows]


def train_bert_re(
    dataset_path: Path,
    model_name: str,
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Transformers/torch are required for BERT-RE.") from exc

    dataset = load_relation_dataset(dataset_path)
    labels = sorted({row.relation for row in dataset})
    label_to_id = {label: idx for idx, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))
    model.train()

    def encode(example: RelationExample) -> dict:
        text = f"[E1] {example.head} [/E1] [E2] {example.tail} [/E2] {example.text}"
        return tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")

    encoded = [encode(row) for row in dataset]
    labels_tensor = torch.tensor([label_to_id[row.relation] for row in dataset], dtype=torch.long)

    def collate(indices: list[int]) -> dict:
        batch = {"input_ids": [], "attention_mask": []}
        for idx in indices:
            batch["input_ids"].append(encoded[idx]["input_ids"][0])
            batch["attention_mask"].append(encoded[idx]["attention_mask"][0])
        labels = labels_tensor[indices]
        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "labels": labels,
        }

    indices = list(range(len(dataset)))
    loader = DataLoader(indices, batch_size=batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        for batch in loader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "label_map.json").write_text(json.dumps(label_to_id, indent=2), encoding="utf-8")
    return {"labels": labels, "num_examples": len(dataset)}


def prepare_re_dataset_command(
    ner: Path,
    output: Path,
    negative_ratio: int = 1,
) -> None:
    examples = [NerExample(**row) for row in read_jsonl(ner)]
    dataset = build_relation_examples(examples, negative_ratio=negative_ratio)
    write_relation_dataset(dataset, output)


def train_bert_re_command(
    dataset: Path,
    model_name: str,
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
) -> None:
    train_bert_re(
        dataset_path=dataset,
        model_name=model_name,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
