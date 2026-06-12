from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from seqeval.metrics import f1_score, precision_score, recall_score

from kg_project.data_sources import default_crf_config_path, load_crf_experiment_config, resolve_crf_paths
from kg_project.data_types import NerExample
from kg_project.ner.data import load_ner_examples
from kg_project.ner.neural import BertCRF, BiLSTMCRF


@dataclass(frozen=True, slots=True)
class NeuralReport:
    precision: float
    recall: float
    f1: float
    num_examples: int


def _build_label_map(examples: Iterable[NerExample]) -> tuple[dict[str, int], list[str]]:
    labels = sorted({label for example in examples for label in example.labels})
    if "O" in labels:
        labels.remove("O")
        labels.insert(0, "O")
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    return label_to_id, labels


def _build_vocab(examples: Iterable[NerExample], min_freq: int = 1) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for example in examples:
        counter.update(example.tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in counter.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def _encode_tokens(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]


def _pad_sequences(sequences: list[list[int]], pad_value: int = 0) -> Tensor:
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)


def _pad_masks(sequences: list[list[int]]) -> Tensor:
    max_len = max(len(seq) for seq in sequences)
    masks = [[True] * len(seq) + [False] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(masks, dtype=torch.bool)


def _labels_to_ids(labels: list[str], label_to_id: dict[str, int]) -> list[int]:
    return [label_to_id[label] for label in labels]


def _evaluate_predictions(y_true: list[list[str]], y_pred: list[list[str]]) -> NeuralReport:
    return NeuralReport(
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        num_examples=len(y_true),
    )


def train_bilstm_crf(
    train_examples: list[NerExample],
    dev_examples: list[NerExample],
    test_examples: list[NerExample],
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
) -> dict:
    label_to_id, id_to_label = _build_label_map(train_examples)
    vocab = _build_vocab(train_examples)
    model = BiLSTMCRF(vocab_size=len(vocab), num_labels=len(label_to_id))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def make_loader(examples: list[NerExample]) -> DataLoader:
        def collate(batch: list[NerExample]) -> dict[str, Tensor]:
            input_ids = [_encode_tokens(item.tokens, vocab) for item in batch]
            labels = [_labels_to_ids(item.labels, label_to_id) for item in batch]
            return {
                "input_ids": _pad_sequences(input_ids),
                "labels": _pad_sequences(labels),
                "mask": _pad_masks(input_ids),
            }

        return DataLoader(examples, batch_size=batch_size, shuffle=True, collate_fn=collate)

    model.train()
    for _ in range(epochs):
        for batch in make_loader(train_examples):
            loss = model.loss(batch["input_ids"], batch["labels"], batch["mask"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def predict(examples: list[NerExample]) -> list[list[str]]:
        model.eval()
        predictions: list[list[str]] = []
        with torch.no_grad():
            for batch in make_loader(examples):
                decoded = model.decode(batch["input_ids"], batch["mask"])
                for seq, mask in zip(decoded, batch["mask"]):
                    length = int(mask.sum().item())
                    predictions.append([id_to_label[idx] for idx in seq[:length]])
        return predictions

    dev_pred = predict(dev_examples)
    test_pred = predict(test_examples)

    return {
        "dev": _evaluate_predictions([item.labels for item in dev_examples], dev_pred),
        "test": _evaluate_predictions([item.labels for item in test_examples], test_pred),
    }


def train_bert_crf(
    train_examples: list[NerExample],
    dev_examples: list[NerExample],
    test_examples: list[NerExample],
    model_name: str = "bert-base-uncased",
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 256,
) -> dict:
    label_to_id, id_to_label = _build_label_map(train_examples)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertCRF(model_name=model_name, num_labels=len(label_to_id))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def encode(example: NerExample) -> dict[str, Tensor]:
        encoding = tokenizer(
            example.tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids()
        labels: list[int] = []
        mask: list[bool] = []
        prev = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(0)
                mask.append(False)
            elif word_id != prev:
                labels.append(label_to_id[example.labels[word_id]])
                mask.append(True)
            else:
                labels.append(0)
                mask.append(False)
            prev = word_id
        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(labels, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }

    def make_loader(examples: list[NerExample]) -> DataLoader:
        encoded = [encode(example) for example in examples]

        def collate(indices: list[int]) -> dict[str, Tensor]:
            input_ids = torch.stack([encoded[idx]["input_ids"] for idx in indices])
            attention_mask = torch.stack([encoded[idx]["attention_mask"] for idx in indices])
            labels = torch.stack([encoded[idx]["labels"] for idx in indices])
            mask = torch.stack([encoded[idx]["mask"] for idx in indices])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "mask": mask,
            }

        return DataLoader(list(range(len(encoded))), batch_size=batch_size, shuffle=True, collate_fn=collate)

    model.train()
    for _ in range(epochs):
        for batch in make_loader(train_examples):
            loss = model.loss(batch["input_ids"], batch["attention_mask"], batch["labels"], batch["mask"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def predict(examples: list[NerExample]) -> list[list[str]]:
        model.eval()
        predictions: list[list[str]] = []
        with torch.no_grad():
            for batch in make_loader(examples):
                decoded = model.decode(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["mask"],
                )
                for seq, mask in zip(decoded, batch["mask"]):
                    length = int(mask.sum().item())
                    predictions.append([id_to_label[idx] for idx in seq[:length]])
        return predictions

    dev_pred = predict(dev_examples)
    test_pred = predict(test_examples)

    return {
        "dev": _evaluate_predictions([item.labels for item in dev_examples], dev_pred),
        "test": _evaluate_predictions([item.labels for item in test_examples], test_pred),
    }


def run_neural_experiments(
    config_path: Path | None = None,
    bert_model: str = "bert-base-uncased",
    bert_epochs: int = 1,
    bilstm_epochs: int = 3,
) -> dict:
    config = load_crf_experiment_config(config_path)
    paths = resolve_crf_paths(config)

    train_examples = load_ner_examples(paths["train_path"])
    dev_examples = load_ner_examples(paths["dev_path"])
    test_examples = load_ner_examples(paths["test_path"])

    bilstm_report = train_bilstm_crf(
        train_examples=train_examples,
        dev_examples=dev_examples,
        test_examples=test_examples,
        epochs=bilstm_epochs,
    )

    bert_report = train_bert_crf(
        train_examples=train_examples,
        dev_examples=dev_examples,
        test_examples=test_examples,
        model_name=bert_model,
        epochs=bert_epochs,
    )

    return {
        "bilstm_crf": {
            "dev": bilstm_report["dev"].__dict__,
            "test": bilstm_report["test"].__dict__,
        },
        "bert_crf": {
            "dev": bert_report["dev"].__dict__,
            "test": bert_report["test"].__dict__,
        },
    }


def run_neural_experiments_command(
    config: Path = default_crf_config_path(),
    bert_model: str = "bert-base-uncased",
    bert_epochs: int = 1,
    bilstm_epochs: int = 3,
) -> None:
    report = run_neural_experiments(config, bert_model=bert_model, bert_epochs=bert_epochs, bilstm_epochs=bilstm_epochs)
    paths = resolve_crf_paths(load_crf_experiment_config(config))
    output_path = Path(paths["reports_dir"]) / "neural_experiment.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
