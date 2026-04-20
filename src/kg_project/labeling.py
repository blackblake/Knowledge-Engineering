from __future__ import annotations

from pathlib import Path
import random

import typer

from kg_project.data_types import JobPosting, NerExample, read_jsonl, write_jsonl
from kg_project.gazetteer import read_gazetteer_jsonl
from kg_project.text import split_sentences, spans_to_bio_labels, tokenize_with_spans


def _load_job_postings(path: Path) -> list[JobPosting]:
    return [JobPosting(**row) for row in read_jsonl(path)]


def _load_ner_examples(path: Path) -> list[NerExample]:
    return [NerExample(**row) for row in read_jsonl(path)]


def label_job_corpus(input_path: Path, gazetteer_path: Path, output_path: Path) -> list[NerExample]:
    postings = _load_job_postings(input_path)
    gazetteer = read_gazetteer_jsonl(gazetteer_path)
    labeled: list[NerExample] = []
    for posting in postings:
        for sentence_index, sentence in enumerate(split_sentences(posting.description)):
            tokens, token_spans = tokenize_with_spans(sentence)
            mentions = gazetteer.find_mentions(sentence)
            labels = spans_to_bio_labels(token_spans, mentions)
            labeled.append(
                NerExample(
                    id=f"{posting.id}#{sentence_index}",
                    lang=posting.lang,
                    text=sentence,
                    tokens=tokens,
                    labels=labels,
                )
            )
    write_jsonl(labeled, output_path)
    return labeled


def _bio_to_doccano_labels(text: str, tokens: list[str], labels: list[str]) -> list[list[object]]:
    char_offset = 0
    token_offsets: list[tuple[int, int]] = []
    for token in tokens:
        start = text.find(token, char_offset)
        end = start + len(token)
        token_offsets.append((start, end))
        char_offset = end

    spans: list[list[object]] = []
    current_label = ""
    current_start = -1
    current_end = -1

    for idx, label in enumerate(labels):
        if label == "O":
            if current_label:
                spans.append([current_start, current_end, current_label])
                current_label = ""
            continue

        prefix, entity_type = label.split("-", 1)
        start, end = token_offsets[idx]
        if prefix == "B" or entity_type != current_label:
            if current_label:
                spans.append([current_start, current_end, current_label])
            current_label = entity_type
            current_start = start
            current_end = end
        else:
            current_end = end

    if current_label:
        spans.append([current_start, current_end, current_label])
    return spans


def sample_gold_candidates(input_path: Path, output_path: Path, sample_size: int, seed: int = 42) -> list[dict]:
    examples = _load_ner_examples(input_path)
    candidates = [example for example in examples if any(label != "O" for label in example.labels)]
    rng = random.Random(seed)
    selected = candidates if len(candidates) <= sample_size else rng.sample(candidates, sample_size)
    records = []
    for example in selected:
        records.append(
            {
                "id": example.id,
                "text": example.text,
                "labels": _bio_to_doccano_labels(example.text, example.tokens, example.labels),
                "meta": {
                    "lang": example.lang,
                    "tokens": example.tokens,
                    "weak_labels": example.labels,
                },
            }
        )
    write_jsonl(records, output_path)
    return records


def weak_label_command(
    jobs: Path = typer.Option(..., help="Input JD corpus in JSONL format."),
    gazetteer: Path = typer.Option(..., help="Merged gazetteer JSONL file."),
    output: Path = typer.Option(..., help="Output path for weakly labeled BIO examples."),
) -> None:
    label_job_corpus(input_path=jobs, gazetteer_path=gazetteer, output_path=output)


def sample_gold_command(
    input: Path = typer.Option(..., help="Weakly labeled input JSONL."),
    output: Path = typer.Option(..., help="Output path for the gold candidate batch."),
    sample_size: int = typer.Option(100, help="Number of examples to sample."),
    seed: int = typer.Option(42, help="Sampling seed."),
) -> None:
    sample_gold_candidates(input_path=input, output_path=output, sample_size=sample_size, seed=seed)
