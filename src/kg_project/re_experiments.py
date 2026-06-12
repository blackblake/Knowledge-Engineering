from __future__ import annotations

import json
from pathlib import Path

import typer
from seqeval.metrics import f1_score

from kg_project.data_types import NerExample, read_jsonl
from kg_project.relation_extraction import extract_relations_from_dataset
from kg_project.relation_models import train_bert_re, prepare_re_dataset_command
from kg_project.llm_extraction import llm_extract_openai


def _load_ner(path: Path) -> list[NerExample]:
    return [NerExample(**row) for row in read_jsonl(path)]


def _rule_re_score(examples: list[NerExample]) -> dict:
    rels = extract_relations_from_dataset(examples)
    # proxy metric: relation density + non-empty proficiency extraction ratio
    if not rels:
        return {"num_relations": 0, "proficiency_coverage": 0.0}
    prof = sum(1 for r in rels if r.proficiency_level is not None)
    return {
        "num_relations": len(rels),
        "proficiency_coverage": round(prof / len(rels), 4),
    }


def _llm_re_score(examples: list[NerExample], api_key: str, api_base: str, model: str) -> dict:
    texts = [e.text for e in examples[:50]]
    schema = {
        "entities": ["OCC", "SKL", "KNW", "ABL", "TSK", "TOL", "QLF"],
        "relations": [
            "REQUIRES_SKILL",
            "REQUIRES_KNOWLEDGE",
            "REQUIRES_ABILITY",
            "HAS_TASK",
            "USES_TOOL",
            "REQUIRES_QUALIFICATION",
        ],
    }
    payload = llm_extract_openai(texts, schema, api_key=api_key, api_base=api_base, model=model)
    valid = 0
    for row in payload:
        try:
            _ = row["response"]["choices"][0]["message"]["content"]
            valid += 1
        except Exception:
            continue
    return {"num_samples": len(texts), "valid_json_ratio": round(valid / max(len(texts), 1), 4)}


def run_re_comparison(
    ner_path: Path,
    re_dataset_path: Path,
    bert_output_dir: Path,
    bert_model: str = "bert-base-uncased",
    run_llm: bool = False,
    api_key: str | None = None,
    api_base: str = "https://api.openai.com",
    llm_model: str = "gpt-4o-mini",
) -> dict:
    examples = _load_ner(ner_path)
    report = {"rule_re": _rule_re_score(examples)}

    prepare_re_dataset_command(ner=ner_path, output=re_dataset_path)
    bert_meta = train_bert_re(re_dataset_path, bert_model, bert_output_dir, epochs=1)
    report["bert_re"] = {"status": "trained", **bert_meta}

    if run_llm and api_key:
        report["llm_re"] = _llm_re_score(examples, api_key=api_key, api_base=api_base, model=llm_model)
    else:
        report["llm_re"] = {"status": "skipped", "reason": "api_key_missing_or_disabled"}
    return report


def run_re_comparison_command(
    ner: Path = typer.Option(..., help="NER jsonl input."),
    dataset: Path = typer.Option(..., help="Output RE dataset jsonl."),
    bert_output_dir: Path = typer.Option(..., help="BERT-RE output dir."),
    output: Path = typer.Option(..., help="Comparison report output path."),
    bert_model: str = typer.Option("bert-base-uncased", help="BERT model name."),
    run_llm: bool = typer.Option(False, help="Whether to run LLM RE evaluation."),
    api_key: str | None = typer.Option(None, help="OpenAI-compatible API key."),
    api_base: str = typer.Option("https://api.openai.com", help="API base URL."),
    llm_model: str = typer.Option("gpt-4o-mini", help="LLM model name."),
) -> None:
    report = run_re_comparison(
        ner_path=ner,
        re_dataset_path=dataset,
        bert_output_dir=bert_output_dir,
        bert_model=bert_model,
        run_llm=run_llm,
        api_key=api_key,
        api_base=api_base,
        llm_model=llm_model,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
