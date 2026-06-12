from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from zipfile import ZipFile

import pandas as pd
import typer

from kg_project.data_sources import default_crf_config_path, load_crf_experiment_config, resolve_crf_paths
from kg_project.data_types import GazetteerEntry, JobPosting, NerExample, write_jsonl


def convert_skillspan_record(record: dict) -> NerExample:
    labels: list[str] = []
    for skill_tag, knowledge_tag in zip(record["tags_skill"], record["tags_knowledge"]):
        if skill_tag != "O":
            labels.append(skill_tag.replace("B", "B-SKL").replace("I", "I-SKL"))
        elif knowledge_tag != "O":
            labels.append(knowledge_tag.replace("B", "B-KNW").replace("I", "I-KNW"))
        else:
            labels.append("O")
    return NerExample(
        id=str(record.get("idx", "unknown")),
        lang="en",
        text=" ".join(record["tokens"]),
        tokens=record["tokens"],
        labels=labels,
        metadata={"source": record.get("source", "skillspan")},
    )


def convert_skillspan_split(input_path: Path, output_path: Path) -> list[NerExample]:
    rows: list[NerExample] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(convert_skillspan_record(json.loads(line)))
    write_jsonl(rows, output_path)
    return rows


def _parse_onet_table(lines: list[str], entity_type: str, source_name: str) -> list[GazetteerEntry]:
    reader = csv.DictReader(lines, delimiter="\t")
    label_candidates = ["Element Name", "Commodity Title", "Title", "O*NET-SOC Title"]
    description_candidates = ["Description", "Commodity Description"]
    entries: list[GazetteerEntry] = []
    for row in reader:
        label = next((row.get(key, "").strip() for key in label_candidates if row.get(key, "").strip()), "")
        if not label:
            continue
        description = next((row.get(key, "").strip() for key in description_candidates if row.get(key, "").strip()), "")
        code = row.get("O*NET-SOC Code", "") or row.get("Commodity Code", "") or row.get("Element ID", "") or label
        entries.append(
            GazetteerEntry(
                canonical_id=f"onet:{entity_type}:{code}",
                entity_type=entity_type,
                term=label,
                source=source_name,
                description=description,
            )
        )
    return entries


def build_formal_gazetteer(onet_zip_path: Path, esco_path: Path | None = None) -> list[GazetteerEntry]:
    entries: list[GazetteerEntry] = []
    name_map = {
        "knowledge": "KNW",
        "skills": "SKL",
        "abilities": "ABL",
        "technology skills": "TOL",
        "tools": "TOL",
        "occupation": "OCC",
    }
    with ZipFile(onet_zip_path) as archive:
        for member in archive.namelist():
            lowered = member.lower()
            entity_type = next((value for key, value in name_map.items() if key in lowered), None)
            if not entity_type or not lowered.endswith(".txt"):
                continue
            with archive.open(member) as handle:
                decoded = [line.decode("utf-8", errors="ignore") for line in handle]
                entries.extend(_parse_onet_table(decoded, entity_type, "O*NET"))

    if esco_path and esco_path.exists() and esco_path.suffix.lower() == ".csv":
        with esco_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = row.get("preferredLabel", "").strip()
                if not label:
                    continue
                raw_type = (row.get("conceptType", "") or row.get("entityType", "")).casefold()
                entity_type = {
                    "skill": "SKL",
                    "occupation": "OCC",
                    "knowledge": "KNW",
                    "qualification": "QLF",
                    "tool": "TOL",
                }.get(raw_type, "SKL")
                aliases = tuple(item.strip() for item in re.split(r"[|;]", row.get("altLabels", "")) if item.strip())
                entries.append(
                    GazetteerEntry(
                        canonical_id=row.get("conceptUri") or row.get("uri") or label,
                        entity_type=entity_type,
                        term=label,
                        source="ESCO",
                        description=row.get("description", ""),
                        aliases=aliases,
                    )
                )
    return entries


def filter_jd_rows(jd_parquet_path: Path, max_samples: int, min_chars: int, target_keywords: list[str]) -> list[JobPosting]:
    frame = pd.read_parquet(jd_parquet_path)
    frame = frame.rename(columns={"Position": "title", "Long Description": "description", "id": "id"})
    frame["title"] = frame["title"].fillna("")
    frame["description"] = frame["description"].fillna("")
    lang_column = "Long Description_lang" if "Long Description_lang" in frame.columns else None
    if lang_column:
        frame = frame[frame[lang_column].fillna("en").str.lower() == "en"]
    frame = frame[frame["description"].str.len() >= min_chars]
    keywords = tuple(keyword.casefold() for keyword in target_keywords)

    def is_target(row: pd.Series) -> bool:
        haystack = " ".join(str(row.get(column, "")) for column in ["title", "description", "Primary Keyword"]).casefold()
        return any(keyword in haystack for keyword in keywords)

    filtered = frame[frame.apply(is_target, axis=1)].head(max_samples)
    rows: list[JobPosting] = []
    for idx, row in filtered.iterrows():
        rows.append(
            JobPosting(
                id=str(row.get("id", idx)),
                lang="en",
                title=str(row.get("title", "")),
                description=str(row.get("description", "")),
            )
        )
    return rows


def prepare_crf_data(config_path: Path | None = None) -> dict[str, str]:
    config = load_crf_experiment_config(config_path)
    paths = resolve_crf_paths(config)
    external_dir = paths["external_dir"]

    for split in ["train", "dev", "test"]:
        convert_skillspan_split(external_dir / "skillspan" / f"{split}.json", paths[f"{split}_path"])

    esco_file = external_dir / "esco" / "esco_download.csv"
    gazetteer = build_formal_gazetteer(
        onet_zip_path=external_dir / "onet" / "db_text.zip",
        esco_path=esco_file if esco_file.exists() else None,
    )
    write_jsonl(gazetteer, paths["gazetteer_path"])

    jd_rows = filter_jd_rows(
        jd_parquet_path=external_dir / "jd" / "job_descriptions.parquet",
        max_samples=config["sources"]["jd"]["max_samples"],
        min_chars=config["sources"]["jd"]["min_chars"],
        target_keywords=config["sources"]["jd"]["target_keywords"],
    )
    write_jsonl(jd_rows, paths["jd_sample_path"])

    return {
        "train": str(paths["train_path"]),
        "dev": str(paths["dev_path"]),
        "test": str(paths["test_path"]),
        "gazetteer": str(paths["gazetteer_path"]),
        "jd_sample": str(paths["jd_sample_path"]),
    }


def prepare_crf_data_command(
    config: Path = typer.Option(default_crf_config_path(), help="Path to the CRF experiment config."),
) -> None:
    outputs = prepare_crf_data(config_path=config)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")
