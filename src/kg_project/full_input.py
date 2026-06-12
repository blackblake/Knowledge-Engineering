from __future__ import annotations

import json
from pathlib import Path
import re

import typer

from kg_project.data_types import GazetteerEntry, JobPosting, NerExample, read_jsonl, write_jsonl
from kg_project.text import tokenize_with_spans, spans_to_bio_labels
from kg_project.data_types import EntityMention


TECH_TERMS: tuple[tuple[str, str, str], ...] = (
    (".net", "SKL", ".NET"),
    ("c++", "SKL", "C++"),
    ("php", "SKL", "PHP"),
    ("java", "SKL", "Java"),
    ("python", "SKL", "Python"),
    ("javascript", "SKL", "JavaScript"),
    ("typescript", "SKL", "TypeScript"),
    ("c#", "SKL", "C#"),
    ("go", "SKL", "Go"),
    ("ruby", "SKL", "Ruby"),
    ("kotlin", "SKL", "Kotlin"),
    ("scala", "SKL", "Scala"),
    ("sql", "SKL", "SQL"),
    ("react", "TOL", "React"),
    ("react native", "TOL", "React Native"),
    ("angular", "TOL", "Angular"),
    ("vue", "TOL", "Vue"),
    ("spring", "TOL", "Spring"),
    ("django", "TOL", "Django"),
    ("flask", "TOL", "Flask"),
    ("laravel", "TOL", "Laravel"),
    ("node.js", "TOL", "Node.js"),
    ("nodejs", "TOL", "Node.js"),
    ("aws", "TOL", "AWS"),
    ("azure", "TOL", "Azure"),
    ("docker", "TOL", "Docker"),
    ("kubernetes", "TOL", "Kubernetes"),
    ("k8s", "TOL", "Kubernetes"),
    ("git", "TOL", "Git"),
    ("linux", "TOL", "Linux"),
    ("bash", "TOL", "Bash"),
    ("terraform", "TOL", "Terraform"),
    ("ansible", "TOL", "Ansible"),
    ("mysql", "TOL", "MySQL"),
    ("postgresql", "TOL", "PostgreSQL"),
    ("mongodb", "TOL", "MongoDB"),
    ("redis", "TOL", "Redis"),
    ("excel", "TOL", "Excel"),
    ("pytorch", "TOL", "PyTorch"),
    ("tensorflow", "TOL", "TensorFlow"),
    ("machine learning", "KNW", "Machine Learning"),
    ("data analysis", "KNW", "Data Analysis"),
    ("seo", "KNW", "SEO"),
    ("ppc", "KNW", "PPC"),
)

TERM_ENTITY_TYPES = {"SKL", "TOL", "KNW"}
MAX_TERM_TOKENS = 6
GENERIC_TITLES = {
    "junior",
    "middle",
    "middle+",
    "senior",
    "lead",
    "native",
    "remote",
}
GENERIC_TERMS = {
    "ability",
    "business",
    "development",
    "experience",
    "knowledge",
    "management",
    "requirements",
    "required",
    "skills",
    "software",
    "team",
    "work",
}
AMBIGUOUS_SINGLE_TOKEN_TERMS = {"access", "go", "office", "spring", "word"}


def _clean_title(title: str) -> str:
    original = title.strip()
    title = original
    title = re.sub(r"^\s*\d+(?:\s*[/_-]\s*\d+)+\s*", "", title)
    title = re.sub(r"^\s*\d+\s*[.)]\s*", "", title)
    title = re.sub(r"^\s*\d+\s+", "", title)
    title = re.sub(r"\s*\+\s*(?:welcome\s+bonus|sign[- ]?on\s+bonus|bonus|equity).*$", "", title, flags=re.I)
    title = re.sub(r"\([^)]*\)", "", title)
    title = re.sub(r"\s+", " ", title).strip(" -|/")

    if not title or title.casefold() in GENERIC_TITLES or not any(char.isalpha() for char in title):
        title = _infer_title_from_original(original) or "Software Engineer"
    elif _is_generic_software_title(title):
        inferred_tech = _first_tech_label(original)
        if inferred_tech and inferred_tech.casefold() not in title.casefold():
            title = f"{title} ({inferred_tech})"
    return title[:80] or "Unknown Role"


def _first_tech_label(text: str) -> str | None:
    lowered = text.casefold()
    for raw, _, label in sorted(TECH_TERMS, key=lambda row: len(row[0]), reverse=True):
        pattern = r"(?<![A-Za-z0-9+#.])" + re.escape(raw) + r"(?![A-Za-z0-9+#.])"
        if re.search(pattern, lowered, flags=re.I):
            return label
    return None


def _infer_title_from_original(title: str) -> str | None:
    lowered = title.casefold()
    if "react native" in lowered:
        return "React Native Developer"
    label = _first_tech_label(title)
    if not label:
        return None
    if label in {"AWS", "Azure", "Docker", "Kubernetes", "Terraform", "Ansible", "Linux"}:
        return f"{label} Engineer"
    return f"{label} Developer"


def _is_generic_software_title(title: str) -> bool:
    lowered = title.casefold()
    generic_words = ("software", "backend", "back end", "frontend", "front end", "full stack", "developer", "engineer")
    return any(word in lowered for word in generic_words)


def _entry_for_term(raw: str, entity_type: str, label: str) -> GazetteerEntry:
    key = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    return GazetteerEntry(
        canonical_id=f"local:{entity_type.lower()}:{key}",
        entity_type=entity_type,
        term=label,
        source="curated",
        description=f"Curated technology term from job descriptions: {label}",
        aliases=(raw,),
    )


def _occupation_entry(title: str) -> GazetteerEntry:
    key = re.sub(r"[^a-z0-9]+", "_", title.casefold()).strip("_")[:80] or "unknown_role"
    return GazetteerEntry(
        canonical_id=f"local:occupation:{key}",
        entity_type="OCC",
        term=title,
        source="job_title",
        description="Occupation title extracted from the job posting title.",
    )


def _term_key(text: str) -> tuple[str, ...]:
    tokens, _ = tokenize_with_spans(text)
    return tuple(token.casefold() for token in tokens)


def _is_indexable_term(entry: GazetteerEntry, variant: str, key: tuple[str, ...]) -> bool:
    if entry.entity_type not in TERM_ENTITY_TYPES or not key or len(key) > MAX_TERM_TOKENS:
        return False
    normalized = " ".join(key)
    if normalized in GENERIC_TERMS:
        return False
    if entry.source != "curated" and len(key) == 1:
        token = key[0]
        if len(token) < 3 or token in AMBIGUOUS_SINGLE_TOKEN_TERMS:
            return False
    return any(char.isalpha() for char in variant)


def _build_term_index(entries: list[GazetteerEntry]) -> dict[tuple[str, ...], GazetteerEntry]:
    term_index: dict[tuple[str, ...], GazetteerEntry] = {}
    for entry in entries:
        for variant in entry.variants():
            key = _term_key(variant)
            if _is_indexable_term(entry, variant, key):
                term_index.setdefault(key, entry)

    # Curated technology terms should win over broad O*NET labels when both exist.
    for raw, entity_type, label in TECH_TERMS:
        entry = _entry_for_term(raw, entity_type, label)
        for variant in entry.variants():
            key = _term_key(variant)
            if _is_indexable_term(entry, variant, key):
                term_index[key] = entry
    return term_index


def build_enriched_gazetteer(base_path: Path, output_path: Path, postings: list[JobPosting]) -> list[GazetteerEntry]:
    entries = [GazetteerEntry(**row) for row in read_jsonl(base_path)] if base_path.exists() else []
    by_key: dict[tuple[str, str], GazetteerEntry] = {
        (entry.entity_type, entry.term.casefold()): entry for entry in entries
    }

    for raw, entity_type, label in TECH_TERMS:
        by_key.setdefault((entity_type, label.casefold()), _entry_for_term(raw, entity_type, label))

    for posting in postings:
        title = _clean_title(posting.title)
        by_key.setdefault(("OCC", title.casefold()), _occupation_entry(title))

    merged = sorted(by_key.values(), key=lambda e: (e.entity_type, e.term.casefold()))
    write_jsonl(merged, output_path)
    return merged


def _find_term_mentions(text: str, term_index: dict[tuple[str, ...], GazetteerEntry]) -> list[EntityMention]:
    mentions: list[EntityMention] = []
    occupied: list[tuple[int, int]] = []
    tokens, token_spans = tokenize_with_spans(text)
    normalized_tokens = [token.casefold() for token in tokens]
    for start_index in range(len(tokens)):
        max_end = min(len(tokens), start_index + MAX_TERM_TOKENS)
        for end_index in range(max_end, start_index, -1):
            key = tuple(normalized_tokens[start_index:end_index])
            entry = term_index.get(key)
            if not entry:
                continue
            start, _ = token_spans[start_index]
            _, end = token_spans[end_index - 1]
            if any(not (end <= a or start >= b) for a, b in occupied):
                break
            occupied.append((start, end))
            mentions.append(
                EntityMention(
                    text=text[start:end],
                    start=start,
                    end=end,
                    entity_type=entry.entity_type,
                    canonical_id=entry.canonical_id,
                    source=entry.source,
                )
            )
            break
    return sorted(mentions, key=lambda m: m.start)


def _make_example(
    posting: JobPosting,
    title: str,
    sentence: str,
    index: int,
    term_index: dict[tuple[str, ...], GazetteerEntry],
) -> NerExample:
    text = f"{title} requires {sentence.strip()}"
    tokens, token_spans = tokenize_with_spans(text)
    occ_end = len(title)
    mentions = [
        EntityMention(
            text=title,
            start=0,
            end=occ_end,
            entity_type="OCC",
            canonical_id=_occupation_entry(title).canonical_id,
            source="job_title",
        )
    ]
    mentions.extend(_find_term_mentions(text, term_index))
    labels = spans_to_bio_labels(token_spans, mentions)
    return NerExample(id=f"{posting.id}#full#{index}", lang=posting.lang, text=text, tokens=tokens, labels=labels)


def build_full_weak_labels(
    jobs_path: Path,
    base_gazetteer_path: Path,
    output_ner_path: Path,
    output_gazetteer_path: Path,
    max_jobs: int = 10000,
    max_sentences_per_job: int = 6,
) -> dict:
    postings = [JobPosting(**row) for row in read_jsonl(jobs_path)[:max_jobs]]
    enriched = build_enriched_gazetteer(base_gazetteer_path, output_gazetteer_path, postings)
    term_index = _build_term_index(enriched)

    examples: list[NerExample] = []
    sentence_splitter = re.compile(r"(?<=[.!?])\s+|\r?\n+")
    for posting in postings:
        title = _clean_title(posting.title)
        emitted = 0
        for sentence in sentence_splitter.split(posting.description):
            if emitted >= max_sentences_per_job:
                break
            if not sentence.strip() or not _find_term_mentions(sentence, term_index):
                continue
            examples.append(_make_example(posting, title, sentence, emitted, term_index))
            emitted += 1

    write_jsonl(examples, output_ner_path)
    return {
        "jobs": len(postings),
        "examples": len(examples),
        "gazetteer_entries": len(enriched),
        "indexed_terms": len(term_index),
        "ner_path": str(output_ner_path),
        "gazetteer_path": str(output_gazetteer_path),
    }


def build_full_input_command(
    jobs: Path = typer.Option(Path("data/processed/crf/jd_sample.jsonl"), help="Processed JD sample JSONL."),
    base_gazetteer: Path = typer.Option(
        Path("data/processed/crf/formal_gazetteer.jsonl"),
        help="Base formal gazetteer JSONL.",
    ),
    output_ner: Path = typer.Option(Path("data/processed/full_weak_ner.jsonl"), help="Output full weak NER JSONL."),
    output_gazetteer: Path = typer.Option(
        Path("data/processed/full_gazetteer.jsonl"),
        help="Output enriched gazetteer JSONL.",
    ),
    max_jobs: int = typer.Option(10000, help="Maximum job postings to process."),
    max_sentences_per_job: int = typer.Option(6, help="Maximum matched sentences per job."),
) -> None:
    payload = build_full_weak_labels(
        jobs_path=jobs,
        base_gazetteer_path=base_gazetteer,
        output_ner_path=output_ner,
        output_gazetteer_path=output_gazetteer,
        max_jobs=max_jobs,
        max_sentences_per_job=max_sentences_per_job,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
