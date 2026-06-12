from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import typer


def _load_texts(input_path: Path) -> list[dict]:
    rows = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def uie_extract(texts: Iterable[str], schema: dict) -> list[dict]:
    try:
        from paddlenlp import Taskflow
    except ImportError as exc:
        raise RuntimeError("PaddleNLP is required for UIE extraction.") from exc

    try:
        extractor = Taskflow(
            "information_extraction",
            schema=schema,
            model="uie-base",
            use_static_model=False,
        )
    except TypeError:
        extractor = Taskflow("information_extraction", schema=schema, model="uie-base")
    return extractor(list(texts))


def llm_extract_openai(
    texts: Iterable[str],
    schema: dict,
    api_key: str,
    api_base: str,
    model: str,
) -> list[dict]:
    results: list[dict] = []
    system_prompt = (
        "You are an information extraction system. "
        "Extract entities and relations based on the provided schema. "
        "Return strict JSON with fields: entities (list) and relations (list)."
    )
    for text in texts:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps({"schema": schema, "text": text}, ensure_ascii=False),
                },
            ],
            "temperature": 0.2,
        }
        request = Request(
            f"{api_base.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urlopen(request, timeout=60) as response:
            content = response.read().decode("utf-8")
        results.append({"text": text, "response": json.loads(content)})
    return results


def uie_extract_command(
    input: Path = typer.Option(..., help="Input JSONL with {text} records."),
    output: Path = typer.Option(..., help="Output JSONL path."),
    schema: str = typer.Option(..., help="JSON string schema for UIE."),
) -> None:
    rows = _load_texts(input)
    texts = [row["text"] for row in rows]
    payload = uie_extract(texts, json.loads(schema))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row, result in zip(rows, payload):
            handle.write(json.dumps({"id": row.get("id"), "text": row["text"], "result": result}, ensure_ascii=False))
            handle.write("\n")


def llm_extract_command(
    input: Path = typer.Option(..., help="Input JSONL with {text} records."),
    output: Path = typer.Option(..., help="Output JSONL path."),
    schema: str = typer.Option(..., help="JSON string schema for LLM extraction."),
    api_key: str = typer.Option(..., help="OpenAI-compatible API key."),
    api_base: str = typer.Option("https://api.openai.com", help="OpenAI-compatible base URL."),
    model: str = typer.Option("gpt-4o-mini", help="Model name."),
) -> None:
    rows = _load_texts(input)
    texts = [row["text"] for row in rows]
    payload = llm_extract_openai(texts, json.loads(schema), api_key, api_base, model)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row, result in zip(rows, payload):
            handle.write(json.dumps({"id": row.get("id"), "text": row["text"], "result": result}, ensure_ascii=False))
            handle.write("\n")
