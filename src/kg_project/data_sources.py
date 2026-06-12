from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import typer
import yaml

from kg_project.schema import project_root


@dataclass(frozen=True, slots=True)
class DownloadTarget:
    name: str
    url: str
    target_path: Path
    required: bool = True


def default_crf_config_path() -> Path:
    return project_root() / "config" / "crf_experiment.yaml"


def load_crf_experiment_config(path: Path | None = None) -> dict:
    config_path = path or default_crf_config_path()
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def resolve_crf_paths(config: dict) -> dict[str, Path]:
    return {name: project_root() / raw_path for name, raw_path in config["paths"].items()}


def build_download_targets(config: dict) -> list[DownloadTarget]:
    paths = resolve_crf_paths(config)
    external_dir = paths["external_dir"]
    targets = [
        DownloadTarget(
            name=f"skillspan_{split}",
            url=url,
            target_path=external_dir / "skillspan" / f"{split}.json",
        )
        for split, url in config["sources"]["skillspan"]["split_urls"].items()
    ]
    targets.extend(
        [
            DownloadTarget(
                name="onet_text_zip",
                url=config["sources"]["onet"]["text_zip_url"],
                target_path=external_dir / "onet" / "db_text.zip",
            ),
            DownloadTarget(
                name="esco_download",
                url=config["sources"]["esco"]["confirmation_url"],
                target_path=external_dir / "esco" / "esco_download.bin",
                required=config["sources"]["esco"].get("required", False),
            ),
            DownloadTarget(
                name="jd_parquet",
                url=config["sources"]["jd"]["parquet_url"],
                target_path=external_dir / "jd" / "job_descriptions.parquet",
            ),
        ]
    )
    return targets


def download_file(url: str, target_path: Path, force: bool = False) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        return target_path
    urlretrieve(url, target_path)
    return target_path


def fetch_crf_data(config_path: Path | None = None, force: bool = False) -> list[dict[str, str]]:
    config = load_crf_experiment_config(config_path)
    results: list[dict[str, str]] = []
    for target in build_download_targets(config):
        try:
            download_file(target.url, target.target_path, force=force)
            results.append({"name": target.name, "status": "downloaded", "path": str(target.target_path)})
        except (HTTPError, URLError, OSError, TimeoutError) as exc:
            status = "failed" if target.required else "skipped"
            results.append(
                {
                    "name": target.name,
                    "status": status,
                    "path": str(target.target_path),
                    "reason": str(exc),
                }
            )
            if target.required:
                raise
    return results


def fetch_crf_data_command(
    config: Path = typer.Option(default_crf_config_path(), help="Path to the CRF experiment config."),
    force: bool = typer.Option(False, help="Redownload files even if they already exist."),
) -> None:
    results = fetch_crf_data(config_path=config, force=force)
    for row in results:
        typer.echo(f"{row['name']}: {row['status']} -> {row['path']}")
