from __future__ import annotations

from pathlib import Path

import typer

from kg_project.api import create_app
from kg_project.full_input import build_full_weak_labels
from kg_project.graph_pipeline import build_graph_from_ner


def run_final_demo_command(
    ner: Path = typer.Option(Path("data/processed/full_weak_ner.jsonl"), help="NER jsonl input."),
    gazetteer: Path = typer.Option(Path("data/processed/full_gazetteer.jsonl"), help="Gazetteer jsonl."),
    jobs: Path = typer.Option(Path("data/processed/crf/jd_sample.jsonl"), help="Processed JD sample JSONL."),
    base_gazetteer: Path = typer.Option(
        Path("data/processed/crf/formal_gazetteer.jsonl"),
        help="Base formal gazetteer JSONL.",
    ),
    nodes: Path = typer.Option(Path("data/reports/graph_nodes_final.csv"), help="Output nodes csv."),
    edges: Path = typer.Option(Path("data/reports/graph_edges_final.csv"), help="Output edges csv."),
    relations: Path = typer.Option(Path("data/reports/graph_relations_final.jsonl"), help="Output relations jsonl."),
    report: Path = typer.Option(Path("data/reports/graph_report_final.json"), help="Output report json."),
    rebuild_input: bool = typer.Option(True, help="Rebuild full weak NER and gazetteer before graph construction."),
    max_jobs: int = typer.Option(10000, help="Maximum job postings to use when rebuilding full input."),
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
) -> None:
    if rebuild_input or not ner.exists() or not gazetteer.exists():
        build_full_weak_labels(
            jobs_path=jobs,
            base_gazetteer_path=base_gazetteer,
            output_ner_path=ner,
            output_gazetteer_path=gazetteer,
            max_jobs=max_jobs,
        )
    build_graph_from_ner(
        ner_path=ner,
        gazetteer_path=gazetteer,
        nodes_path=nodes,
        edges_path=edges,
        relations_path=relations,
        report_path=report,
    )
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required. install with pip install uvicorn") from exc

    app = create_app(nodes_path=nodes, edges_path=edges, use_neo4j=False)
    uvicorn.run(app, host=host, port=port)
