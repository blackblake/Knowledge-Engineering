from __future__ import annotations

import typer

from kg_project.gazetteer import build_gazetteer_command
from kg_project.labeling import sample_gold_command, weak_label_command
from kg_project.ner.experiments import compare_ner_command

app = typer.Typer(help="Career-skill knowledge graph pipeline.")

app.command("build-gazetteer")(build_gazetteer_command)
app.command("weak-label")(weak_label_command)
app.command("sample-gold")(sample_gold_command)
app.command("compare-ner")(compare_ner_command)
