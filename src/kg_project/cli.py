from __future__ import annotations

import typer

from kg_project.api import run_api_command
from kg_project.data_sources import fetch_crf_data_command
from kg_project.downstream import growth_path_command, recommend_command, skill_gap_command
from kg_project.evaluation import evaluate_graph_command, evaluate_recommendations_command
from kg_project.final_demo import run_final_demo_command
from kg_project.full_input import build_full_input_command
from kg_project.gazetteer import build_gazetteer_command
from kg_project.graph_pipeline import build_graph_command
from kg_project.labeling import sample_gold_command, weak_label_command
from kg_project.llm_extraction import llm_extract_command, uie_extract_command
from kg_project.ner.neural_training import run_neural_experiments_command
from kg_project.relation_models import prepare_re_dataset_command, train_bert_re_command
from kg_project.re_experiments import run_re_comparison_command
from kg_project.visualization import visualize_graph_command
from kg_project.ner.experiments import compare_ner_command, run_formal_crf_experiment_command
from kg_project.preprocess import prepare_crf_data_command

app = typer.Typer(help="Career-skill knowledge graph pipeline.")

app.command("build-gazetteer")(build_gazetteer_command)
app.command("build-graph")(build_graph_command)
app.command("recommend-roles")(recommend_command)
app.command("skill-gap")(skill_gap_command)
app.command("growth-path")(growth_path_command)
app.command("evaluate-graph")(evaluate_graph_command)
app.command("evaluate-recommendations")(evaluate_recommendations_command)
app.command("prepare-re-dataset")(prepare_re_dataset_command)
app.command("train-bert-re")(train_bert_re_command)
app.command("run-re-comparison")(run_re_comparison_command)
app.command("visualize-graph")(visualize_graph_command)
app.command("run-api")(run_api_command)
app.command("run-final-demo")(run_final_demo_command)
app.command("build-full-input")(build_full_input_command)
app.command("uie-extract")(uie_extract_command)
app.command("llm-extract")(llm_extract_command)
app.command("run-neural-experiments")(run_neural_experiments_command)
app.command("fetch-crf-data")(fetch_crf_data_command)
app.command("prepare-crf-data")(prepare_crf_data_command)
app.command("run-crf-experiment")(run_formal_crf_experiment_command)
app.command("weak-label")(weak_label_command)
app.command("sample-gold")(sample_gold_command)
app.command("compare-ner")(compare_ner_command)


if __name__ == "__main__":
    app()
