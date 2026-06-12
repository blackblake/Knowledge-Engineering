import importlib


def test_experiments_module_exposes_formal_crf_entrypoint() -> None:
    experiments = importlib.import_module("kg_project.ner.experiments")
    assert hasattr(experiments, "run_formal_crf_experiment")
