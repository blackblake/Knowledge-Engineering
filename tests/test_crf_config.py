from pathlib import Path

import yaml


def test_crf_experiment_config_exists_and_has_two_variants() -> None:
    config_path = Path("config/crf_experiment.yaml")
    assert config_path.exists()

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert sorted(payload["experiments"].keys()) == ["crf_base", "crf_gazetteer"]
