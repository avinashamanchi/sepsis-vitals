"""
tests/test_retrain_cli.py — replacing the committed model cannot happen by
accident, and the generator options used are recorded.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def model_dir(tmp_path):
    dest = tmp_path / "models"
    dest.mkdir()
    for name in ("sepsis_model.joblib", "model_metadata.json", "imputation_medians.json", "manifest.json"):
        shutil.copy(ROOT / "models" / name, dest / name)
    return dest


def test_retraining_into_a_model_directory_needs_replace(model_dir):
    import retrain

    with pytest.raises(SystemExit):
        retrain.main(["--output", str(model_dir), "--patients", "10"])
    # nothing was archived or written
    assert not (model_dir / "archive").exists()


def test_archive_keeps_a_verifiable_copy_of_the_current_model(model_dir):
    import retrain
    from sepsis_vitals.ml.artifacts import verify_artifacts

    archive = retrain.archive_existing_model(model_dir)
    assert archive.parent == model_dir / "archive"
    status = verify_artifacts(archive)
    assert status.state == "ready" and status.validation_status == "synthetic-development"
    assert retrain.archive_existing_model(model_dir) == archive  # idempotent


def test_horizon_label_mode_needs_an_explicit_horizon(tmp_path):
    import retrain

    with pytest.raises(SystemExit):
        retrain.main(["--output", str(tmp_path / "new"), "--label-mode", "onset_within_horizon"])


def test_generator_options_are_recorded_in_provenance():
    import retrain
    from sepsis_vitals.ml.profile_evaluation import PROFILES

    _, _, _, provenance = retrain.load_synthetic_data(n_patients=40, prevalence=0.2, seed=3,
                                                      **PROFILES["decoupled"])
    assert provenance["generator_options"] == PROFILES["decoupled"]
    _, _, _, legacy = retrain.load_synthetic_data(n_patients=40, prevalence=0.2, seed=3)
    assert legacy["generator_options"] == {}
