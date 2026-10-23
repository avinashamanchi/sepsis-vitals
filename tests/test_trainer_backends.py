"""
tests/test_trainer_backends.py — optional model backends: genuinely
unavailable (skipped, reason recorded) versus broken (error) versus usable.

Uses disposable fake packages under tmp_path, never the real LightGBM or
XGBoost, so the outcome does not depend on the host.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("sklearn")


@pytest.fixture()
def fake_pkg(tmp_path, monkeypatch):
    """Create a throwaway importable package; returns its unique name."""
    monkeypatch.syspath_prepend(str(tmp_path))

    def make(init_source: str | None) -> str:
        name = f"fakebackend_{uuid.uuid4().hex[:8]}"
        pkg = tmp_path / name
        pkg.mkdir()
        if init_source is not None:
            (pkg / "__init__.py").write_text(init_source)
        return name

    return make


def test_missing_package_is_unavailable():
    from sepsis_vitals.ml.trainer import load_optional_backend

    estimator, reason = load_optional_backend(f"not_installed_{uuid.uuid4().hex[:8]}", "Classifier")
    assert estimator is None and "not installed" in reason


@pytest.mark.parametrize("raise_line", [
    "raise OSError('dlopen: Library not loaded: @rpath/libomp.dylib')",                       # LightGBM style
    "class XGBoostError(ValueError): pass\nraise XGBoostError('Library (libx.dylib) could not be loaded')",  # XGBoost style
])
def test_missing_native_runtime_is_unavailable(fake_pkg, raise_line):
    from sepsis_vitals.ml.trainer import load_optional_backend

    estimator, reason = load_optional_backend(fake_pkg(raise_line), "Classifier")
    assert estimator is None and "native runtime" in reason


def test_leftover_directory_without_init_is_broken(fake_pkg):
    """The stage-4 failure: a half-removed package imported as an empty namespace."""
    from sepsis_vitals.ml.trainer import BrokenBackendError, load_optional_backend

    with pytest.raises(BrokenBackendError, match="without __init__"):
        load_optional_backend(fake_pkg(None), "Classifier")


def test_failing_internal_import_is_broken(fake_pkg):
    from sepsis_vitals.ml.trainer import BrokenBackendError, load_optional_backend

    with pytest.raises(BrokenBackendError, match="failed to import"):
        load_optional_backend(fake_pkg("import a_module_that_is_not_there_xyz"), "Classifier")


def test_missing_estimator_class_is_broken(fake_pkg):
    from sepsis_vitals.ml.trainer import BrokenBackendError, load_optional_backend

    with pytest.raises(BrokenBackendError, match="has no Classifier"):
        load_optional_backend(fake_pkg("OTHER = 1"), "Classifier")


def test_unexpected_errors_are_not_treated_as_unavailable(fake_pkg):
    from sepsis_vitals.ml.trainer import load_optional_backend

    with pytest.raises(RuntimeError, match="boom"):
        load_optional_backend(fake_pkg("raise RuntimeError('boom')"), "Classifier")


def test_a_usable_backend_returns_its_estimator(fake_pkg):
    from sepsis_vitals.ml.trainer import load_optional_backend

    name = fake_pkg("class Classifier:\n    pass\n")
    estimator, reason = load_optional_backend(name, "Classifier")
    assert reason is None and estimator.__name__ == "Classifier"


def test_configs_record_availability_and_strict_mode_refuses_gaps(monkeypatch):
    from sepsis_vitals.ml import trainer

    def fake_loader(module_name, attribute):
        return (None, f"{module_name} is not installed") if module_name == "lightgbm" else (object, None)

    monkeypatch.setattr(trainer, "load_optional_backend", fake_loader)
    monkeypatch.delenv("SEPSIS_REQUIRE_ALL_BACKENDS", raising=False)
    configs = trainer._get_model_configs()
    assert "LightGBM" not in configs and "XGBoost" in configs
    assert trainer.BACKEND_STATUS == {"LightGBM": "lightgbm is not installed", "XGBoost": "available"}

    monkeypatch.setenv("SEPSIS_REQUIRE_ALL_BACKENDS", "1")
    with pytest.raises(trainer.BrokenBackendError, match="LightGBM is required"):
        trainer._get_model_configs()


def test_inference_scales_features_exactly_once(tmp_path):
    """Training, evaluation (N38) and inference must agree: one scaling pass."""
    import shutil
    from pathlib import Path

    from sepsis_vitals.ml.predictor import SepsisPredictor

    root = Path(__file__).resolve().parents[1] / "models"
    if not (root / "manifest.json").exists():
        pytest.skip("committed model artifacts unavailable")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in ("sepsis_model.joblib", "model_metadata.json", "imputation_medians.json", "manifest.json"):
        shutil.copy(root / name, model_dir / name)

    calls = []

    class CountingScaler:
        def transform(self, X):
            calls.append(X.shape)
            return X  # identity: the committed model was trained unscaled

    predictor = SepsisPredictor(model_dir=str(model_dir), state_dir=str(tmp_path / "state"))
    predictor.load()
    predictor.scaler = CountingScaler()
    predictor.predict({"heart_rate": 118, "resp_rate": 24, "sbp": 96, "temperature": 38.6}, patient_id="scale-check")
    assert len(calls) == 1
