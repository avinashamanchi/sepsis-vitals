"""
tests/test_provenance.py — every generated report records code, environment,
backends, seeds, dataset fingerprint, candidates and runtime.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")


def _frame(shift: float = 0.0):
    return pd.DataFrame({"patient_id": ["a", "a", "b"], "heart_rate": [80.0 + shift, 95.0, 110.0]})


def test_dataset_fingerprint_is_deterministic_and_sensitive():
    from sepsis_vitals.ml.provenance import dataset_fingerprint

    first = dataset_fingerprint([_frame(), _frame()])
    assert first == dataset_fingerprint([_frame(), _frame()])
    assert first["shapes"] == [[3, 2], [3, 2]]
    assert dataset_fingerprint([_frame(0.1), _frame()])["sha256"] != first["sha256"]
    # Order of the splits matters (train/val/test are different roles)
    assert dataset_fingerprint([_frame(0.1), _frame()])["sha256"] != \
        dataset_fingerprint([_frame(), _frame(0.1)])["sha256"]


def test_run_provenance_records_everything_a_report_needs():
    from sepsis_vitals.ml.provenance import markdown, run_provenance

    record = run_provenance(
        backend_status={"LightGBM": "lightgbm native runtime missing", "XGBoost": "available"},
        seeds={"data": 42}, dataset={"description": "synthetic;", "sha256": "ab" * 32, "shapes": [[3, 2]]},
        candidates=["GradientBoosting"], runtime_seconds=12.34, command="python scripts/x.py",
    )
    assert set(record) >= {"generated_at", "code", "environment", "optional_backends", "seeds",
                           "dataset", "candidate_models", "runtime_seconds", "command"}
    assert record["environment"]["python"]
    assert record["environment"]["dependency_lock_sha256"] is None or \
        len(record["environment"]["dependency_lock_sha256"]) == 64
    assert "scikit-learn" in record["environment"]["packages"]
    assert record["runtime_seconds"] == 12.3
    text = "\n".join(markdown(record))
    for expected in ("## Provenance", "LightGBM: lightgbm native runtime missing", "Seeds: data 42",
                     "GradientBoosting", "runtime 12.3 s", "python scripts/x.py"):
        assert expected in text, expected


def test_ablation_markdown_includes_provenance():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ablation_script", root / "scripts" / "run_no_labs_ablation.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    from sepsis_vitals.ml.provenance import run_provenance

    report = {
        "comparison": {arm: {"n_features": 1, "best_model": "GradientBoosting", "held_out_test_auroc": 0.9,
                             "held_out_test_auprc": 0.5} for arm in ("full", "no_labs")},
        "auroc_delta_no_labs_minus_full": -0.1,
        "data_provenance": {"n_patients": 10, "seed": 1},
        "method": "m.", "limitations": ["synthetic"],
        "candidate_models": {"full": ["GradientBoosting"], "no_labs": ["GradientBoosting"]},
        "provenance": run_provenance(backend_status={"LightGBM": "not installed"}),
    }
    text = script._markdown_report(report)
    assert "## Provenance" in text and "Not available on this host: LightGBM (see Provenance)" in text
