"""
sepsis_vitals.ml.artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~
Versioned model artifacts: manifest, integrity, compatibility, and status.

A model directory is deployable only with a ``manifest.json`` that records:

* the SHA-256 of every artifact (checked *before* anything is unpickled,
  because ``joblib.load`` executes code from the file);
* the feature schema (names and hash) the model was trained on;
* the scikit-learn version used to serialise the model;
* a ``validation_status`` describing the evidence behind the model.

Validation statuses describe evidence, not clinical approval. No status in
this module makes a model clinically usable: ``CLINICALLY_APPROVED_STATUSES``
is deliberately empty until an approved validation process defines one.

Command line::

    python -m sepsis_vitals.ml.artifacts build models/ --validation-status synthetic-development
    python -m sepsis_vitals.ml.artifacts verify models/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

#: Evidence levels a manifest may declare. None of them permits clinical use.
VALIDATION_STATUSES = ("synthetic-development", "retrospective-research", "unvalidated")
#: Statuses that would permit clinical use. Intentionally empty.
CLINICALLY_APPROVED_STATUSES: frozenset = frozenset()

_ARTIFACT_FILES = (
    "sepsis_model.joblib",
    "model_metadata.json",
    "imputation_medians.json",
    "scaler.joblib",
    "conformal_predictor.joblib",
)


class ModelArtifactError(RuntimeError):
    """Raised when artifacts are missing, tampered with, or incompatible."""

    def __init__(self, state: str, reason: str) -> None:
        super().__init__(reason)
        self.state = state
        self.reason = reason


@dataclass
class ArtifactStatus:
    """What the API reports about prediction readiness."""

    state: str  # ready | absent | invalid | incompatible | unverified
    reason: str = ""
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    validation_status: Optional[str] = None
    training_data: Optional[str] = None
    artifact_sha256: Optional[str] = None
    feature_schema_sha256: Optional[str] = None
    pipeline_version: Optional[str] = None
    known_issues: List[str] = field(default_factory=list)

    @property
    def clinically_ready(self) -> bool:
        return self.state == "ready" and self.validation_status in CLINICALLY_APPROVED_STATUSES

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["prediction_ready"] = self.state in ("ready", "unverified")
        out["clinically_ready"] = self.clinically_ready
        out["clinical_use"] = "permitted" if self.clinically_ready else "not-permitted"
        return out

    def provenance(self) -> Dict[str, Any]:
        """Fields recorded with every prediction."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "validation_status": self.validation_status,
            "artifact_sha256": self.artifact_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "pipeline_version": self.pipeline_version,
            "artifact_state": self.state,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_schema_hash(names: List[str]) -> str:
    return hashlib.sha256(json.dumps(list(names)).encode()).hexdigest()


def _sklearn_version() -> Optional[str]:
    try:
        import sklearn

        return str(sklearn.__version__)
    except ImportError:
        return None


def build_manifest(
    model_dir: Path,
    validation_status: str,
    training_data: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Describe the artifacts currently in *model_dir*."""
    if validation_status not in VALIDATION_STATUSES:
        raise ValueError(f"validation_status must be one of {VALIDATION_STATUSES}")
    metadata = json.loads((model_dir / "model_metadata.json").read_text())
    names = list(metadata["feature_names"])
    artifacts = {
        name: {"sha256": sha256_file(model_dir / name), "bytes": (model_dir / name).stat().st_size}
        for name in _ARTIFACT_FILES
        if (model_dir / name).exists()
    }
    provenance = metadata.get("data_provenance", {})
    return {
        "manifest_version": MANIFEST_VERSION,
        "model_id": model_id or f"{metadata.get('model_name', 'model')}-{metadata.get('version', '0')}",
        "model_version": metadata.get("version"),
        "pipeline_version": metadata.get("pipeline_version", metadata.get("version")),
        "artifacts": artifacts,
        "feature_schema": {"names": names, "sha256": feature_schema_hash(names)},
        "runtime": {"scikit_learn": _sklearn_version()},
        "validation_status": validation_status,
        "training_data": training_data or provenance.get("source", "unknown"),
        "known_issues": metadata.get("known_issues", []),
    }


def write_manifest(model_dir: Path, validation_status: str, **kwargs: Any) -> Path:
    manifest = build_manifest(model_dir, validation_status, **kwargs)
    path = model_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def verify_artifacts(model_dir: Path, allow_unverified: bool = False) -> ArtifactStatus:
    """Check that *model_dir* holds an intact, compatible model. Raises on failure.

    Never unpickles anything. Returns the status to report when loading may
    proceed.
    """
    model_path = model_dir / "sepsis_model.joblib"
    if not model_path.exists():
        raise ModelArtifactError("absent", f"No model artifact in {model_dir}")

    manifest_path = model_dir / MANIFEST_NAME
    if not manifest_path.exists():
        if allow_unverified:
            return ArtifactStatus(
                state="unverified",
                reason="No manifest; integrity and compatibility were not checked "
                "(SEPSIS_ALLOW_UNVERIFIED_MODEL=true)",
                validation_status="unvalidated",
            )
        raise ModelArtifactError(
            "invalid", "No manifest.json: refusing to load an unverified pickle"
        )

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ModelArtifactError("invalid", f"Unreadable manifest: {exc.msg}") from exc
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ModelArtifactError("incompatible", "Unsupported manifest version")

    for name, expected in manifest.get("artifacts", {}).items():
        path = model_dir / name
        if not path.exists():
            raise ModelArtifactError("invalid", f"Artifact listed in manifest is missing: {name}")
        if sha256_file(path) != expected.get("sha256"):
            raise ModelArtifactError("invalid", f"Checksum mismatch for {name}")
    if "sepsis_model.joblib" not in manifest.get("artifacts", {}):
        raise ModelArtifactError("invalid", "Manifest does not cover sepsis_model.joblib")

    metadata = json.loads((model_dir / "model_metadata.json").read_text())
    names = list(metadata.get("feature_names", []))
    schema = manifest.get("feature_schema", {})
    if feature_schema_hash(names) != schema.get("sha256"):
        raise ModelArtifactError("incompatible", "Feature schema differs from the manifest")

    trained_with = manifest.get("runtime", {}).get("scikit_learn")
    running = _sklearn_version()
    if trained_with and running and trained_with != running:
        raise ModelArtifactError(
            "incompatible",
            f"Model serialised with scikit-learn {trained_with}; runtime has {running}",
        )

    status = manifest.get("validation_status")
    if status not in VALIDATION_STATUSES:
        raise ModelArtifactError("invalid", "Manifest has no recognised validation_status")

    return ArtifactStatus(
        state="ready",
        model_id=manifest.get("model_id"),
        model_version=manifest.get("model_version"),
        validation_status=status,
        training_data=manifest.get("training_data"),
        artifact_sha256=manifest["artifacts"]["sepsis_model.joblib"]["sha256"],
        feature_schema_sha256=schema.get("sha256"),
        pipeline_version=manifest.get("pipeline_version"),
        known_issues=list(manifest.get("known_issues", [])),
    )


def check_feature_compatibility(feature_names: List[str]) -> None:
    """The current feature pipeline must produce every feature the model needs."""
    import numpy as np
    import pandas as pd

    from sepsis_vitals.ml.trainer import prepare_features

    probe = pd.DataFrame([{
        "patient_id": "_", "timestamp": pd.Timestamp("2026-01-01"), "age_years": 50,
        "temperature": 37.0, "heart_rate": 80, "resp_rate": 16, "sbp": 120, "dbp": 80,
        "spo2": 97, "gcs": 15, "map": 93, "lactate": np.nan, "wbc": np.nan,
        "procalcitonin": np.nan, "has_hypertension": 0, "has_diabetes": 0, "has_ckd": 0,
        "has_copd": 0, "has_heart_failure": 0,
    }])
    features, _ = prepare_features(probe)
    missing = [name for name in feature_names if name not in features.columns]
    if missing:
        raise ModelArtifactError(
            "incompatible", f"Feature pipeline does not produce {len(missing)} model feature(s)"
        )


def allow_unverified_from_env() -> bool:
    return os.getenv("SEPSIS_ALLOW_UNVERIFIED_MODEL", "false").lower() == "true"


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="write manifest.json for a model directory")
    build.add_argument("model_dir", type=Path)
    build.add_argument("--validation-status", required=True, choices=VALIDATION_STATUSES)
    build.add_argument("--training-data")
    verify = sub.add_parser("verify", help="check a model directory against its manifest")
    verify.add_argument("model_dir", type=Path)
    opts = parser.parse_args(argv)

    if opts.cmd == "build":
        path = write_manifest(opts.model_dir, opts.validation_status, training_data=opts.training_data)
        print(f"wrote {path}")
        return 0
    try:
        status = verify_artifacts(opts.model_dir)
        check_feature_compatibility(
            json.loads((opts.model_dir / "model_metadata.json").read_text())["feature_names"]
        )
    except ModelArtifactError as exc:
        print(f"{exc.state}: {exc.reason}", file=sys.stderr)
        return 1
    print(json.dumps(status.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
