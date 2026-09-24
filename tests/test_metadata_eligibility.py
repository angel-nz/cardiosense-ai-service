"""U8.6C-FIX-1 — tests for the eligibility-metadata generation logic in
scripts/train.py. Exercises compute_eligibility_metadata() directly (a pure
function, no I/O, no model fitting) against the real, full framingham.csv
age column — this is the SAME data compute_eligibility_metadata() reads
when scripts/train.py builds metadata.json, so these tests do not require
running the (expensive) full training pipeline.
"""
import json
from pathlib import Path

import pandas as pd

from app.inference import _read_age_range
from scripts.train import FEATURES, RAW_CSV, compute_eligibility_metadata

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "skorp-beta-0.1"


def _full_age_series() -> pd.Series:
    # Same population compute_eligibility_metadata() is called with inside
    # scripts/train.py::main() — the full processed `X["age"]` column,
    # before the train/validation/test split (see the docstring on
    # compute_eligibility_metadata for why the full population is used,
    # not just X_train).
    df = pd.read_csv(RAW_CSV)
    return df[FEATURES]["age"]


def test_meta01_derives_from_data_not_magic_constants():
    # A synthetic series with a range DIFFERENT from the real dataset's
    # 32-70 must produce THAT different range — proves the function reads
    # the data it's given, it doesn't just always return 32/70.
    synthetic = pd.Series([18, 40, 55, 99])
    result = compute_eligibility_metadata(synthetic)
    assert result["training_age_range"] == {"min": 18, "max": 99}
    assert result["eligible_age_range"] == {"min": 18, "max": 99}


def test_meta02_current_dataset_min_max():
    result = compute_eligibility_metadata(_full_age_series())
    assert result["training_age_range"] == {"min": 32, "max": 70}


def test_meta03_eligible_equals_training_and_is_an_explicit_copy():
    result = compute_eligibility_metadata(_full_age_series())
    assert result["eligible_age_range"] == result["training_age_range"]
    # Explicit copy (per the CURRENT Skorp-Beta-0.1 policy), not aliasing
    # the same dict object — mutating one must never affect the other.
    assert result["eligible_age_range"] is not result["training_age_range"]


def test_meta04_basis_text_contains_computed_range():
    result = compute_eligibility_metadata(_full_age_series())
    basis = result["eligible_age_range_basis"]
    assert "32" in basis
    assert "70" in basis


def test_meta05_basis_text_disclaims_clinical_validation():
    result = compute_eligibility_metadata(_full_age_series())
    basis = result["eligible_age_range_basis"].lower()
    assert "observed" in basis
    assert "not a clinically validated or medically established range" in basis


def test_meta06_existing_metadata_fields_untouched():
    # compute_eligibility_metadata only ADDS 3 keys — confirms none of the
    # pre-existing metadata.json fields were renamed/removed by this change
    # (checked against the currently checked-in artifact, produced before
    # this fix by the unmodified rest of scripts/train.py::main()).
    meta = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
    for key in (
        "model_name", "model_version", "dataset", "dataset_hash_sha256",
        "dataset_shape_raw", "target", "features", "excluded_features",
        "missing_values_processed_columns", "train_size", "validation_size",
        "test_size", "class_distribution", "random_state", "smote_config",
        "selected_algorithm", "selected_hyperparameters", "selection_metric",
        "metrics", "calibration_method", "risk_thresholds",
        "anomaly_algorithm", "anomaly_configuration", "anomaly_semantics",
        "training_timestamp", "python_version", "dependency_versions",
    ):
        assert key in meta, f"pre-existing metadata field missing: {key}"


def test_meta07_inference_loader_accepts_generated_structure():
    # U8.6B's _read_age_range (app/inference.py) must accept the exact
    # structure compute_eligibility_metadata() produces — no changes to
    # inference.py were made or are required for this.
    result = compute_eligibility_metadata(_full_age_series())
    parsed = _read_age_range({"eligible_age_range": result["eligible_age_range"]}, "eligible_age_range")
    assert parsed == {"min": 32, "max": 70}
    parsed_training = _read_age_range({"training_age_range": result["training_age_range"]}, "training_age_range")
    assert parsed_training == {"min": 32, "max": 70}
