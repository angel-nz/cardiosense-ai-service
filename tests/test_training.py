"""Current training-contract checks. These tests never retrain or overwrite artifacts."""
import json
from pathlib import Path

import joblib
import pytest

from scripts import train

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "skorp-beta-0.2"
EXPECTED_FEATURES = [
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "glucose",
]


def load(name):
    return json.loads((ARTIFACT_DIR / name).read_text())


def test_official_training_entrypoint_is_beta02_exact_11_feature_contract():
    assert train.MODEL_VERSION == "Skorp-Beta-0.2"
    assert train.FEATURES == EXPECTED_FEATURES
    assert len(train.FEATURES) == 11
    assert "heartRate" not in train.FEATURES
    assert train.OUT == ARTIFACT_DIR


def test_beta02_artifacts_exist_without_training():
    required = (
        "classifier.joblib", "imputer.joblib", "anomaly_model.joblib",
        "feature_importance.json", "feature_importance.csv", "metrics.json",
        "metadata.json", "risk_thresholds.json", "dataset_audit.json",
        "split_indices.json", "test_predictions.csv", "coefficient_diagnostics.json",
    )
    for name in required:
        assert (ARTIFACT_DIR / name).exists(), f"missing approved artifact {name}"


def test_metadata_matches_current_training_contract():
    meta = load("metadata.json")
    assert meta["model_name"] == "Skorp"
    assert meta["model_version"] == "Skorp-Beta-0.2"
    assert meta["features"] == EXPECTED_FEATURES
    assert meta["target"] == "TenYearCHD"
    assert meta["dataset_hash_sha256"] == train.EXPECTED_HASH
    assert meta["dataset_shape_raw"] == [5905, 16]
    assert meta["training_age_range"] == {"min": 32, "max": 81}
    assert meta["eligible_age_range"] == {"min": 32, "max": 81}
    assert "heartRate" not in json.dumps({"features": meta["features"], "anomaly_configuration": meta["anomaly_configuration"]})


def test_classifier_and_anomaly_artifacts_use_exact_feature_order():
    classifier = joblib.load(ARTIFACT_DIR / "classifier.joblib")
    imputer = joblib.load(ARTIFACT_DIR / "imputer.joblib")
    anomaly = joblib.load(ARTIFACT_DIR / "anomaly_model.joblib")
    assert list(classifier.feature_names_in_) == EXPECTED_FEATURES
    assert list(imputer.feature_names_in_) == EXPECTED_FEATURES
    assert list(anomaly.feature_names_in_) == EXPECTED_FEATURES


def test_classifier_architecture_metadata_is_beta02_approved_family():
    meta = load("metadata.json")
    assert meta["selected_algorithm"] == "LogisticRegression+SMOTE"
    assert "sigmoid" in meta["calibration_method"]
    assert meta["smote_config"] == {
        "random_state": 42, "k_neighbors": 5, "applied_to": "train_only_per_cv_fold"
    }
    assert meta["anomaly_configuration"]["features"] == EXPECTED_FEATURES


def test_feature_importance_is_complete_11_feature_contract():
    fi = load("feature_importance.json")
    rows = fi["features"]
    assert [row["feature"] for row in rows] == EXPECTED_FEATURES
    assert len(rows) == 11
    assert fi["scoring"] == "average_precision"
    assert fi["n_repeats"] == 50
    assert fi["random_state"] == 42
    assert sum(row["importance_percent"] for row in rows) == pytest.approx(100.0)


def test_risk_thresholds_unchanged():
    assert load("risk_thresholds.json") == {"low_max": 0.2, "moderate_max": 0.35}


def test_split_contract_is_reproducible_metadata():
    meta = load("metadata.json")
    assert meta["random_state"] == 42
    assert meta["split_sizes"] == {"train": 4133, "validation": 886, "test": 886}
