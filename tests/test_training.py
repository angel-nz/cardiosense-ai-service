"""Training pipeline sanity checks — validates artifacts already produced by
scripts/train.py (does not re-run training; that's an offline, separate step)."""
import json
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "skorp-beta-0.1"


def load(name):
    with open(ARTIFACT_DIR / name) as f:
        return json.load(f)


def test_artifacts_exist():
    for f in ("classifier.joblib", "imputer.joblib", "anomaly_model.joblib",
              "feature_importance.json", "metrics.json", "metadata.json",
              "risk_thresholds.json"):
        assert (ARTIFACT_DIR / f).exists(), f"missing artifact {f}"


def test_metadata_columns_and_target():
    meta = load("metadata.json")
    assert meta["target"] == "TenYearCHD"
    assert set(meta["features"]) == {
        "male", "age", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes",
        "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
    }
    assert "TenYearCHD" not in meta["features"]


def test_excluded_features():
    meta = load("metadata.json")
    assert set(meta["excluded_features"]) == {"education", "prevalentStroke", "prevalentHyp"}
    assert not set(meta["excluded_features"]) & set(meta["features"])


def test_split_reproducible_sizes():
    meta = load("metadata.json")
    total = meta["train_size"] + meta["validation_size"] + meta["test_size"]
    assert meta["random_state"] == 42
    # ~70/15/15 split
    assert abs(meta["train_size"] / total - 0.70) < 0.02
    assert abs(meta["validation_size"] / total - 0.15) < 0.02
    assert abs(meta["test_size"] / total - 0.15) < 0.02


def test_smote_config_train_only():
    meta = load("metadata.json")
    assert meta["smote_config"]["applied_to"] == "train_only_per_cv_fold"


def test_model_selection_recorded():
    metrics = load("metrics.json")
    assert metrics["selected_model"] in [c["name"] for c in metrics["candidates"]]
    assert "pr_auc" in metrics["test_metrics"]


def test_calibration_recorded():
    meta = load("metadata.json")
    assert "CalibratedClassifierCV" in meta["calibration_method"]


def test_feature_importance_top6_sums_100_or_na():
    fi = load("feature_importance.json")
    if fi["note"] is None:
        assert len(fi["top_6"]) == 6
        assert abs(sum(fi["top_6"].values()) - 100.0) < 0.05
        vals = list(fi["top_6"].values())
        assert vals == sorted(vals, reverse=True)
    else:
        assert fi["top_6"] == {} or all(v is None for v in fi["top_6"].values())


def test_model_version_string():
    meta = load("metadata.json")
    assert meta["model_version"] == "Skorp-Beta-0.1"


def test_dataset_hash_present():
    meta = load("metadata.json")
    assert len(meta["dataset_hash_sha256"]) == 64
