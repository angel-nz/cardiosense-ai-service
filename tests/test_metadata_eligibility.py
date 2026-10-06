"""Eligibility tests for the official Skorp-Beta-0.2 training contract."""
import json
from pathlib import Path

import pandas as pd
import pytest

from app.inference import _read_age_range
from scripts.train import FEATURES, RAW_CSV, compute_eligibility_metadata

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "skorp-beta-0.2"


def _full_age_series() -> pd.Series:
    df = pd.read_csv(RAW_CSV)
    return df[FEATURES]["age"]


def _is_model_eligible(age: int, age_range: dict) -> bool:
    return age_range["min"] <= age <= age_range["max"]


def test_meta01_derives_from_data_not_magic_constants():
    synthetic = pd.Series([18, 40, 55, 99])
    result = compute_eligibility_metadata(synthetic)
    assert result["training_age_range"] == {"min": 18, "max": 99}
    assert result["eligible_age_range"] == {"min": 18, "max": 99}


def test_meta02_current_dataset_support_is_32_81():
    result = compute_eligibility_metadata(_full_age_series())
    assert result["training_age_range"] == {"min": 32, "max": 81}
    assert result["eligible_age_range"] == {"min": 32, "max": 81}


def test_meta03_eligible_equals_training_and_is_an_explicit_copy():
    result = compute_eligibility_metadata(_full_age_series())
    assert result["eligible_age_range"] == result["training_age_range"]
    assert result["eligible_age_range"] is not result["training_age_range"]


def test_meta04_basis_is_model_support_not_clinical_range():
    basis = compute_eligibility_metadata(_full_age_series())["eligible_age_range_basis"].lower()
    assert "observed" in basis
    assert "not clinical validation" in basis


def test_meta05_current_artifact_matches_generated_eligibility_structure():
    meta = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
    generated = compute_eligibility_metadata(_full_age_series())
    for key in ("training_age_range", "eligible_age_range", "eligible_age_range_basis"):
        assert meta[key] == generated[key]
    assert meta["model_version"] == "Skorp-Beta-0.2"
    assert meta["features"] == FEATURES


def test_meta06_inference_loader_accepts_current_structure():
    meta = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
    assert _read_age_range(meta, "eligible_age_range") == {"min": 32, "max": 81}
    assert _read_age_range(meta, "training_age_range") == {"min": 32, "max": 81}


@pytest.mark.parametrize(
    ("age", "eligible"),
    [(31, False), (32, True), (70, True), (71, True), (81, True), (82, False)],
)
def test_meta07_exact_model_eligibility_boundaries(age, eligible):
    meta = json.loads((ARTIFACT_DIR / "metadata.json").read_text())
    age_range = _read_age_range(meta, "eligible_age_range")
    assert age_range is not None
    assert _is_model_eligible(age, age_range) is eligible
