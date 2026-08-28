"""
Skorp-Beta-0.1 — offline training pipeline for CardioSense.

Run:
    python3 scripts/train.py

Produces (under artifacts/skorp-beta-0.1/):
    classifier.joblib       — calibrated risk classifier (full pipeline)
    imputer.joblib          — mean imputer fit on TRAIN only
    anomaly_model.joblib    — IsolationForest fit on TRAIN normals (y==0)
    feature_importance.json — permutation importance, top 6, sums to 100.00
    metrics.json            — full metrics (CV, validation, test)
    metadata.json           — model/dataset/training metadata
    risk_thresholds.json    — LOW/MODERATE/HIGH thresholds
    REPORT.md               — human-readable training report

This script is OFFLINE ONLY. The AI Service never re-trains at startup —
it only loads these artifacts (see app/inference.py).
"""
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import imblearn
import joblib
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    brier_score_loss,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ─── Constants (Skorp-Beta-0.1 configuration — do not tune post-hoc) ────────
MODEL_NAME = "Skorp"
MODEL_VERSION = "Skorp-Beta-0.1"
RANDOM_STATE = 42

FEATURES = [
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]
EXCLUDED_FEATURES = ["education", "prevalentStroke", "prevalentHyp"]
TARGET = "TenYearCHD"

RISK_THRESHOLDS = {"low_max": 0.20, "moderate_max": 0.35}  # LOW<0.20 MOD<0.35 HIGH>=0.35

ANOMALY_CONFIG = {"n_estimators": 200, "contamination": 0.05, "random_state": RANDOM_STATE}

RAW_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "framingham.csv"
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "skorp-beta-0.1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def dataset_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def risk_level(score: float) -> str:
    if score < RISK_THRESHOLDS["low_max"]:
        return "low"
    if score < RISK_THRESHOLDS["moderate_max"]:
        return "moderate"
    return "high"


def evaluate(y_true, proba, threshold=0.35):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "threshold_used": threshold,
    }


def select_best(rows):
    """Rule 10: max PR-AUC; tie (<=0.005) -> max ROC-AUC; tie -> max recall;
    tie -> min Brier; tie -> simpler/more stable model."""
    def key(r):
        return (round(r["metrics"]["pr_auc"], 3), round(r["metrics"]["roc_auc"], 3),
                round(r["metrics"]["recall"], 3), -round(r["metrics"]["brier_score"], 4))
    return sorted(rows, key=key, reverse=True)[0]


def main():
    print(f"=== Training {MODEL_VERSION} ===")
    if not RAW_CSV.exists():
        print(f"FATAL: dataset not found at {RAW_CSV}")
        sys.exit(1)

    raw_hash = dataset_hash(RAW_CSV)
    df_raw = pd.read_csv(RAW_CSV)
    print(f"Raw dataset: {df_raw.shape}, sha256={raw_hash}")

    # ─── 3. Minimal processed dataset — only required columns ──────────────
    missing_required = [c for c in FEATURES + [TARGET] if c not in df_raw.columns]
    if missing_required:
        print(f"FATAL: dataset missing required columns: {missing_required}")
        sys.exit(1)

    df = df_raw[FEATURES + [TARGET]].copy()
    missing_report = df[FEATURES].isna().sum().to_dict()
    print("Missing values (processed columns):", missing_report)

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # ─── 7. Split 70/15/15 stratified, random_state=42 ──────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE,
    )
    sizes = {"train": len(X_train), "validation": len(X_val), "test": len(X_test)}
    print("Split sizes:", sizes)

    class_dist = {
        "train": y_train.value_counts().to_dict(),
        "validation": y_val.value_counts().to_dict(),
        "test": y_test.value_counts().to_dict(),
    }

    # ─── 6. Imputation — mean, fit on TRAIN ONLY ────────────────────────────
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES, index=X_train.index)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=FEATURES, index=X_val.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=FEATURES, index=X_test.index)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ─── 9/10/11. Candidate models ──────────────────────────────────────────
    candidates = []

    # Candidate A: RandomForest + SMOTE (grid search, SMOTE inside CV folds only)
    rf_smote_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    rf_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    }
    print("Fitting RandomForest + SMOTE grid search...")
    gs_rf_smote = GridSearchCV(rf_smote_pipe, rf_grid, scoring="average_precision", cv=cv, n_jobs=-1)
    gs_rf_smote.fit(X_train_imp, y_train)
    val_proba = gs_rf_smote.predict_proba(X_val_imp)[:, 1]
    candidates.append({
        "name": "RandomForest+SMOTE",
        "pipeline": gs_rf_smote.best_estimator_,
        "hyperparameters": gs_rf_smote.best_params_,
        "cv_pr_auc": float(gs_rf_smote.best_score_),
        "metrics": evaluate(y_val, val_proba),
        "uses_smote": True,
    })

    # Candidate B: RandomForest + class_weight=balanced, NO SMOTE (independent experiment)
    rf_bal_pipe = SkPipeline([
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")),
    ])
    rf_bal_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    }
    print("Fitting RandomForest + class_weight=balanced grid search...")
    gs_rf_bal = GridSearchCV(rf_bal_pipe, rf_bal_grid, scoring="average_precision", cv=cv, n_jobs=-1)
    gs_rf_bal.fit(X_train_imp, y_train)
    val_proba = gs_rf_bal.predict_proba(X_val_imp)[:, 1]
    candidates.append({
        "name": "RandomForest+class_weight_balanced",
        "pipeline": gs_rf_bal.best_estimator_,
        "hyperparameters": gs_rf_bal.best_params_,
        "cv_pr_auc": float(gs_rf_bal.best_score_),
        "metrics": evaluate(y_val, val_proba),
        "uses_smote": False,
    })

    # Candidate C: LogisticRegression + SMOTE (baseline, single config)
    lr_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    print("Fitting LogisticRegression + SMOTE...")
    cv_scores = []
    for tr_idx, te_idx in cv.split(X_train_imp, y_train):
        lr_pipe.fit(X_train_imp.iloc[tr_idx], y_train.iloc[tr_idx])
        p = lr_pipe.predict_proba(X_train_imp.iloc[te_idx])[:, 1]
        cv_scores.append(average_precision_score(y_train.iloc[te_idx], p))
    lr_pipe.fit(X_train_imp, y_train)
    val_proba = lr_pipe.predict_proba(X_val_imp)[:, 1]
    candidates.append({
        "name": "LogisticRegression+SMOTE",
        "pipeline": lr_pipe,
        "hyperparameters": {"max_iter": 2000},
        "cv_pr_auc": float(np.mean(cv_scores)),
        "metrics": evaluate(y_val, val_proba),
        "uses_smote": True,
    })

    # Candidate D: HistGradientBoosting + SMOTE (single config)
    hgb_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])
    print("Fitting HistGradientBoosting + SMOTE...")
    cv_scores = []
    for tr_idx, te_idx in cv.split(X_train_imp, y_train):
        hgb_pipe.fit(X_train_imp.iloc[tr_idx], y_train.iloc[tr_idx])
        p = hgb_pipe.predict_proba(X_train_imp.iloc[te_idx])[:, 1]
        cv_scores.append(average_precision_score(y_train.iloc[te_idx], p))
    hgb_pipe.fit(X_train_imp, y_train)
    val_proba = hgb_pipe.predict_proba(X_val_imp)[:, 1]
    candidates.append({
        "name": "HistGradientBoosting+SMOTE",
        "pipeline": hgb_pipe,
        "hyperparameters": {},
        "cv_pr_auc": float(np.mean(cv_scores)),
        "metrics": evaluate(y_val, val_proba),
        "uses_smote": True,
    })

    for c in candidates:
        print(f"  {c['name']}: CV PR-AUC={c['cv_pr_auc']:.4f} | VAL PR-AUC={c['metrics']['pr_auc']:.4f} "
              f"ROC-AUC={c['metrics']['roc_auc']:.4f} Recall={c['metrics']['recall']:.4f} "
              f"Brier={c['metrics']['brier_score']:.4f}")

    selected = select_best(candidates)
    print(f"\n>>> Selected: {selected['name']} <<<")

    # ─── 13. Calibration — CalibratedClassifierCV, SMOTE applied per-fold ───
    print("Calibrating selected model (sigmoid, cv=5, SMOTE inside each fold)...")
    calibrated = CalibratedClassifierCV(estimator=selected["pipeline"], method="sigmoid", cv=cv)
    calibrated.fit(X_train_imp, y_train)

    # ─── Final TEST evaluation (independent, never used for selection) ─────
    test_proba = calibrated.predict_proba(X_test_imp)[:, 1]
    test_metrics = evaluate(y_test, test_proba, threshold=RISK_THRESHOLDS["moderate_max"])
    frac_pos, mean_pred = calibration_curve(y_test, test_proba, n_bins=10, strategy="uniform")
    test_metrics["calibration_curve"] = {
        "mean_predicted_prob": [float(v) for v in mean_pred],
        "fraction_of_positives": [float(v) for v in frac_pos],
    }
    print("TEST metrics:", {k: v for k, v in test_metrics.items() if k != "calibration_curve"})

    roc_auc_target_met = test_metrics["roc_auc"] >= 0.80
    print(f"ROC-AUC >= 0.80 objective: {'MET' if roc_auc_target_met else 'NOT MET'} "
          f"(actual: {test_metrics['roc_auc']:.4f})")

    # ─── 15. Anomaly detector — IsolationForest on TRAIN normals only ───────
    print("Training anomaly detector (IsolationForest on y_train==0)...")
    X_train_normal = X_train_imp[y_train.values == 0]
    anomaly_model = IsolationForest(**ANOMALY_CONFIG)
    anomaly_model.fit(X_train_normal)

    # Sanity check semantics on TEST: anomaly_score<0 -> is_anomaly True
    test_anomaly_scores = anomaly_model.decision_function(X_test_imp)
    test_is_anomaly = test_anomaly_scores < 0
    anomaly_rate_test = float(test_is_anomaly.mean())
    print(f"Anomaly rate on TEST: {anomaly_rate_test:.4f}")

    # ─── 17. Feature importance — Permutation Importance on TEST, top 6 ────
    print("Computing permutation importance on TEST set...")
    perm = permutation_importance(
        calibrated, X_test_imp, y_test,
        scoring="average_precision", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    importances = pd.Series(perm.importances_mean, index=FEATURES).clip(lower=0)
    top6 = importances.sort_values(ascending=False).head(6)
    total = top6.sum()
    if total <= 0:
        # No candidate had positive importance — do not fabricate. Flag clearly.
        feature_importance_pct = {k: None for k in top6.index}
        feature_importance_note = "N/A — permutation importance was zero/negative for all features"
    else:
        raw_pct = (top6 / total * 100)
        rounded = raw_pct.round(2)
        # largest-remainder adjustment so the six values sum to exactly 100.00
        diff = round(100.00 - rounded.sum(), 2)
        if abs(diff) >= 0.01:
            idx = rounded.idxmax()
            rounded[idx] = round(rounded[idx] + diff, 2)
        feature_importance_pct = {k: float(v) for k, v in rounded.items()}
        feature_importance_note = None
    print("Top 6 feature importance (%):", feature_importance_pct)

    # ─── Persist artifacts ───────────────────────────────────────────────
    joblib.dump(calibrated, ARTIFACT_DIR / "classifier.joblib")
    joblib.dump(imputer, ARTIFACT_DIR / "imputer.joblib")
    joblib.dump(anomaly_model, ARTIFACT_DIR / "anomaly_model.joblib")

    with open(ARTIFACT_DIR / "risk_thresholds.json", "w") as f:
        json.dump(RISK_THRESHOLDS, f, indent=2)

    with open(ARTIFACT_DIR / "feature_importance.json", "w") as f:
        json.dump({
            "top_6": feature_importance_pct,
            "note": feature_importance_note,
            "method": "permutation_importance",
            "scoring": "average_precision",
            "evaluated_on": "test_set",
            "n_repeats": 10,
        }, f, indent=2)

    metrics_out = {
        "candidates": [
            {
                "name": c["name"],
                "cv_pr_auc": c["cv_pr_auc"],
                "hyperparameters": c["hyperparameters"],
                "uses_smote": c["uses_smote"],
                "validation_metrics": c["metrics"],
            }
            for c in candidates
        ],
        "selected_model": selected["name"],
        "selection_metric": "PR-AUC (validation) -> ROC-AUC -> Recall -> Brier -> simplicity",
        "test_metrics": test_metrics,
        "roc_auc_target_0_80_met": roc_auc_target_met,
        "anomaly_rate_on_test": anomaly_rate_test,
    }
    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "dataset": "framingham.csv (user-provided)",
        "dataset_hash_sha256": raw_hash,
        "dataset_shape_raw": list(df_raw.shape),
        "target": TARGET,
        "features": FEATURES,
        "excluded_features": EXCLUDED_FEATURES,
        "missing_values_processed_columns": {k: int(v) for k, v in missing_report.items()},
        "train_size": sizes["train"],
        "validation_size": sizes["validation"],
        "test_size": sizes["test"],
        "class_distribution": {
            split: {str(k): int(v) for k, v in d.items()} for split, d in class_dist.items()
        },
        "random_state": RANDOM_STATE,
        "smote_config": {"random_state": RANDOM_STATE, "k_neighbors": 5, "applied_to": "train_only_per_cv_fold"},
        "selected_algorithm": selected["name"],
        "selected_hyperparameters": selected["hyperparameters"],
        "selection_metric": "PR-AUC (primary), ROC-AUC/Recall/Brier as tie-breakers",
        "metrics": test_metrics,
        "calibration_method": "CalibratedClassifierCV(method='sigmoid', cv=StratifiedKFold(5))",
        "risk_thresholds": RISK_THRESHOLDS,
        "anomaly_algorithm": "IsolationForest",
        "anomaly_configuration": ANOMALY_CONFIG,
        "anomaly_semantics": "decision_function() < 0 => is_anomaly=true (more negative = more atypical)",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "dependency_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "imbalanced-learn": imblearn.__version__,
        },
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ─── Human-readable report ───────────────────────────────────────────
    report_lines = [
        f"# {MODEL_VERSION} — Training Report",
        "",
        f"Generated: {metadata['training_timestamp']}",
        "",
        "## Dataset",
        f"- Source: framingham.csv (user-provided), sha256 `{raw_hash}`",
        f"- Raw shape: {df_raw.shape}",
        f"- Features ({len(FEATURES)}): {', '.join(FEATURES)}",
        f"- Excluded features: {', '.join(EXCLUDED_FEATURES)}",
        f"- Target: {TARGET}",
        f"- Missing values (processed columns): {missing_report}",
        "",
        "## Split",
        f"- Train: {sizes['train']} | Validation: {sizes['validation']} | Test: {sizes['test']}",
        f"- Class distribution: {class_dist}",
        "",
        "## SMOTE",
        f"- {metadata['smote_config']}",
        "",
        "## Candidate models (validation set)",
    ]
    for c in candidates:
        report_lines.append(
            f"- **{c['name']}** — hyperparams={c['hyperparameters']} — "
            f"CV PR-AUC={c['cv_pr_auc']:.4f} | VAL PR-AUC={c['metrics']['pr_auc']:.4f} "
            f"ROC-AUC={c['metrics']['roc_auc']:.4f} Recall={c['metrics']['recall']:.4f} "
            f"Precision={c['metrics']['precision']:.4f} F1={c['metrics']['f1']:.4f} "
            f"Accuracy={c['metrics']['accuracy']:.4f} Brier={c['metrics']['brier_score']:.4f}"
        )
    report_lines += [
        "",
        f"## Selected model: {selected['name']}",
        f"- Rationale: highest PR-AUC on validation; ties broken by ROC-AUC, then Recall, "
        f"then lower Brier Score, then simpler/more stable model.",
        f"- Hyperparameters: {selected['hyperparameters']}",
        "",
        "## Calibration",
        f"- Method: {metadata['calibration_method']}",
        "",
        "## TEST metrics (final, independent)",
        f"- ROC-AUC: {test_metrics['roc_auc']:.4f}",
        f"- PR-AUC: {test_metrics['pr_auc']:.4f}",
        f"- Precision: {test_metrics['precision']:.4f}",
        f"- Recall: {test_metrics['recall']:.4f}",
        f"- F1: {test_metrics['f1']:.4f}",
        f"- Accuracy: {test_metrics['accuracy']:.4f}",
        f"- Brier Score: {test_metrics['brier_score']:.4f}",
        f"- ROC-AUC >= 0.80 objective: {'MET' if roc_auc_target_met else 'NOT MET'}",
        "",
        "## Anomaly detection",
        f"- {ANOMALY_CONFIG}",
        f"- Anomaly rate on TEST: {anomaly_rate_test:.4f}",
        "",
        "## Feature importance (Permutation Importance, TEST set, top 6)",
        f"- {feature_importance_pct}" if feature_importance_note is None else f"- {feature_importance_note}",
        "",
        "## Risk thresholds",
        f"- LOW: 0.00 <= score < {RISK_THRESHOLDS['low_max']}",
        f"- MODERATE: {RISK_THRESHOLDS['low_max']} <= score < {RISK_THRESHOLDS['moderate_max']}",
        f"- HIGH: {RISK_THRESHOLDS['moderate_max']} <= score <= 1.00",
        "",
        f"## Model version: {MODEL_VERSION}",
        f"## Dataset hash: {raw_hash}",
    ]
    (ARTIFACT_DIR / "REPORT.md").write_text("\n".join(report_lines))

    print(f"\nArtifacts written to {ARTIFACT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
