"""Official offline CardioSense training entrypoint for Skorp-Beta-0.2.

This source encodes the approved 11-feature PRE-T-A training contract. It is
never invoked by the HTTP runtime. The checked-in approved Beta-0.2 artifacts
are frozen inputs; this script refuses to overwrite their existing directory.
"""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import imblearn
import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASH = '533fe2625b5aace28c21714cb9ba55ec35cba11177c31ad5d7a20c79101c8e76'
FEATURES = ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'diabetes',
            'totChol', 'sysBP', 'diaBP', 'BMI', 'glucose']
VERSION = 'Skorp-Beta-0.2'
MODEL_NAME = 'Skorp'
MODEL_VERSION = VERSION
RANDOM_STATE = 42
TARGET = 'TenYearCHD'
RAW_CSV = ROOT / 'data' / 'raw' / 'framingham.csv'
OUT = ROOT / 'artifacts' / 'skorp-beta-0.2'
ARTIFACT_DIR = OUT



def compute_eligibility_metadata(age_series: pd.Series) -> dict:
    """Derive model support from the observed training population.

    This is model eligibility/support metadata, not a clinically validated
    age range. The returned dictionaries are independent objects.
    """
    age_min = int(age_series.min())
    age_max = int(age_series.max())
    training = {"min": age_min, "max": age_max}
    eligible = dict(training)
    return {
        "training_age_range": training,
        "eligible_age_range": eligible,
        "eligible_age_range_basis": "Observed dataset support only, not clinical validation.",
    }

def read_verified(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_HASH:
        raise ValueError(f'BLOCKED: dataset hash mismatch: {digest}')
    return pd.read_csv(path)


def split(X, y):
    train, temp, yt, ytemp = train_test_split(X, y, test_size=.30, stratify=y, random_state=42)
    val, test, yv, ys = train_test_split(temp, ytemp, test_size=.50, stratify=ytemp, random_state=42)
    return train, val, test, yt, yv, ys


def pipeline():
    # Preprocessing is fitted inside each calibration/CV training fold.
    return Pipeline([('imputer', SimpleImputer(strategy='mean')),
                     ('scaler', StandardScaler()), ('smote', SMOTE(random_state=42, k_neighbors=5)),
                     ('clf', LogisticRegression(max_iter=2000, random_state=42))])


def evaluate(y, p, threshold):
    p = validate_probabilities(p)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
                accuracy=float(accuracy_score(y, pred)), brier_score=float(brier_score_loss(y, p)),
                sensitivity=float(tp / (tp + fn)) if tp + fn else 0.,
                specificity=float(tn / (tn + fp)) if tn + fp else 0.,
                ppv=float(tp / (tp + fp)) if tp + fp else 0.,
                npv=float(tn / (tn + fn)) if tn + fn else 0.,
                threshold_used=threshold, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def validate_probabilities(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and in [0, 1]')
    return p


def levels(p):
    p = validate_probabilities(p)
    return np.where(p < .20, 'LOW', np.where(p < .35, 'MODERATE', 'HIGH'))


def normalize_importance(means, stds):
    positive = np.maximum(means, 0)
    total = positive.sum()
    return [dict(feature=f, importance_mean_raw=float(m), importance_std_raw=float(s),
                 importance_percent=float(100 * pos / total) if total > 0 else 0.)
            for f, m, s, pos in zip(FEATURES, means, stds, positive)]


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def main():
    raw = read_verified(RAW_CSV)
    X, y = raw[FEATURES].copy(), raw[TARGET].copy()
    if y.isna().any() or set(y.unique()) != {0, 1} or np.isinf(X.to_numpy()).any():
        raise ValueError('BLOCKED: invalid target or infinite feature values')
    for f in ['male', 'currentSmoker', 'BPMeds', 'diabetes']:
        if not set(X[f].dropna().unique()) <= {0, 1}:
            raise ValueError(f'BLOCKED: invalid binary feature {f}')
    xt, xv, xs, yt, yv, ys = split(X, y)
    rowhash = pd.util.hash_pandas_object(X, index=False)
    overlaps = {f'{a}_{b}': len(set(rowhash.loc[u.index]) & set(rowhash.loc[v.index]))
                for a, u, b, v in [('train', xt, 'test', xs), ('train', xt, 'validation', xv),
                                   ('validation', xv, 'test', xs)]}
    if any(overlaps.values()):
        raise ValueError(f'BLOCKED: duplicate feature vectors cross splits: {overlaps}')
    OUT.mkdir(parents=True, exist_ok=False)
    audit = dict(raw_shape=list(raw.shape), dataset_hash_sha256=EXPECTED_HASH,
                 duplicate_raw_rows=int(raw.duplicated().sum()), duplicate_feature_rows=int(X.duplicated().sum()),
                 cross_split_feature_duplicates=overlaps, subject_identifiers_available=False,
                 limitation='Subject-level leakage cannot be independently ruled out: no subject identifiers are available.',
                 missing={k: int(v) for k, v in X.isna().sum().items()},
                 ranges={f: dict(min=float(X[f].min()), max=float(X[f].max())) for f in FEATURES},
                 smoker_zero_cigarettes=int(((X.currentSmoker == 1) & (X.cigsPerDay == 0)).sum()),
                 nonsmoker_positive_cigarettes=int(((X.currentSmoker == 0) & (X.cigsPerDay > 0)).sum()),
                 action='No rows removed, no clipping; mean imputation fitted on training data only.')
    dump('dataset_audit.json', audit)
    splits = {name: frame.index.tolist() for name, frame in [('train', xt), ('validation', xv), ('test', xs)]}
    dump('split_indices.json', splits)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_ap = []
    for a, b in cv.split(xt, yt):
        m = pipeline().fit(xt.iloc[a], yt.iloc[a])
        cv_ap.append(float(average_precision_score(yt.iloc[b], m.predict_proba(xt.iloc[b])[:, 1])))
    model = CalibratedClassifierCV(estimator=pipeline(), method='sigmoid', cv=cv)
    model.fit(xt, yt)
    p = model.predict_proba(xs)[:, 1]
    metrics = {str(t): evaluate(ys, p, t) for t in [.20, .35]}
    strata = {}
    for name in ['LOW', 'MODERATE', 'HIGH']:
        mask = levels(p) == name
        strata[name] = dict(count=int(mask.sum()), positives=int(ys.to_numpy()[mask].sum()),
                            observed_event_rate=float(ys.to_numpy()[mask].mean()) if mask.any() else None,
                            mean_probability=float(p[mask].mean()) if mask.any() else None)
    frac, mean = calibration_curve(ys, p, n_bins=10, strategy='uniform')
    imputer = SimpleImputer(strategy='mean').fit(xt)
    anomaly = IsolationForest(n_estimators=200, contamination=.05, random_state=42)
    anomaly.fit(pd.DataFrame(imputer.transform(xt[yt == 0]), columns=FEATURES))
    scores = anomaly.decision_function(pd.DataFrame(imputer.transform(xs), columns=FEATURES))
    perm = permutation_importance(model, xs, ys, scoring='average_precision', n_repeats=50, random_state=42, n_jobs=1)
    importance = normalize_importance(perm.importances_mean, perm.importances_std)
    dump('feature_importance.json', dict(features=importance, method='permutation_importance', scoring='average_precision',
         evaluated_on='independent held-out TEST rows', n_repeats=50, random_state=42,
         label='Global normalized predictive-dependence percentage according to permutation importance using average precision on TEST.',
         limitations='Not causality or a percentage of individual patient risk. Correlated variables can share or suppress importance.',
         positive_mean_sum=float(np.maximum(perm.importances_mean, 0).sum())))
    baseline = json.loads((ROOT / 'artifacts/skorp-beta-0.1/metadata.json').read_text())
    comparison = {k: dict(baseline_12=baseline['metrics'][k], candidate_11=metrics['0.35'][k],
                         delta=metrics['0.35'][k] - baseline['metrics'][k])
                  for k in ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'accuracy', 'brier_score']}
    dump('metrics.json', dict(test_by_threshold=metrics, test_strata=strata, comparison_to_reported_baseline=comparison,
         baseline_source='Supplied historical metadata; historical model not loaded or retrained.',
         comparison_limitation='Same row split and model family; candidate moves imputation inside CV folds and uses recorded dependency versions. Differences cannot be attributed exclusively to feature removal.',
         cv_uncalibrated_pr_auc=cv_ap, validation_metrics=evaluate(yv, model.predict_proba(xv)[:, 1], .35),
         calibration_curve=dict(mean_predicted_prob=mean.tolist(), fraction_of_positives=frac.tolist()),
         anomaly_rate_test=float((scores < 0).mean()), roc_auc_target_0_80_met=metrics['0.35']['roc_auc'] >= .80))
    dump('risk_thresholds.json', dict(low_max=.20, moderate_max=.35))
    dump('metadata.json', dict(model_name='Skorp', model_version=VERSION, features=FEATURES, target=TARGET,
         dataset_hash_sha256=EXPECTED_HASH, dataset_shape_raw=list(raw.shape), random_state=42,
         selected_algorithm='LogisticRegression+SMOTE', selection='Family/config fixed before TEST; no candidate search or TEST tuning.',
         calibration_method="CalibratedClassifierCV(method='sigmoid', cv=StratifiedKFold(5, shuffle=True, random_state=42))",
         preprocessing='Classifier accepts raw 11-feature frames; each calibrated estimator owns its imputer/scaler. Separate imputer is for IsolationForest only.',
         smote_config=dict(random_state=42, k_neighbors=5, applied_to='train_only_per_cv_fold'),
         anomaly_configuration=dict(n_estimators=200, contamination=.05, training_population='TRAIN y==0', features=FEATURES),
         **compute_eligibility_metadata(X.age),
         split_sizes={k: len(v) for k, v in splits.items()},
         class_distribution={k: {str(c): int(n) for c, n in y.loc[v].value_counts().items()} for k, v in splits.items()},
         python_version=platform.python_version(), dependency_versions=dict(numpy=np.__version__, pandas=pd.__version__,
         sklearn=sklearn.__version__, imblearn=imblearn.__version__, joblib=joblib.__version__),
         training_timestamp=datetime.now(timezone.utc).isoformat(), deployment_status='offline candidate only'))
    for name, obj in [('classifier', model), ('imputer', imputer), ('anomaly_model', anomaly)]:
        joblib.dump(obj, OUT / f'{name}.joblib')
    pd.DataFrame(dict(row_index=xs.index, target=ys, probability=p, risk_level=levels(p), anomaly_score=scores)).to_csv(OUT / 'test_predictions.csv', index=False)
    pd.DataFrame(importance).to_csv(OUT / 'feature_importance.csv', index=False)
    # Use the same closure path on future explicit offline training runs.
    try:
        from scripts.close_pre_t_a_fix1 import close
    except ModuleNotFoundError:  # direct `python scripts/train.py` execution
        from close_pre_t_a_fix1 import close
    close()
    print(json.dumps(dict(metrics=metrics, strata=strata, importance=importance), indent=2))


if __name__ == '__main__':
    main()
