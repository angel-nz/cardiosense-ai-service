"""Offline candidate checks; no service, application or database connection."""
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('block_a', ROOT / 'scripts/train_pre_t_a.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


@pytest.fixture(scope='module')
def data():
    raw = t.read_verified(ROOT / 'data/raw/framingham.csv')
    return t.split(raw[t.FEATURES], raw[t.TARGET])


def test_hash_mismatch_stops_before_parsing(tmp_path):
    bad = tmp_path / 'bad.csv'
    bad.write_text('not the approved dataset')
    with pytest.raises(ValueError, match='BLOCKED: dataset hash mismatch'):
        t.read_verified(bad)


def test_independent_rows_and_no_feature_duplicates(data):
    xt, xv, xs, *_ = data
    assert [len(xt), len(xv), len(xs)] == [4133, 886, 886]
    assert not (set(xt.index) & set(xs.index) or set(xt.index) & set(xv.index) or set(xs.index) & set(xv.index))
    audit = json.loads((t.OUT / 'dataset_audit.json').read_text())
    assert not any(audit['cross_split_feature_duplicates'].values())


def test_classifier_fold_preprocessing_and_feature_contract(data):
    xt, _, _, yt, *_ = data
    model = joblib.load(t.OUT / 'classifier.joblib')
    assert list(model.feature_names_in_) == t.FEATURES
    assert len(t.FEATURES) == 11
    assert 'heartRate' not in t.FEATURES
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for calibrated, (a, _) in zip(model.calibrated_classifiers_, cv.split(xt, yt)):
        pipe = calibrated.estimator
        assert pipe.n_features_in_ == 11
        np.testing.assert_allclose(pipe.named_steps['imputer'].statistics_, xt.iloc[a].mean().to_numpy())
        assert pipe.named_steps['scaler'].n_features_in_ == 11
        assert pipe.named_steps['smote'].n_features_in_ == 11
        assert pipe.named_steps['clf'].n_features_in_ == 11


def test_persisted_predictions_metrics_and_strata(data):
    _, _, xs, _, _, ys = data
    model = joblib.load(t.OUT / 'classifier.joblib')
    p = model.predict_proba(xs)[:, 1]
    saved = pd.read_csv(t.OUT / 'test_predictions.csv')
    np.testing.assert_allclose(p, saved.probability, atol=1e-15)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    report = json.loads((t.OUT / 'metrics.json').read_text())
    for threshold in [.20, .35]:
        assert t.evaluate(ys, p, threshold) == report['test_by_threshold'][str(threshold)]
    for name, values in report['test_strata'].items():
        mask = t.levels(p) == name
        assert values['count'] == int(mask.sum())
        assert values['positives'] == int(ys.to_numpy()[mask].sum())
    assert sum(v['count'] for v in report['test_strata'].values()) == len(xs)


def test_anomaly_artifacts_train_only(data):
    xt, _, xs, yt, *_ = data
    imputer = joblib.load(t.OUT / 'imputer.joblib')
    model = joblib.load(t.OUT / 'anomaly_model.joblib')
    assert list(imputer.feature_names_in_) == t.FEATURES
    assert list(model.feature_names_in_) == t.FEATURES
    np.testing.assert_allclose(imputer.statistics_, xt.mean())
    assert model.n_features_in_ == 11
    # Refit verifies the persisted detector really used only negative TRAIN rows.
    expected = t.IsolationForest(n_estimators=200, contamination=.05, random_state=42)
    expected.fit(pd.DataFrame(imputer.transform(xt[yt == 0]), columns=t.FEATURES))
    transformed = pd.DataFrame(imputer.transform(xs), columns=t.FEATURES)
    np.testing.assert_allclose(model.decision_function(transformed), expected.decision_function(transformed))


def test_importance_normalization_and_complete_contract():
    obj = json.loads((t.OUT / 'feature_importance.json').read_text())
    assert obj['n_repeats'] == 50 and obj['random_state'] == 42
    assert obj['scoring'] == 'average_precision'
    rows = obj['features']
    assert [r['feature'] for r in rows] == t.FEATURES
    means = np.array([r['importance_mean_raw'] for r in rows])
    stds = np.array([r['importance_std_raw'] for r in rows])
    assert np.isfinite(means).all() and np.isfinite(stds).all() and (stds >= 0).all()
    assert rows == t.normalize_importance(means, stds)
    assert sum(r['importance_percent'] for r in rows) == pytest.approx(100)
    synthetic = t.normalize_importance(np.array([-2., 0., 1.] + [0.] * 8), np.zeros(11))
    assert [r['importance_percent'] for r in synthetic[:3]] == [0, 0, 100]
    assert all(r['importance_percent'] == 0 for r in t.normalize_importance(-np.ones(11), np.zeros(11)))


def test_unrounded_risk_boundaries():
    p = np.array([.19999999, .20, .34999999, .35])
    assert t.levels(p).tolist() == ['LOW', 'MODERATE', 'MODERATE', 'HIGH']


def test_versioned_metadata():
    meta = json.loads((t.OUT / 'metadata.json').read_text())
    assert meta['model_version'] == 'Skorp-Beta-0.2'
    assert meta['features'] == t.FEATURES
    assert meta['dataset_hash_sha256'] == t.EXPECTED_HASH
    assert 'heartRate' not in json.dumps(meta)
    assert meta['deployment_status'] == 'offline candidate only'


REQUIRED_METADATA = ['model_name', 'model_version', 'features', 'target', 'dataset_hash_sha256',
    'dataset_shape_raw', 'random_state', 'selected_algorithm', 'calibration_method', 'preprocessing',
    'smote_config', 'anomaly_configuration', 'training_age_range', 'eligible_age_range',
    'eligible_age_range_basis', 'split_sizes', 'class_distribution', 'python_version',
    'dependency_versions', 'training_timestamp', 'deployment_status', 'excluded_features',
    'missing_values_processed_columns', 'selected_hyperparameters', 'risk_thresholds', 'metrics']


@pytest.mark.parametrize('field', REQUIRED_METADATA)
def test_required_metadata_present(field):
    meta = json.loads((t.OUT / 'metadata.json').read_text())
    assert field in meta and meta[field] is not None


def test_beta02_eligibility_and_metadata_values(data):
    meta = json.loads((t.OUT / 'metadata.json').read_text())
    assert meta['training_age_range'] == meta['eligible_age_range'] == {'min': 32, 'max': 81}
    assert meta['excluded_features'] == ['education', 'prevalentStroke', 'prevalentHyp']
    raw = t.read_verified(ROOT / 'data/raw/framingham.csv')
    assert meta['missing_values_processed_columns'] == {f: int(raw[f].isna().sum()) for f in t.FEATURES}
    assert meta['selected_hyperparameters'] == {'max_iter': 2000, 'random_state': 42}
    assert meta['risk_thresholds'] == {'low_max': .20, 'moderate_max': .35}
    metrics = json.loads((t.OUT / 'metrics.json').read_text())
    assert meta['metrics'] == {**metrics['test_by_threshold']['0.35'], 'calibration_curve': metrics['calibration_curve']}


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, -.001, 1.001])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError, match='finite and in'):
        t.levels(np.array([.1, bad]))
    with pytest.raises(ValueError, match='finite and in'):
        t.evaluate([0, 1], np.array([.1, bad]), .35)


@pytest.mark.parametrize('threshold', [.20, .35])
def test_threshold_metrics_from_confusion_matrix(data, threshold):
    _, _, xs, _, _, ys = data
    p = joblib.load(t.OUT / 'classifier.joblib').predict_proba(xs)[:, 1]
    positive = p >= threshold
    tp = int(((ys == 1) & positive).sum())
    tn = int(((ys == 0) & ~positive).sum())
    fp = int(((ys == 0) & positive).sum())
    fn = int(((ys == 1) & ~positive).sum())
    v = t.evaluate(ys, p, threshold)
    assert [v[k] for k in ['tp','tn','fp','fn']] == [tp,tn,fp,fn]
    assert v['sensitivity'] == v['recall'] == tp / (tp + fn)
    assert v['specificity'] == tn / (tn + fp)
    assert v['npv'] == tn / (tn + fn)
    assert v['ppv'] == v['precision'] == tp / (tp + fp)
    assert v['f1'] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert v['accuracy'] == (tp + tn) / len(ys)


def test_seeded_pipeline_reproducibility(data):
    # Two independent fits of the actual training pipeline on TRAIN only.
    # Explicit tolerance for this environment: rtol=0, atol=1e-12.
    xt, xv, _, yt, *_ = data
    a, b = t.pipeline().fit(xt, yt), t.pipeline().fit(xt, yt)
    assert a.named_steps['smote'].random_state == b.named_steps['smote'].random_state == 42
    assert a.named_steps['clf'].random_state == b.named_steps['clf'].random_state == 42
    np.testing.assert_allclose(a.predict_proba(xv), b.predict_proba(xv), rtol=0, atol=1e-12)
    np.testing.assert_allclose(a.named_steps['clf'].coef_, b.named_steps['clf'].coef_, rtol=0, atol=1e-12)
    cv1, cv2 = [StratifiedKFold(5, shuffle=True, random_state=42) for _ in range(2)]
    for (a1,b1), (a2,b2) in zip(cv1.split(xt,yt),cv2.split(xt,yt)):
        np.testing.assert_array_equal(a1,a2)
        np.testing.assert_array_equal(b1,b2)


def test_coefficient_diagnostics_match_five_estimators():
    model = joblib.load(t.OUT / 'classifier.joblib')
    obj = json.loads((t.OUT / 'coefficient_diagnostics.json').read_text())
    assert [r['feature'] for r in obj['features']] == t.FEATURES
    assert len(model.calibrated_classifiers_) == 5
    values = np.array([c.estimator.named_steps['clf'].coef_[0] for c in model.calibrated_classifiers_])
    for i, row in enumerate(obj['features']):
        v = values[:, i]
        np.testing.assert_array_equal(row['coefficients_by_fold'],v)
        assert row['mean_coefficient'] == v.mean()
        assert row['std_coefficient'] == v.std(ddof=0)
        assert row['min'] == v.min() and row['max'] == v.max()
        assert row['sign_counts'] == dict(positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum()))
        assert row['sign_consistency_fraction'] == max(row['sign_counts'].values())/5
        assert row['same_sign_all_folds'] == (len(set(np.sign(v)))==1)


def test_baseline_calibration_and_anomaly_comparison():
    m = json.loads((t.OUT / 'metrics.json').read_text())
    b = json.loads((ROOT / 'artifacts/skorp-beta-0.1/metrics.json').read_text())
    c = m['calibration_comparison']
    assert c['baseline_12']['curve'] == b['test_metrics']['calibration_curve']
    assert c['candidate_11']['curve'] == m['calibration_curve']
    assert c['baseline_12']['brier_score'] == b['test_metrics']['brier_score']
    assert c['candidate_11']['brier_score'] == m['test_by_threshold']['0.35']['brier_score']
    a = m['anomaly_comparison']
    assert a['baseline_12'] == b['anomaly_rate_on_test']
    assert a['candidate_11'] == m['anomaly_rate_test']
    assert a['delta'] == a['candidate_11']-a['baseline_12']


def test_importance_csv_json_agree():
    csv = pd.read_csv(t.OUT / 'feature_importance.csv')
    rows = json.loads((t.OUT / 'feature_importance.json').read_text())['features']
    assert len(csv) == len(rows) == 11
    assert csv.feature.tolist() == t.FEATURES
    for key in ['importance_mean_raw','importance_std_raw','importance_percent']:
        np.testing.assert_allclose(csv[key], [r[key] for r in rows], rtol=0, atol=1e-14)
    assert csv.importance_percent.sum() == pytest.approx(100)
