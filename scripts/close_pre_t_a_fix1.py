"""Refresh reports/metadata from existing frozen models; never fit or overwrite models."""
import json
import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from train_pre_t_a import ROOT, OUT, FEATURES, read_verified, split, evaluate, dump


def coefficient_diagnostics(model):
    values = np.array([c.estimator.named_steps['clf'].coef_[0]
                       for c in model.calibrated_classifiers_])
    if values.shape != (5, 11):
        raise ValueError('Expected five base estimators with eleven coefficients each')
    rows = []
    for i, feature in enumerate(FEATURES):
        v = values[:, i]
        counts = dict(positive=int((v > 0).sum()), negative=int((v < 0).sum()), zero=int((v == 0).sum()))
        rows.append(dict(feature=feature, coefficients_by_fold=v.tolist(),
                         mean_coefficient=float(v.mean()), std_coefficient=float(v.std(ddof=0)),
                         min=float(v.min()), max=float(v.max()), sign_counts=counts,
                         sign_consistency_fraction=max(counts.values()) / len(v),
                         same_sign_all_folds=len(set(np.sign(v))) == 1))
    return dict(features=rows, fold_order='calibrated_classifiers_ order; fold indices 1 through 5',
                std_definition='Population standard deviation, ddof=0',
                interpretation='Coefficients of the internal LogisticRegression estimators in each fold-specific standardized space. The prediction model is the calibrated ensemble. These coefficients are not permutation importance, not percentages, and do not imply causality.',
                sign_consistency_definition='Largest positive/negative/zero sign count divided by five; exact zero is its own category.')


def close():
    raw = read_verified(ROOT / 'data/raw/framingham.csv')
    _, xv, xs, _, yv, ys = split(raw[FEATURES], raw.TenYearCHD)
    model = joblib.load(OUT / 'classifier.joblib')
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError('Unexpected candidate features')
    p = model.predict_proba(xs)[:, 1]
    metrics = json.loads((OUT / 'metrics.json').read_text())
    metrics['test_by_threshold'] = {str(t): evaluate(ys, p, t) for t in [.20, .35]}
    metrics['validation_metrics'] = evaluate(yv, model.predict_proba(xv)[:, 1], .35)
    frac, mean = calibration_curve(ys, p, n_bins=10, strategy='uniform')
    metrics['calibration_curve'] = dict(mean_predicted_prob=mean.tolist(), fraction_of_positives=frac.tolist())
    base = json.loads((ROOT / 'artifacts/skorp-beta-0.1/metrics.json').read_text())
    b, c = base['anomaly_rate_on_test'], metrics['anomaly_rate_test']
    metrics['anomaly_comparison'] = dict(baseline_12=b, candidate_11=c, delta=c-b)
    metrics['calibration_comparison'] = dict(
        method='calibration_curve, n_bins=10, strategy=uniform; empty bins omitted',
        baseline_12=dict(brier_score=base['test_metrics']['brier_score'],
                         curve=base['test_metrics']['calibration_curve']),
        candidate_11=dict(brier_score=metrics['test_by_threshold']['0.35']['brier_score'],
                          curve=metrics['calibration_curve']),
        brier_delta=metrics['test_by_threshold']['0.35']['brier_score']-base['test_metrics']['brier_score'])
    metrics['comparison_design'] = 'Descriptive only. Supplied baseline reports; no significance claim, no clinical improvement claim, no exclusive attribution to feature removal.'
    dump('metrics.json', metrics)
    meta = json.loads((OUT / 'metadata.json').read_text())
    meta.update(excluded_features=['education', 'prevalentStroke', 'prevalentHyp'],
                missing_values_processed_columns={f:int(raw[f].isna().sum()) for f in FEATURES},
                selected_hyperparameters=dict(max_iter=2000, random_state=42),
                risk_thresholds=dict(low_max=.20, moderate_max=.35),
                metrics={**metrics['test_by_threshold']['0.35'], 'calibration_curve': metrics['calibration_curve']})
    dump('metadata.json', meta)
    dump('coefficient_diagnostics.json', coefficient_diagnostics(model))
    print('FIX1: metadata, metrics and coefficients refreshed; no model fitting or binary writes.')


if __name__ == '__main__':
    close()
