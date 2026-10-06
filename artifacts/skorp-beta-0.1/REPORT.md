# Skorp-Beta-0.1 — Training Report

Generated: 2026-10-04T23:15:26.364650+00:00

## Dataset
- Source: framingham.csv (user-provided), sha256 `533fe2625b5aace28c21714cb9ba55ec35cba11177c31ad5d7a20c79101c8e76`
- Raw shape: (5905, 16)
- Features (12): male, age, currentSmoker, cigsPerDay, BPMeds, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose
- Excluded features: education, prevalentStroke, prevalentHyp
- Target: TenYearCHD
- Missing values (processed columns): {'male': 0, 'age': 0, 'currentSmoker': 0, 'cigsPerDay': 42, 'BPMeds': 176, 'diabetes': 0, 'totChol': 127, 'sysBP': 0, 'diaBP': 0, 'BMI': 31, 'heartRate': 3, 'glucose': 629}

## Split
- Train: 4133 | Validation: 886 | Test: 886
- Class distribution: {'train': {0: 3381, 1: 752}, 'validation': {0: 724, 1: 162}, 'test': {0: 725, 1: 161}}

## SMOTE
- {'random_state': 42, 'k_neighbors': 5, 'applied_to': 'train_only_per_cv_fold'}

## Candidate models (validation set)
- **RandomForest+SMOTE** — hyperparams={'clf__max_depth': 10, 'clf__min_samples_split': 5, 'clf__n_estimators': 300} — CV PR-AUC=0.3362 | VAL PR-AUC=0.3451 ROC-AUC=0.7220 Recall=0.6543 Precision=0.3222 F1=0.4318 Accuracy=0.6851 Brier=0.1538
- **RandomForest+class_weight_balanced** — hyperparams={'clf__max_depth': 10, 'clf__min_samples_split': 5, 'clf__n_estimators': 300} — CV PR-AUC=0.3367 | VAL PR-AUC=0.3520 ROC-AUC=0.7365 Recall=0.8333 Precision=0.2652 F1=0.4024 Accuracy=0.5474 Brier=0.1780
- **LogisticRegression+SMOTE** — hyperparams={'max_iter': 2000} — CV PR-AUC=0.3658 | VAL PR-AUC=0.4185 ROC-AUC=0.7600 Recall=0.8827 Precision=0.2453 F1=0.3839 Accuracy=0.4819 Brier=0.2052
- **HistGradientBoosting+SMOTE** — hyperparams={} — CV PR-AUC=0.3168 | VAL PR-AUC=0.3396 ROC-AUC=0.6896 Recall=0.4506 Precision=0.3443 F1=0.3904 Accuracy=0.7427 Brier=0.1461

## Selected model: LogisticRegression+SMOTE
- Rationale: highest PR-AUC on validation; ties broken by ROC-AUC, then Recall, then lower Brier Score, then simpler/more stable model.
- Hyperparameters: {'max_iter': 2000}

## Calibration
- Method: CalibratedClassifierCV(method='sigmoid', cv=StratifiedKFold(5))

## TEST metrics (final, independent)
- ROC-AUC: 0.7268
- PR-AUC: 0.3515
- Precision: 0.4096
- Recall: 0.2112
- F1: 0.2787
- Accuracy: 0.8014
- Brier Score: 0.1352
- ROC-AUC >= 0.80 objective: NOT MET

## Anomaly detection
- {'n_estimators': 200, 'contamination': 0.05, 'random_state': 42}
- Anomaly rate on TEST: 0.0666

## Feature importance (Permutation Importance, TEST set, top 6)
- {'age': 50.97, 'male': 16.03, 'cigsPerDay': 14.26, 'sysBP': 13.25, 'diabetes': 2.87, 'diaBP': 2.62}

## Risk thresholds
- LOW: 0.00 <= score < 0.2
- MODERATE: 0.2 <= score < 0.35
- HIGH: 0.35 <= score <= 1.00

## Model version: Skorp-Beta-0.1
## Dataset hash: 533fe2625b5aace28c21714cb9ba55ec35cba11177c31ad5d7a20c79101c8e76