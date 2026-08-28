# Skorp-Beta-0.1 - Training Report

Generated: 2026-08-17T18:47:38.986896+00:00

## Dataset

- Source: framingham.csv (user-provided), sha256 `2c0e57dc0361b420becf1facec0a054af06c420eae0ae2faf0fdc8591fadb018`
- Raw shape: (4240, 16)
- Features (12): male, age, currentSmoker, cigsPerDay, BPMeds, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose
- Excluded features: education, prevalentStroke, prevalentHyp
- Target: TenYearCHD
- Missing values (processed columns): {'male': 0, 'age': 0, 'currentSmoker': 0, 'cigsPerDay': 29, 'BPMeds': 53, 'diabetes': 0, 'totChol': 50, 'sysBP': 0, 'diaBP': 0, 'BMI': 19, 'heartRate': 1, 'glucose': 388}

## Split

- Train: 2968 | Validation: 636 | Test: 636
- Class distribution: {'train': {0: 2517, 1: 451}, 'validation': {0: 539, 1: 97}, 'test': {0: 540, 1: 96}}

## SMOTE

- {'random_state': 42, 'k_neighbors': 5, 'applied_to': 'train_only_per_cv_fold'}

## Candidate models (validation set)

- **RandomForest+SMOTE** = hyperparams={'clf__max_depth': 10, 'clf__min_samples_split': 5, 'clf__n_estimators': 200} � CV PR-AUC=0.3156 | VAL PR-AUC=0.2566 ROC-AUC=0.6742 Recall=0.5876 Precision=0.2556 F1=0.3563 Accuracy=0.6761 Brier=0.1536
- **RandomForest+class_weight_balanced** = hyperparams={'clf__max_depth': 10, 'clf__min_samples_split': 2, 'clf__n_estimators': 300} � CV PR-AUC=0.3342 | VAL PR-AUC=0.2638 ROC-AUC=0.6600 Recall=0.7010 Precision=0.2186 F1=0.3333 Accuracy=0.5723 Brier=0.1736
- **LogisticRegression+SMOTE** = hyperparams={'max_iter': 2000} = CV PR-AUC=0.3542 | VAL PR-AUC=0.2751 ROC-AUC=0.6979 Recall=0.8351 Precision=0.2154 F1=0.3425 Accuracy=0.5110 Brier=0.2161
- **HistGradientBoosting+SMOTE** = hyperparams={} = CV PR-AUC=0.2889 | VAL PR-AUC=0.2434 ROC-AUC=0.6452 Recall=0.3299 Precision=0.2540 F1=0.2870 Accuracy=0.7500 Brier=0.1443

## Selected model: LogisticRegression+SMOTE

- Rationale: highest PR-AUC on validation; ties broken by ROC-AUC, then Recall, then lower Brier Score, then simpler/more stable model.
- Hyperparameters: {'max_iter': 2000}

## Calibration

- Method: CalibratedClassifierCV(method='sigmoid', cv=StratifiedKFold(5))

## TEST metrics (final, independent)

- ROC-AUC: 0.6948
- PR-AUC: 0.3420
- Precision: 0.4524
- Recall: 0.1979
- F1: 0.2754
- Accuracy: 0.8428
- Brier Score: 0.1181
- ROC-AUC >= 0.80 objective: NOT MET

## Anomaly detection

- {'n_estimators': 200, 'contamination': 0.05, 'random_state': 42}
- Anomaly rate on TEST: 0.0660

## Feature importance (Permutation Importance, TEST set, top 6)

- {'age': 39.25, 'sysBP': 22.32, 'cigsPerDay': 16.43, 'male': 11.76, 'glucose': 6.44, 'diaBP': 3.8}

## Risk thresholds

- LOW: 0.00 <= score < 0.2
- MODERATE: 0.2 <= score < 0.35
- HIGH: 0.35 <= score <= 1.00

## Model version: Skorp-Beta-0.1

## Dataset hash: 2c0e57dc0361b420becf1facec0a054af06c420eae0ae2faf0fdc8591fadb018
