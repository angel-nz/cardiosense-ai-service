# Skorp-Beta-0.1 — CardioSense AI Service

Primer modelo ML real de CardioSense.

## Propósito

Servicio FastAPI que expone inferencia real de riesgo cardiovascular a 10
años (`TenYearCHD`) y detección de anomalías, consumido exclusivamente por
el backend Node/Express de CardioSense (`lib/aiClient.ts`) — nunca por el
frontend directamente.

## Dataset

- `data/raw/framingham.csv` — Framingham Heart Study (4240 filas, 16 columnas).
- Hash SHA-256 real calculado en cada entrenamiento y guardado en `metadata.json`.
- Target: `TenYearCHD` (0/1).
- Features usadas (12): `male, age, currentSmoker, cigsPerDay, BPMeds, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose`.
- Columnas excluidas: `education, prevalentStroke, prevalentHyp`.
- El dataset procesado es un subconjunto mínimo (`features + target`) generado en memoria por `scripts/train.py` — no se persiste un CSV procesado aparte.

## Mapeo sex → male

El dataset usa `male`; la API pública usa `sex` (0=femenino, 1=masculino) para
alinearse con el contrato real del backend. `sex=1` se mapea a `male=1` sin
invertir la semántica.

## Preprocessing

- Imputación por media (`SimpleImputer(strategy="mean")`), **fit solo en TRAIN**, reutilizada en validación/test/inferencia — persistida en `imputer.joblib`.
- Split 70/15/15 estratificado por `TenYearCHD`, `random_state=42`.
- SMOTE (`k_neighbors=5, random_state=42`) aplicado **solo dentro de cada fold de entrenamiento** (vía `imblearn.Pipeline` + `StratifiedKFold(5, shuffle=True, random_state=42)`), nunca sobre validación, test ni inferencia.

## Entrenamiento y selección

`scripts/train.py` (proceso offline, no expuesto por HTTP) evalúa 4 candidatos:

1. RandomForest + SMOTE (grid search: `n_estimators∈{100,200,300}`, `max_depth∈{None,10,20}`, `min_samples_split∈{2,5}`)
2. RandomForest + `class_weight="balanced"` (sin SMOTE — experimento independiente, mismo grid)
3. LogisticRegression + SMOTE (baseline)
4. HistGradientBoosting + SMOTE

Selección: mayor PR-AUC en validación → desempate por ROC-AUC → Recall →
menor Brier Score → modelo más simple. El TEST nunca se usa para seleccionar,
solo para el reporte final. Ver `artifacts/skorp-beta-0.1/REPORT.md` y
`metrics.json` para los resultados reales de la corrida vigente.

## Calibración

`CalibratedClassifierCV(method="sigmoid", cv=StratifiedKFold(5))` envolviendo
el pipeline ganador (SMOTE incluido dentro de cada fold de calibración) —
nunca se calibra sobre datos SMOTEados. Brier Score y curva de calibración
(10 bins) quedan en `metrics.json`.

## Anomalías

`IsolationForest(n_estimators=200, contamination=0.05, random_state=42)`,
entrenado únicamente con `X_train[y_train==0]` (pacientes sin `TenYearCHD`
en el conjunto de entrenamiento). `TenYearCHD` nunca es feature del detector.

Semántica: `anomaly_score = decision_function(X)`; `is_anomaly = anomaly_score < 0`
(más negativo = más atípico).

## Feature Importance

Permutation Importance (`scoring="average_precision"`, 10 repeticiones,
`random_state=42`) calculada sobre **TEST** (nunca sobre datos SMOTEados).
Top 6, normalizado a exactamente 100.00% con 2 decimales (ajuste por mayor
resto). Si todas las importancias fueran ≤0, se reporta `N/A` explícito en
`feature_importance.json` en vez de fabricar valores — no ocurrió en la
corrida vigente.

## API

- `GET /health` → `{status, model_loaded, model_version}` — sin datos sensibles.
- `POST /predict` → requiere header `X-Internal-Key` (debe igualar `AI_SERVICE_KEY` del backend). 401 si falta, 403 si es incorrecta. Contrato exacto en `app/schemas.py`, dictado por `backend/src/lib/aiClient.ts` (no al revés).
- No existen endpoints de entrenamiento/administración (`/train`, `/retrain`, `/models`) — el entrenamiento es 100% offline vía `scripts/train.py`.

## Ejecución

```bash
python -m pip install -r requirements.txt
python scripts/train.py          # skorp-beta-0.1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Tests

```bash
AI_SERVICE_KEY=internal-dev-key python3 -m pytest tests/ -v
```

`tests/test_api.py` cubre T1–T11 (health, auth, payload inválido, rangos,
umbrales de riesgo, coherencia de anomalía, feature importance, versión,
determinismo). `tests/test_training.py` valida los artefactos/metadata ya
generados (columnas, target, exclusiones, tamaños de split, SMOTE
train-only, calibración registrada, top-6 suma 100%).

## Limitaciones conocidas

- `buildFeatures()` en `aiClient.ts` envía siempre `sex: 0` (comentario del propio código: "Added in health record v2; default for now") — Skorp recibe correctamente ese valor y lo mapea a `male=0`, pero esto significa que **todas** las predicciones reales del backend ignoran el sexo real del paciente hasta que ese campo se agregue al modelo de Health Record. No es un bug de Skorp; es una limitación aguas arriba, documentada aquí y no corregida..
- ROC-AUC objetivo (≥0.80) — ver `metrics.json`/`REPORT.md` para el resultado real obtenido; no se manipuló el experimento para alcanzarlo artificialmente.
