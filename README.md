# Skorp-Beta-0.2 — CardioSense AI Service

Runtime FastAPI de CardioSense para inferencia de riesgo cardiovascular a 10 años (`TenYearCHD`), detección de anomalías, personalización Phase R y scoring de forecast Phase S. El servicio es consumido por el backend; el frontend no lo invoca directamente.

## Contrato runtime aprobado

PRE-T-B promueve los artefactos ya aprobados de `artifacts/skorp-beta-0.2/`; el runtime **no reentrena** modelos. El clasificador recibe exactamente estas 11 variables, en este orden:

`male, age, currentSmoker, cigsPerDay, BPMeds, diabetes, totChol, sysBP, diaBP, BMI, glucose`

La API usa `sex` (0=female, 1=male) y el runtime lo mapea a la columna de modelo `male`. Los schemas Pydantic son estrictos (`extra=forbid`), por lo que campos obsoletos o desconocidos se rechazan con 422.

El soporte observado del modelo expuesto por `/health` es **32–81**. Este rango representa elegibilidad del modelo basada en soporte observado, no un rango clínico validado ni la validación general de un HealthRecord.

## Artefactos

La fuente runtime es `artifacts/skorp-beta-0.2/`:

- `classifier.joblib`: LogisticRegression + SMOTE, calibrado con sigmoid.
- `imputer.joblib`: imputador separado usado por IsolationForest.
- `anomaly_model.joblib`: IsolationForest sobre las mismas 11 variables.
- `feature_importance.json`: permutation importance aprobada sobre TEST independiente, `average_precision`, 50 repeticiones, seed 42.
- `risk_thresholds.json`: umbrales canónicos 0.20 / 0.35.
- `metadata.json`: identidad, features, soporte, hashes y métricas del artefacto.

La feature importance runtime transporta **las 11 variables completas**, incluidas las variables con importancia normalizada 0%. No se recalcula por request y no representa causalidad ni porcentaje de riesgo individual.

## API

- `GET /health` → `status`, `model_loaded`, `model_version`, `eligible_age_range`, `training_age_range`.
- `POST /predict` → requiere `X-Internal-Key`; recibe exactamente los 11 inputs clínicos del contrato backend y opcionalmente el bloque de Phase R.
- `POST /forecast/score` → scoring interno de Phase S con el mismo contrato de 11 variables.
- No hay endpoints de entrenamiento o administración de modelos.

Los niveles de riesgo se clasifican siempre sobre la probabilidad **sin redondear**: LOW `<0.20`, MODERATE `0.20–<0.35`, HIGH `>=0.35`. El valor mostrado se redondea después y nunca reclasifica el nivel.

## Phase R — R-BPBC-3

`app/personalization/` implementa el motor stateless de individualización. Solo condiciona `sysBP`, `diaBP`, `log(totChol)` y `log(glucose)`. `BMI` puede intervenir como covariable current-only en la corrección de comparabilidad, pero no se condiciona. El ajuste de logit permanece acotado a ±0.30, se conserva el orden causal por evidencia REAL y los statuses congelados.

El wire opcional `personalization` en `/predict` es estricto en sus allowlists. Versiones incompatibles o payloads de personalización inválidos degradan al resultado GLOBAL con el status/reason correspondiente; los errores del input global mantienen sus 4xx/5xx normales.

## Phase S

El scoring de forecast usa el mismo runtime Beta-0.2 de 11 variables. Los forecasts siguen siendo simulaciones y nunca se convierten en evidencia REAL para R ni en HealthRecords/Predictions clínicas.

## Entrenamiento y provenance

`scripts/train.py` es el **único entrypoint oficial vigente** de entrenamiento offline y representa exactamente `Skorp-Beta-0.2` con las 11 variables canónicas del contrato runtime. El directorio aprobado `artifacts/skorp-beta-0.2/` está congelado; el script no se ejecuta en runtime y su creación de salida falla si ese directorio ya existe, evitando sobrescritura accidental.

`scripts/train_pre_t_a.py` queda únicamente como shim de compatibilidad/auditoría hacia el mismo pipeline oficial Beta-0.2; no contiene una implementación de entrenamiento alternativa. `artifacts/skorp-beta-0.1/` se conserva como evidencia experimental histórica para trazabilidad y comparación, no como pipeline actual ni como ruta recomendada de entrenamiento.

## Ejecución

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Opcionalmente `SKORP_ARTIFACT_DIR` puede apuntar a otra copia íntegra del artefacto Beta-0.2; por defecto se usa `artifacts/skorp-beta-0.2`.

## Tests

```bash
AI_SERVICE_KEY=internal-dev-key python -m pytest -q
```

La suite cubre API/inferencia, schema estricto, contrato de 11 features, feature importance completa, soporte 32–81, Phase R, Phase S, umbrales y regresiones históricas/PRE-T-A.
