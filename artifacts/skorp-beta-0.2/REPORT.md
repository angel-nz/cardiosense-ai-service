# PRE-T-A-FIX3 — CORRECT BASELINE PROVENANCE

PASS: se corrigió exclusivamente la interpretación de procedencia del test. Suite final: **65 passed, 0 failed, 0 xfailed, 0 skipped**. Modelos y evidencia numérica permanecen byte-idénticos. No se avanzó a PRE-T-B.

## Procedencias separadas

| Objeto | Estado y significado |
|---|---|
| Historical deployed Beta-0.1 | Contrato eligible_age_range=32–70, según validación previa del proyecto comunicada por el usuario. El artefacto auténtico no está en este paquete Work. **NOT EXECUTED — authentic deployed Beta-0.1 artifact not present in this Work package**. No se afirma reverificación local ni se fabrica metadata histórico. |
| 12-feature experimental baseline | Nuevo dataset 5905 × 16; 12 features incluyendo heartRate; training_age_range y eligible_age_range=32–81. Su model_version almacenado reutilizó incorrectamente Skorp-Beta-0.1 antes de PRE-T. Evidencia comparativa únicamente; no fuente de verdad del contrato histórico desplegado. |
| Skorp-Beta-0.2 | Candidato PRE-T no desplegado; exactamente 11 features sin heartRate; training_age_range y eligible_age_range=32–81; ambos rangos aceptados por _read_age_range(). |
| Current RAW_CSV | 5905 filas, edad observada 32–81; generación dinámica desde min/max reales aprobada. No se impone el rango histórico al dataset nuevo. |

Hash del dataset nuevo, verificado sin modificarlo: `533fe2625b5aace28c21714cb9ba55ec35cba11177c31ad5d7a20c79101c8e76`.

## Gates

| Gate | Resultado | Evidencia |
|---|---|---|
| Procedencia baseline12 | PASS | test_baseline12_mislabeled_beta01_is_not_historical_contract comprueba versión almacenada, hash, shape, 12 features únicas, heartRate y ambos rangos 32–81 |
| Generación del dataset actual | PASS | Expectations dinámicas min/max; basis calculado; disclaimer clínico; loader de estructura generada |
| Candidato Beta-0.2 | PASS | Test explícito de versión, once features, ausencia del campo retirado, ambos rangos 32–81 y loader |
| Suite offline | PASS | PRE-T-A: 46; training regressions: 10; metadata eligibility/provenance: 9; total 65 |
| Modelos y evidencia congelada | PASS | Seis archivos requeridos idénticos por SHA-256 antes/después |
| Baseline experimental preservado | PASS | Ocho archivos de la carpeta skorp-beta-0.1 sin cambios; esto no los convierte en artefactos históricos desplegados |
| Resto de FIX1/FIX2 | PASS | Metadata, métricas, thresholds, coeficientes, dataset y runtime sin cambios |
| Hashes de entrega | PASS | Manifest actualizado y todos los hashes contrastados con el ZIP final |
| Reverificación del artefacto histórico desplegado | NOT EXECUTED | Artefacto auténtico no disponible en este Work package; no constituye fallo del candidato |
| Objetivo ROC-AUC ≥0.80 | FAIL | Actual 0.7308460056; permanece visible, no tratado como bloqueo de PRE-T-A |
| Identidad por sujeto | LIMITATION | SUBJECT-LEVEL LEAKAGE CANNOT BE INDEPENDENTLY RULED OUT; no usado como blocker |
| Reentrenamiento / recálculo de importancia / recálculo de métricas persistidas | NOT EXECUTED | Sin ejecución de train_pre_t_a.py ni close_pre_t_a_fix1.py |
| AI runtime / backend / frontend / Prisma / PostgreSQL / Block B | NOT EXECUTED | Sin modificaciones de aplicación, sin acceso DB, sin commit |

## Cambio exacto

Único archivo de código cambiado: `tests/test_metadata_eligibility.py`.

Se sustituyó `test_beta01_historical_artifact_eligibility` por `test_baseline12_mislabeled_beta01_is_not_historical_contract`. Se retiró el assert histórico 32–70 contra el baseline incorrectamente etiquetado y se añadieron asserts de procedencia conforme a FIX3. Se mantienen los asserts del loader sobre ambos rangos. No hay skip, xfail, fixtures falsos, CSV antiguo ni filtrado del dataset.

El fuente `artifacts/skorp-beta-0.1/metadata.json` conserva exactamente sus bytes. Su copia de evidencia se entrega como `evidence/baseline12-metadata.json`; no se entrega bajo la etiqueta de artefacto histórico desplegado. Los informes FIX1/FIX2 anteriores quedan superados respecto de esta interpretación de procedencia.

Otros archivos legítimamente actualizados: `artifacts/skorp-beta-0.2/REPORT.md`, `SHA256SUMS.json`; se añaden `PRE-T-A-FIX3-REPORT.md`, logs, JUnit y evidencia de integridad. Los scripts de entrenamiento/cierre y todos los JSON numéricos existentes permanecen sin cambios.

## Ejecución de la suite

Python 3.12.14; mismas versiones fijadas en requirements-pre-t-a.txt. Las dependencias de la sesión anterior no se podían cargar y se instalaron en work/fix3-deps, sin modificar el entorno global.

```text
python -m pip install --target work/fix3-deps -r work/ai-service/requirements-pre-t-a.txt
python -m pytest work/ai-service/tests/test_pre_t_a.py work/ai-service/tests/test_training.py work/ai-service/tests/test_metadata_eligibility.py -q --junitxml=work/fix3-tests.xml
python work/package_fix3.py
```

PYTHONPATH se configuró a work/fix3-deps para pytest. Resultado observado: **65 passed in 6.23s**, salida 0. JUnit contiene 65 casos sin fallos, errores o skips; salida pytest sin xfails. Los tests de reproducibilidad/componentes ya existentes crean fits transitorios en memoria para verificar determinismo: no sustituyen, reentrenan ni guardan el modelo candidato. Las comprobaciones numéricas de la suite no alteran métricas persistidas.

Es evidencia offline; no se afirma validación del runtime desplegado o del entorno normal de Angel.

## Integridad y entrega

evidence/fix3-integrity.json contiene hashes antes/después de classifier.joblib, anomaly_model.joblib, imputer.joblib, feature_importance.json, feature_importance.csv y test_predictions.csv, así como de los ocho archivos del baseline experimental y todos los restantes archivos previos fuera del test/reporte/manifest autorizados.

SHA256SUMS.json cubre cada archivo del ZIP excepto el propio manifest, con rutas relativas a ai-service. La copia baseline12-metadata.json se contrasta byte por byte con su fuente; no se edita ni normaliza. Se verifica nuevamente la integridad de los archivos protegidos después de escribir reportes y empaquetar.

Se conservan como válidas las métricas @0.20/@0.35, sensibilidad/especificidad/PPV/NPV, comparación de calibración y anomalías 12f/11f, coeficientes, once importancias, reproducibilidad y validación de probabilidades. No se reevalúa selección, no se ajustan thresholds ni se usa TEST para tuning.


## Referencia numérica preservada

Valores de FIX1 sin recálculo en FIX3; cualquier ruta skorp-beta-0.1 en la evidencia comparativa corresponde al baseline experimental, no al artefacto desplegado.

## Dataset y modelo preservados

Dataset SHA-256 `533fe2625b5aace28c21714cb9ba55ec35cba11177c31ad5d7a20c79101c8e76`; forma 5905 × 16. TRAIN 4133 / VALIDATION 886 / TEST 886. Semillas 42.
Features exactas, en orden: male, age, currentSmoker, cigsPerDay, BPMeds, diabetes, totChol, sysBP, diaBP, BMI, glucose.
No nueva selección de modelo/features, no uso de TEST para tuning. No se ejecutó train_pre_t_a.py durante FIX1.
LogisticRegression + SMOTE, cinco estimadores base calibrados con sigmoid. Cada pipeline contiene imputación media y escalado ajustados dentro del pliegue. IsolationForest usa TRAIN con y=0 e imputador TRAIN separado.
El clasificador recibe las once columnas crudas; el imputador separado es para IsolationForest. Se conserva este contrato offline y el timestamp de entrenamiento original: 2026-10-05T00:17:58.718431+00:00.
No hay duplicados de filas ni vectores de once variables entre particiones. Sin IDs de sujetos, la fuga por sujeto no puede descartarse independientemente.

## Metadatos completados

Se añadieron excluded_features=[education, prevalentStroke, prevalentHyp], missing_values_processed_columns para las once variables, selected_hyperparameters={max_iter:2000, random_state:42}, risk_thresholds={low_max:0.20, moderate_max:0.35} y metrics (TEST canónico @0.35 más curva).
Se conservan hash, shape, target, features, rangos y basis de edad, tamaños/distribuciones, preprocesamiento, SMOTE, algoritmo, calibración, anomalías, versiones, fecha de entrenamiento y estado offline.
Faltantes: {"male": 0, "age": 0, "currentSmoker": 0, "cigsPerDay": 42, "BPMeds": 176, "diabetes": 0, "totChol": 127, "sysBP": 0, "diaBP": 0, "BMI": 31, "glucose": 629}.

## Métricas TEST por umbral

| Umbral | TP | FP | TN | FN | Sensitivity / recall | Specificity | Precision / PPV | NPV | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 102 | 195 | 530 | 59 | 0.6335403727 | 0.7310344828 | 0.3434343434 | 0.8998302207 | 0.4454148472 | 0.7133182844 |
| 0.35 | 40 | 52 | 673 | 121 | 0.2484472050 | 0.9282758621 | 0.4347826087 | 0.8476070529 | 0.3162055336 | 0.8047404063 |

Fórmulas: sensitivity=TP/(TP+FN); specificity=TN/(TN+FP); PPV=TP/(TP+FP); NPV=TN/(TN+FN). evaluate() usa 0 si el denominador es cero (convención computacional explícita; no ocurre aquí). Pruebas reconstruyen TP/FP/TN/FN con máscaras independientes y comprueban todas las métricas.
Fronteras sin redondeo: p<0.20 LOW; 0.20≤p<0.35 MODERATE; p≥0.35 HIGH. Probabilidades no finitas o fuera de [0,1] se rechazan antes de clasificar.

## Comparación descriptiva 12f vs 11f

| Métrica | Baseline 12f suministrado | Candidato 11f | Delta |
|---|---:|---:|---:|
| roc_auc | 0.726793745984 | 0.730846005569 | +0.004052259584 |
| pr_auc | 0.351497191659 | 0.354022581859 | +0.002525390200 |
| precision | 0.409638554217 | 0.434782608696 | +0.025144054479 |
| recall | 0.211180124224 | 0.248447204969 | +0.037267080745 |
| f1 | 0.278688524590 | 0.316205533597 | +0.037517009007 |
| accuracy | 0.801354401806 | 0.804740406321 | +0.003386004515 |
| brier_score | 0.135182738371 | 0.134708803488 | -0.000473934883 |
| anomaly_rate | 0.066591422122 | 0.065462753950 | -0.001128668172 |

Precision/recall/F1/accuracy de esta comparación usan 0.35. Anomaly rate corresponde a decision_function<0. No se recalcula el baseline: se usan sus métricas suministradas, conservadas en evidence/baseline12-metrics.json.
Comparación descriptiva, sin significación estadística ni afirmación de mejora clínica. No atribuir diferencias exclusivamente a retirar una variable: también cambió la colocación de la imputación en CV y el entorno de dependencias.

## Calibración comparable

Ambas curvas usan diez intervalos uniformes de probabilidad; los bins vacíos se omiten en los artefactos. Se alinean por intervalo a partir de la probabilidad media, no por posición arbitraria.

| Intervalo de p | 12f probabilidad media | 12f positivos observados | 11f probabilidad media | 11f positivos observados |
|---|---:|---:|---:|---:|
| 0.0–0.1 | 0.0676816687 | 0.0643939394 | 0.0673508620 | 0.0566037736 |
| 0.1–0.2 | 0.1465845675 | 0.1358024691 | 0.1460738450 | 0.1358024691 |
| 0.2–0.3 | 0.2453326203 | 0.2822085890 | 0.2465148298 | 0.2795031056 |
| 0.3–0.4 | 0.3452691052 | 0.3493975904 | 0.3450266658 | 0.4047619048 |
| 0.4–0.5 | 0.4376247473 | 0.5333333333 | 0.4410803668 | 0.4333333333 |
| 0.5–0.6 | 0.5465583001 | 0.3076923077 | 0.5452018369 | 0.5000000000 |
| 0.6–0.7 | 0.6404723497 | 0.6666666667 | 0.6318111266 | 0.4285714286 |
| 0.7–0.8 | 0.7358754890 | 0.3333333333 | 0.7454231322 | 0.3333333333 |
| 0.8–0.9 | vacío | vacío | vacío | vacío |
| 0.9–1.0 | vacío | vacío | vacío | vacío |

Brier 12f = 0.135182738371048; Brier 11f = 0.13470880348792705; delta = -0.00047393488312094667.
La tabla describe calibración observada en TEST; no prueba calibración poblacional ni compara significación estadística. No se inventan tamaños de bins del baseline.

## Diagnóstico secundario de coeficientes

Coeficientes de los LogisticRegression internos, extraídos de calibrated.estimator.named_steps["clf"]. Espacio estandarizado propio de cada pliegue. El modelo de predicción es el ensemble calibrado. NO son permutation importance, NO son porcentajes y NO implican causalidad.
std usa ddof=0. Consistencia = máximo conteo entre signos positivo/negativo/cero dividido por 5; se muestran conteos, no porcentajes de riesgo. El JSON conserva íntegros los cinco valores por variable.

| Feature | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std | Min | Max | Signos + / − / 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| male | 0.28830501 | 0.30039604 | 0.29865283 | 0.23553505 | 0.26790356 | 0.27815850 | 0.02424624 | 0.23553505 | 0.30039604 | 5 / 0 / 0 |
| age | 0.55872025 | 0.63091946 | 0.58679954 | 0.61153902 | 0.59915430 | 0.59742651 | 0.02422822 | 0.55872025 | 0.63091946 | 5 / 0 / 0 |
| currentSmoker | -0.05197332 | 0.03456343 | -0.01967910 | 0.03896876 | -0.00998138 | -0.00162032 | 0.03431672 | -0.05197332 | 0.03896876 | 2 / 3 / 0 |
| cigsPerDay | 0.16797367 | 0.17303086 | 0.20651446 | 0.22381123 | 0.19663391 | 0.19359283 | 0.02082548 | 0.16797367 | 0.22381123 | 5 / 0 / 0 |
| BPMeds | 0.10488221 | 0.11338306 | 0.10070483 | 0.08999853 | 0.07754096 | 0.09730192 | 0.01242158 | 0.07754096 | 0.11338306 | 5 / 0 / 0 |
| diabetes | 0.09908429 | 0.03181047 | 0.11605002 | 0.13157323 | 0.08371135 | 0.09244587 | 0.03430778 | 0.03181047 | 0.13157323 | 5 / 0 / 0 |
| totChol | 0.15428693 | 0.13908125 | 0.20526285 | 0.18902011 | 0.18452043 | 0.17443431 | 0.02416870 | 0.13908125 | 0.20526285 | 5 / 0 / 0 |
| sysBP | 0.26475820 | 0.20014128 | 0.37150945 | 0.22610160 | 0.35246875 | 0.28299586 | 0.06796409 | 0.20014128 | 0.37150945 | 5 / 0 / 0 |
| diaBP | 0.10314901 | 0.13454869 | -0.03491830 | 0.10853284 | 0.05174416 | 0.07261128 | 0.06009526 | -0.03491830 | 0.13454869 | 4 / 1 / 0 |
| BMI | 0.00526993 | -0.02062059 | -0.01258946 | 0.03624198 | -0.07245579 | -0.01283079 | 0.03564170 | -0.07245579 | 0.03624198 | 2 / 3 / 0 |
| glucose | 0.05721164 | 0.13534172 | 0.06426391 | 0.02557326 | 0.08242584 | 0.07296327 | 0.03619452 | 0.02557326 | 0.13534172 | 5 / 0 / 0 |

## Permutation importance preservada

TEST, average_precision, n_repeats=50, random_state=42. Dependencia predictiva global normalizada, no causalidad ni porcentaje del riesgo individual. Correlación entre variables puede compartir o suprimir importancia.

| Feature | Mean raw | Std raw | Porcentaje |
|---|---:|---:|---:|
| male | 0.023784374314 | 0.008580837748 | 16.37893643 |
| age | 0.078121776339 | 0.013987768335 | 53.79799324 |
| currentSmoker | -0.000091267766 | 0.000296105487 | 0.00000000 |
| cigsPerDay | 0.017444704232 | 0.006495474442 | 12.01316873 |
| BPMeds | -0.001209467987 | 0.002931680361 | 0.00000000 |
| diabetes | 0.005620580314 | 0.004225814510 | 3.87057176 |
| totChol | -0.005041395858 | 0.008744098469 | 0.00000000 |
| sysBP | 0.015067539627 | 0.012601352544 | 10.37615161 |
| diaBP | 0.002091245576 | 0.002887524688 | 1.44012106 |
| BMI | 0.000231190411 | 0.000522306771 | 0.15920760 |
| glucose | 0.002851768406 | 0.002811795301 | 1.96384958 |

Once entradas JSON y once filas CSV verificadas, iguales con tolerancia absoluta 1e-14; porcentajes suman ≈100%. Medias no positivas reciben 0%; sin valor absoluto ni redondeo de los datos almacenados.

## Estratos TEST preservados

| Estrato | N | Positivos | Tasa observada | Probabilidad media |
|---|---:|---:|---:|---:|
| LOW | 589 | 59 | 0.100169779 | 0.110655185 |
| MODERATE | 205 | 62 | 0.302439024 | 0.262134101 |
| HIGH | 92 | 40 | 0.434782609 | 0.449645839 |




PRE-T-A-FIX3 — PASS
