# Data log — Detección de Fraude (Grupo 33)

Registro de decisiones del proyecto, en orden cronológico. Cada entrada indica qué se decidió, por qué y dónde se ve el efecto.

| Fecha | Decisión | Motivo | Evidencia |
|-------|----------|--------|-----------|
| 2026-09-21 | Dataset ULB `creditcard.csv` (284.807 filas, 492 fraudes) | Datos reales, licencia ODbL, suficiente volumen | Pre-entrega, sección 3 |
| 2026-09-21 | Métricas principales PR-AUC, Recall y Precision | Desbalance 0,17 %: accuracy y ROC-AUC son engañosas | Pre-entrega, sección 4.3 |
| 2026-10-17 | Se abandona la restricción "FPR ≤ 5 %" como criterio | Devolución del tutor: con 56.651 legítimas en test, FPR 5 % implica ~2.800 alertas falsas | Informe §4.3; notebook §8 |
| 2026-10-17 | Se agrega la métrica de negocio Top-K (K = 100 y 500) | Devolución del tutor: traduce el score a una cola de revisión | Informe §4.2; notebook §8 |
| 2026-10-17 | Se agrega un test temporal (80 % inicial → 20 % final por `Time`) | Devolución del tutor; la prevalencia varía entre 0,08 % y 0,33 % según el tramo temporal | Informe §4.4; notebook §9 |
| 2026-10-17 | Se eliminan 1.081 duplicados exactos (19 eran fraudes) | Duplicados inflan métricas si caen en train y test a la vez | `src/data.py`; notebook §3 |
| 2026-10-17 | `Amount` → `Amount_log = log1p(Amount)` y escalado dentro del Pipeline | Asimetría fuerte del monto; escalar dentro del Pipeline evita fuga de información | `src/data.py`, `src/modeling.py` |
| 2026-10-17 | `Time` crudo se reemplaza por hora del día cíclica (`Hour_sin`, `Hour_cos`) | En el test temporal, `Time` toma valores nunca vistos en entrenamiento; la hora sí se repite y la EDA muestra más fraude de madrugada | Ablación "Time crudo en lugar de hora" |
| 2026-10-17 | Desbalance: `class_weight` / `scale_pos_weight` en lugar de submuestreo | Conserva todos los datos; SMOTE se evalúa como variante dentro del Pipeline | `src/modeling.py`; notebook §6 |
| 2026-10-17 | LightGBM: `is_unbalance=True` reemplazado por peso √(neg/pos) y regularización | La configuración original era inestable (PR-AUC ≈ 0) | Prueba exploratoria registrada en el informe, sección 4 |
| 2026-10-17 | Regla de selección fijada antes de mirar los tests | Evita elegir el modelo con el test | `src/experiments.py` → `select_model` |
| 2026-10-17 | Se completa el desempate final de la regla con la PR-AUC media de CV | Random Forest y XGBoost empataban exactamente en Precision@100 de CV (0,644) y la primera versión elegía por el orden de la lista. El criterio agregado usa solo la CV | `src/experiments.py` → `select_model` |
| 2026-10-17 | Umbral operativo = máximo F1 sobre predicciones out-of-fold del desarrollo | Riesgo identificado en la pre-entrega: optimizar el umbral sobre el test sobreajusta | `src/experiments.py`; notebook §7 |
| 2026-10-17 | Escenario económico con costo de revisión de €5 por alerta | Supuesto paramétrico explícito; no hay costos reales del emisor | Informe §6 |
