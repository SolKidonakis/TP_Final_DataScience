# Data log — Detección de Fraude (Grupo 33)

Registro de decisiones del proyecto, en orden cronológico. Cada entrada indica qué se decidió, por qué y dónde se ve el efecto.

| Fecha | Decisión | Motivo | Evidencia |
|-------|----------|--------|-----------|
| 2026-09-21 | Dataset ULB `creditcard.csv` (284.807 filas, 492 fraudes) | Datos reales, licencia ODbL, suficiente volumen | Pre-entrega, sección 3 |
| 2026-09-21 | Métricas principales PR-AUC, Recall y Precision | Desbalance 0,17 %: accuracy y ROC-AUC son engañosas | Pre-entrega, sección 4.3 |
| 2026-10-17 | Se abandona la restricción "FPR ≤ 5 %" como criterio | Devolución del tutor: con 56.651 legítimas en test, FPR 5 % implica ~2.800 alertas falsas | Informe §4.3; notebook §6 |
| 2026-10-17 | Se agrega la métrica de negocio Top-K (K = 100 y 500) | Devolución del tutor: traduce el score a una cola de revisión | Informe §4.2; notebook §4 |
| 2026-10-17 | Se agrega un test temporal (80 % inicial → 20 % final por `Time`) | Devolución del tutor; la prevalencia varía entre 0,08 % y 0,33 % según el tramo temporal | Informe §4.4; notebook §5 |
| 2026-10-17 | Se eliminan 1.081 duplicados exactos (19 eran fraudes) | Duplicados inflan métricas si caen en train y test a la vez | Notebook §1 |
| 2026-10-17 | `Amount` → `Amount_log = log1p(Amount)` y escalado dentro del Pipeline | Asimetría fuerte del monto; escalar dentro del Pipeline evita fuga de información | Notebook §0 (`add_features`, `build_pipeline`) |
| 2026-10-17 | `Time` crudo se reemplaza por hora del día cíclica (`Hour_sin`, `Hour_cos`) | En el test temporal, `Time` toma valores nunca vistos en entrenamiento; la hora sí se repite y la EDA muestra más fraude de madrugada | Notebook §9, ablación "Time crudo en lugar de hora" |
| 2026-10-17 | Desbalance: `class_weight` / `scale_pos_weight` en lugar de submuestreo | Conserva todos los datos; SMOTE se evalúa como variante dentro del Pipeline | Notebook §0 (`make_estimator`) y §3 |
| 2026-10-17 | LightGBM: `is_unbalance=True` reemplazado por peso √(neg/pos) y regularización | La configuración original era inestable (PR-AUC ≈ 0) | Notebook §0 (`make_estimator`), comentario en LightGBM |
| 2026-10-17 | Regla de selección fijada antes de mirar los tests | Evita elegir el modelo con el test | Notebook §3 (`select_model`) |
| 2026-10-17 | Se completa el desempate final de la regla con la PR-AUC media de CV | Random Forest y XGBoost empataban exactamente en Precision@100 de CV (0,644) y la primera versión elegía por el orden de la lista. El criterio agregado usa solo la CV | Notebook §3 (`select_model`) |
| 2026-10-17 | Umbral operativo = máximo F1 sobre predicciones out-of-fold del desarrollo | Riesgo identificado en la pre-entrega: optimizar el umbral sobre el test sobreajusta | Notebook §3 (`best_f1_threshold`) y §6 |
| 2026-10-17 | Escenario económico con costo de revisión de €5 por alerta | Supuesto paramétrico explícito; no hay costos reales del emisor | Notebook §7; informe §6 |
| 2026-10-23 | El notebook vuelve a incluir CV de selección, umbral out-of-fold, ablaciones, auditoría por segmentos, escenario económico y SHAP, y regenera `results/` completo | El notebook simplificado elegía el modelo por PR-AUC del test y no reproducía las tablas del informe; se contradecía con la regla de selección de la pre-entrega | Notebook §3–§10 |
| 2026-10-23 | SMOTE con `sampling_strategy=0.1` (antes proporción 1:1 en la corrida de `results/` previa) | Remuestreo menos agresivo; al reejecutar todo con una sola versión del código, `results/` deja de mezclar corridas | Notebook §0 (`build_pipeline`) |
| 2026-10-23 | Al reejecutar, la regla de selección elige Random Forest (antes XGBoost): empata con XGBoost en Precision@100 de CV (64,4 % vs 64,2 %, un fraude de diferencia) | La regla se mantiene sin cambios; el informe aclara que los cuatro modelos de árboles son indistinguibles dentro del ruido | Notebook §3; informe §4.1 |
| 2026-10-23 | Informe regenerado desde `results/results.json` con `scripts/build_informe.py`; se agregan escenario económico, regla de monto como baseline, glosario, tabla de trazabilidad con la pre-entrega y se reportan H2 (no se sostiene) y H3 (no se confirma con claridad) | El PDF anterior tenía cifras de una versión previa del código (p. ej. LightGBM con PR-AUC 0,048, umbral 0,55) | Informe completo |
