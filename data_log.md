# Data log — Detección de Fraude (Grupo 33)

Registro de decisiones del proyecto, en orden cronológico. Cada entrada indica qué se decidió, por qué y dónde se ve el efecto.

| Fecha | Decisión | Motivo | Evidencia |
|-------|----------|--------|-----------|
| 2026-09-21 | Dataset ULB `creditcard.csv` (284.807 filas, 492 fraudes) | Datos reales, licencia ODbL, suficiente volumen | Pre-entrega, sección 3 |
| 2026-09-21 | Métricas principales PR-AUC, Recall y Precision | Desbalance 0,17 %: accuracy y ROC-AUC son engañosas | Pre-entrega, sección 4.3 |
| 2026-10-17 | Se abandona la restricción "FPR ≤ 5 %" como criterio | Devolución del tutor: con 56.651 legítimas en test, FPR 5 % implica ~2.800 alertas falsas | Informe §4.3; notebook §5 |
| 2026-10-17 | Se agrega la métrica de negocio Top-K (K = 100 y 500) | Devolución del tutor: traduce el score a una cola de revisión | Informe §4.2; notebook §5 |
| 2026-10-17 | Se agrega un test temporal (80 % inicial → 20 % final por `Time`) | Devolución del tutor; la prevalencia varía según el tramo temporal | Informe §4.4; notebook §6 |
| 2026-10-17 | Se eliminan 1.081 duplicados exactos (19 eran fraudes) | Duplicados inflan métricas si caen en train y test a la vez | notebook §1 |
| 2026-10-17 | `Amount` → `Amount_log = log1p(Amount)` y escalado dentro del Pipeline | Asimetría fuerte del monto; escalar dentro del Pipeline evita fuga de información | notebook §0 |
| 2026-10-17 | `Time` crudo se reemplaza por hora del día cíclica (`Hour_sin`, `Hour_cos`) | En el test temporal, `Time` toma valores nunca vistos en entrenamiento; la hora sí se repite | notebook §0; informe §3 |
| 2026-10-17 | Desbalance: `class_weight` / `scale_pos_weight`; SMOTE solo como variante (strategy=0.1) | Conserva todos los datos; SMOTE `auto` inflaba el train y caía el kernel | notebook §0 |
| 2026-10-17 | LightGBM: `is_unbalance=True` reemplazado por peso √(neg/pos) y regularización | La configuración original era inestable (PR-AUC ≈ 0,048 en el informe) | notebook §0 vs informe Tabla 5 |
| 2026-10-24 | CV estratificada 5-fold sobre el train; gana el mayor PR-AUC medio | Evita elegir el modelo mirando el holdout | notebook §3 |
| 2026-10-24 | Umbral operativo = máximo F1 sobre predicciones out-of-fold del train | Optimizar el umbral sobre el test sobreajusta | notebook §5 |
| 2026-10-24 | Importancias leídas con `get_feature_names_out()` | El ColumnTransformer pone `Amount_log` primero; usar `X.columns` etiquetaba mal el gráfico | notebook §7 |
