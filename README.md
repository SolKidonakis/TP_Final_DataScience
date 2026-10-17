# Detección de Fraude en Tarjetas de Crédito — Grupo 33

Proyecto Final Integrador del Diplomado en Ciencia de Datos y Análisis Avanzado.

**Integrantes:** Mariana Chaile · Eugenia Salcedo Aisa · Sol Kidonakis · Marco Garcia · Esteban Derghazarian

## Problema

Clasificar transacciones de tarjetas de crédito como **fraudulentas** o **legítimas** sobre el dataset ULB (Kaggle / OpenML), con desbalance extremo (~0,172 % de fraudes).

**Métricas principales:** PR-AUC, Recall y Precision.  
**Métrica de negocio:** captura en Top-K (¿cuántos fraudes encontramos si revisamos las 100 o 500 operaciones de mayor riesgo?).  
**Validación:** comparación entre split aleatorio estratificado y validación temporal por `Time`.

## Estructura

```
TP_Final_DataScience/
├── notebooks/TP_Final_Deteccion_Fraude.ipynb   # Notebook principal (Colab / local)
├── src/
│   ├── data.py          # Carga OpenML + splits
│   ├── metrics.py       # PR-AUC, captura@K, carga de alertas
│   ├── modeling.py      # Pipelines y modelos
│   └── run_pipeline.py  # Ejecución batch reproducible
├── data/                # Cache local del CSV (ignorado por git)
├── reports/
│   ├── figures/         # Gráficos generados
│   ├── results.json     # Métricas serializadas
│   ├── Informe_Final_Grupo33.pdf
│   └── Presentacion_Final_Grupo33.pdf
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.10+
- Dependencias en `requirements.txt`

```bash
pip install -r requirements.txt
```

## Datos

El dataset ULB se descarga automáticamente desde el **mirror de TensorFlow**  
(`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`)  
y se cachea en `data/creditcard.csv`. Fallback: OpenML `data_id=1597`.  
No se requieren credenciales de Kaggle.

## Ejecución local

Pipeline completo (EDA + modelos + figuras + `results.json`):

```bash
python -m src.run_pipeline
```

Notebook interactivo:

```bash
jupyter notebook notebooks/TP_Final_Deteccion_Fraude.ipynb
```

## Ejecución en Google Colab

1. Subir el repositorio o clonarlo.
2. Instalar dependencias: `!pip install -r requirements.txt`
3. Abrir y ejecutar `notebooks/TP_Final_Deteccion_Fraude.ipynb` de arriba hacia abajo.

## Modelos evaluados

| Experimento | Modelo | Manejo del desbalance |
|-------------|--------|------------------------|
| Baseline | Regresión Logística | `class_weight='balanced'` |
| Exp. 2 | Random Forest | `class_weight` |
| Exp. 3 | XGBoost | `scale_pos_weight` |
| Exp. 4 | LightGBM | `is_unbalance` |
| Exp. 5 | XGBoost + SMOTE | SMOTE dentro del pipeline |

## Resultados principales (reproducibles)

**Mejor modelo (split aleatorio):** XGBoost  
- PR-AUC ≈ 0.827 · Recall ≈ 81 % · Precision ≈ 87 % (umbral 0,5)  
- Top-100 → 77 fraudes · Top-500 → 82 fraudes  
- Umbral operativo sugerido 0,55 → 88 alertas (77 TP / 11 FP)

**Mejor modelo (split temporal):** Random Forest  
- PR-AUC ≈ 0.813 · Recall ≈ 73 % · Precision ≈ 95 %  
- Top-100 → 59 fraudes · Top-500 → 65 fraudes

Regenerar métricas y figuras:

```bash
python -m src.run_pipeline
python scripts/build_informe.py
python scripts/build_presentacion.py
```

## Entregables

- Informe Final: `reports/Informe_Final_Grupo33.pdf` (~9–10 páginas)
- Presentación: `reports/Presentacion_Final_Grupo33.pdf` (12 slides) + `.pptx`
- Código: este repositorio (`notebooks/` + `src/`)

## Licencia de datos

Open Database License (ODbL v1.0) — uso libre para investigación.
