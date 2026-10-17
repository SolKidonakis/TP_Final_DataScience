# Detección de Fraude en Tarjetas de Crédito — Grupo 33

Proyecto Final Integrador del Diplomado en Ciencia de Datos y Análisis Avanzado.

**Integrantes:** Mariana Chaile · Eugenia Salcedo Aisa · Sol Kidonakis · Marco Garcia · Esteban Derghazarian

## Problema

Clasificar transacciones de tarjetas de crédito como **fraudulentas** o **legítimas** (dataset ULB, ~0,172 % de fraudes).

**Métricas principales:** PR-AUC, Recall y Precision.  
**Métrica de negocio:** captura en Top-100 y Top-500.  
**Validación:** split aleatorio estratificado vs validación temporal por `Time`.

## Entregables

- **Informe Final:** `Informe_Final_Grupo33.pdf`
- **Código:** `notebooks/` + `src/`

## Estructura

```
TP_Final_DataScience/
├── Informe_Final_Grupo33.pdf
├── notebooks/TP_Final_Deteccion_Fraude.ipynb
├── src/
│   ├── data.py
│   ├── metrics.py
│   ├── modeling.py
│   └── experiments.py
├── results/                 # tablas y figuras del pipeline
├── data_log.md
├── requirements.txt
└── README.md
```

## Requisitos

```bash
pip install -r requirements.txt
```

## Datos

El dataset ULB se descarga automáticamente desde el mirror de TensorFlow  
(`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`)  
y se cachea en `data/creditcard.csv` (ignorado por git). No se necesitan credenciales de Kaggle.

## Ejecución

```bash
python -m src.experiments
jupyter notebook notebooks/TP_Final_Deteccion_Fraude.ipynb
```

## Licencia de datos

Open Database License (ODbL v1.0).
