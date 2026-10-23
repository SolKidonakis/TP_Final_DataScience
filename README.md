# Detección de Fraude en Tarjetas de Crédito — Grupo 33

Proyecto Final Integrador del Diplomado en Ciencia de Datos y Análisis Avanzado.

**Integrantes:** Mariana Chaile · Eugenia Salcedo Aisa · Sol Kidonakis · Marco Garcia · Esteban Derghazarian

## Entregables

- Informe: [`docs/Informe_Final_Grupo33.pdf`](docs/Informe_Final_Grupo33.pdf)
- Código: [`notebooks/TP_Final_Deteccion_Fraude.ipynb`](notebooks/TP_Final_Deteccion_Fraude.ipynb) (autocontenido)
  — [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SolKidonakis/TP_Final_DataScience/blob/main/notebooks/TP_Final_Deteccion_Fraude.ipynb)

## Estructura

```
TP_Final_DataScience/
├── docs/                    # consigna, pre-entrega e informe final (PDF)
├── notebooks/TP_Final_Deteccion_Fraude.ipynb
├── results/                 # figuras, tablas y results.json que genera el notebook
├── scripts/build_informe.py # arma el informe PDF a partir de results/
├── data_log.md              # registro de decisiones
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

**En Colab:** abrir el notebook con el botón de arriba y ejecutar todo (Entorno de ejecución → Ejecutar todas). La primera celda clona el repo e instala lo que falta.

**En local:**

```bash
pip install -r requirements.txt
jupyter notebook notebooks/TP_Final_Deteccion_Fraude.ipynb
```

El dataset ULB se descarga solo (mirror de TensorFlow) y se cachea en `data/` (no va a git).
Ejecutarlo de arriba hacia abajo (≈ 5–10 min) regenera `results/figures`, `results/tables` y `results/results.json`.

Para rearmar el informe con esas cifras:

```bash
python scripts/build_informe.py
```

## Metodología en una línea

Se elige el modelo con validación cruzada y el umbral con predicciones out-of-fold, ambos sobre el 80 % de desarrollo; el 20 % restante (holdout aleatorio) y el test temporal solo se usan para evaluar. Detalle de las decisiones en [`data_log.md`](data_log.md).

## Licencia de datos

Open Database License (ODbL v1.0).
