# Detección de Fraude en Tarjetas de Crédito — Grupo 33

Proyecto Final Integrador del Diplomado en Ciencia de Datos y Análisis Avanzado.

**Integrantes:** Mariana Chaile · Eugenia Salcedo Aisa · Sol Kidonakis · Marco Garcia · Esteban Derghazarian

## Entregables

- Informe: [`docs/Informe_Final_Grupo33.pdf`](docs/Informe_Final_Grupo33.pdf) 
- Código: [`notebooks/TP_Final_Deteccion_Fraude.ipynb`](notebooks/TP_Final_Deteccion_Fraude.ipynb) 

## Estructura

```
TP_Final_DataScience/
├── docs/Informe_Final_Grupo33.pdf
├── notebooks/TP_Final_Deteccion_Fraude.ipynb
├── results/                 # figuras y tablas que genera el notebook
├── data_log.md
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
jupyter notebook notebooks/TP_Final_Deteccion_Fraude.ipynb
```

El dataset ULB se descarga solo (mirror de TensorFlow) y se cachea en `data/` (no va a git).
Al correr el notebook se actualizan `results/figures` y `results/tables`.

## Licencia de datos

Open Database License (ODbL v1.0).
