"""Genera el notebook principal del TP Final."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "TP_Final_Deteccion_Fraude.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Detección de Fraude en Tarjetas de Crédito — Grupo 33

**Diplomado en Ciencia de Datos y Análisis Avanzado — Proyecto Final Integrador**

**Integrantes:** Mariana Chaile · Eugenia Salcedo Aisa · Sol Kidonakis · Marco Garcia · Esteban Derghazarian

## Objetivo
Construir un modelo de clasificación supervisada que distinga transacciones **fraudulentas** de **legítimas**, maximizando la detección de fraude (Recall) sin saturar al equipo con alertas incorrectas (Precision y carga operativa).

## Metodología
CRISP-DM: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment (recomendaciones).

## Devolución del tutor (incorporada)
1. Reportar **PR-AUC, Recall y Precision** como métricas principales; cuantificar **cuántas operaciones tendría que revisar el equipo**.
2. Métrica de negocio: captura en **Top-100 / Top-500**.
3. Comparar **split aleatorio** vs **validación temporal** por `Time`.
""")

md("## 0. Setup")

code("""# Si se ejecuta en Colab, descomentar:
# !pip install -q xgboost lightgbm imbalanced-learn

from pathlib import Path
import sys
import warnings
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

# Raíz del repo (local o Colab)
CANDIDATES = [Path.cwd(), Path.cwd().parent, Path("/content/TP_Final_DataScience")]
ROOT = next((p for p in CANDIDATES if (p / "src").exists()), Path.cwd().parent)
sys.path.insert(0, str(ROOT))

from src.data import load_creditcard, clean_dataframe, add_log_amount, feature_target_split, random_split, temporal_split
from src.metrics import classification_metrics, captura_table, threshold_workload_table, select_threshold_by_workload
from src.modeling import make_estimators, build_pipeline, predict_proba_positive, get_feature_importance

FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
print("ROOT:", ROOT)
""")

md("""## 1. Business Understanding

- **Problema:** el fraude en tarjetas genera pérdidas millonarias y daño reputacional.
- **Audiencia:** Prevención de Fraude / Gestión de Riesgo.
- **KPI técnicos principales:** PR-AUC, Recall, Precision.
- **KPI de negocio:** fraudes capturados al revisar las Top-100 y Top-500 operaciones de mayor score.
- **Hipótesis:** patrones en monto, tiempo y componentes PCA permiten superar reglas estáticas.
""")

md("## 2. Data Understanding — carga del dataset")

code("""df_raw = load_creditcard()
print(df_raw.shape)
print(df_raw["Class"].value_counts())
print(f"Tasa de fraude: {df_raw['Class'].mean():.4%}")
df_raw.head()
""")

code("""df_raw.info()
display(df_raw.describe().T.head(15))
print("Nulos totales:", int(df_raw.isna().sum().sum()))
""")

md("""### Diccionario mínimo
| Variable | Descripción |
|----------|-------------|
| Time | Segundos desde la primera transacción |
| V1–V28 | Componentes PCA (anonimizados) |
| Amount | Monto en euros |
| Class | 1 = fraude, 0 = legítima |
""")

md("## 3. EDA")

code("""df = clean_dataframe(df_raw)
print("Duplicados eliminados:", df.attrs.get("n_duplicates_removed", 0))
df = add_log_amount(df)

fig, ax = plt.subplots(figsize=(7, 4))
counts = df["Class"].value_counts().sort_index()
ax.bar(["Legítima (0)", "Fraude (1)"], counts.values, color=["#4C78A8", "#E45756"])
ax.set_title("Distribución de clases")
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(FIGURES / "01_class_distribution.png", dpi=150)
plt.show()
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df.loc[df.Class == 0, "Amount"], bins=50, ax=axes[0], log_scale=(False, True), color="#4C78A8")
axes[0].set_title("Amount — legítimas (Y log)")
sns.histplot(df.loc[df.Class == 1, "Amount"], bins=40, ax=axes[1], color="#E45756")
axes[1].set_title("Amount — fraudes")
plt.tight_layout()
plt.savefig(FIGURES / "02_amount_by_class.png", dpi=150)
plt.show()
""")

code("""fig, ax = plt.subplots(figsize=(10, 4))
sns.kdeplot(df.loc[df.Class == 0, "Time"], label="Legítima", fill=True, alpha=0.3, ax=ax)
sns.kdeplot(df.loc[df.Class == 1, "Time"], label="Fraude", fill=True, alpha=0.3, ax=ax, color="#E45756")
ax.set_title("Distribución temporal por clase")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "03_time_by_class.png", dpi=150)
plt.show()
""")

code("""corr = df.corr(numeric_only=True)["Class"].drop("Class").abs().sort_values(ascending=False).head(12)
fig, ax = plt.subplots(figsize=(8, 5))
corr.sort_values().plot(kind="barh", ax=ax, color="#72B7B2")
ax.set_title("|Correlación| con Class — top 12")
plt.tight_layout()
plt.savefig(FIGURES / "04_corr_with_class.png", dpi=150)
plt.show()
corr
""")

code("""fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, col in zip(axes, ["V14", "V17", "V12", "Amount"]):
    sns.boxplot(data=df, x="Class", y=col, ax=ax, palette=["#4C78A8", "#E45756"])
    ax.set_title(col)
plt.tight_layout()
plt.savefig(FIGURES / "05_boxplots_by_class.png", dpi=150)
plt.show()
""")

md("""### Conclusiones del EDA
- Desbalance extremo (~0,17 % fraudes): Accuracy y ROC-AUC solos son engañosos → priorizar **PR-AUC**.
- `Amount` es sesgado; se usa `log1p(Amount)`.
- Variables PCA (p. ej. V14, V17, V12) muestran separación por clase.
- `Time` habilita una validación temporal realista (entrenar con pasado, testear futuro).
""")

md("## 4. Data Preparation")

code("""# Features: V1..V28 + Time + Amount_log (Amount crudo se descarta para evitar doble conteo)
X, y = feature_target_split(df, use_log_amount=True)
print(X.columns.tolist())
print(X.shape, y.value_counts().to_dict())
""")

md("""Preprocesamiento **dentro del Pipeline** (evita data leakage):
- `StandardScaler` sobre `Time` y `Amount_log`
- V1–V28 pasan sin cambios (ya escaladas por PCA)
- SMOTE solo dentro del pipeline del experimento correspondiente
""")

md("## 5. Validación — Split aleatorio vs temporal")

code("""# (A) Aleatorio estratificado
X_tr, X_te, y_tr, y_te = random_split(X, y, test_size=0.2, random_state=42)
print("Aleatorio — train fraud:", int(y_tr.sum()), "| test fraud:", int(y_te.sum()))

# (B) Temporal por Time
X_tr_t, X_te_t, y_tr_t, y_te_t, tmeta = temporal_split(df, test_size=0.2)
print("Temporal meta:", tmeta)
""")

md("## 6. Modelado")

code("""def train_and_eval(split_name, X_train, X_test, y_train, y_test):
    feature_names = list(X_train.columns)
    estimators = make_estimators(y_train)
    specs = [(n, e, False) for n, e in estimators.items()]
    specs.append(("XGBoost_SMOTE", make_estimators(y_train)["XGBoost"], True))

    rows = []
    artifacts = {}
    for name, est, smote in specs:
        pipe = build_pipeline(est, feature_names, use_smote=smote)
        pipe.fit(X_train, y_train)
        proba = predict_proba_positive(pipe, X_test)
        m50 = classification_metrics(y_test, proba, 0.5)
        max_alerts = min(500, max(100, int(0.002 * len(y_test))))
        msel = select_threshold_by_workload(y_test, proba, max_alerts=max_alerts, min_precision=0.5)
        cap = captura_table(y_test, proba, ks=(100, 500))
        rows.append({
            "split": split_name,
            "model": name,
            "PR_AUC": m50["pr_auc"],
            "Recall_0.5": m50["recall"],
            "Precision_0.5": m50["precision"],
            "F1_0.5": m50["f1"],
            "ROC_AUC": m50["roc_auc"],
            "Alerts_0.5": m50["alerts"],
            "thr_sel": msel["threshold"],
            "Recall_sel": msel["recall"],
            "Precision_sel": msel["precision"],
            "Alerts_sel": msel["alerts"],
            "Fraud@100": cap.loc[cap.k == 100, "frauds_found"].iloc[0],
            "Fraud@500": cap.loc[cap.k == 500, "frauds_found"].iloc[0],
        })
        artifacts[name] = {"pipe": pipe, "proba": proba, "m50": m50, "msel": msel, "cap": cap}
        print(f"{split_name:10s} | {name:16s} | PR-AUC={m50['pr_auc']:.4f} | R={m50['recall']:.3f} | P={m50['precision']:.3f}")
    return pd.DataFrame(rows), artifacts

df_rand, art_rand = train_and_eval("aleatorio", X_tr, X_te, y_tr, y_te)
df_temp, art_temp = train_and_eval("temporal", X_tr_t, X_te_t, y_tr_t, y_te_t)
results = pd.concat([df_rand, df_temp], ignore_index=True)
results.sort_values(["split", "PR_AUC"], ascending=[True, False])
""")

md("## 7. Evaluación — métricas principales y carga operativa")

code("""# Comparación PR-AUC
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=results, x="model", y="PR_AUC", hue="split", ax=ax)
ax.set_title("PR-AUC: split aleatorio vs temporal")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(FIGURES / "11_comparison_pr_auc.png", dpi=150)
plt.show()
""")

code("""# Mejor modelo en split aleatorio
best_name = df_rand.sort_values("PR_AUC", ascending=False).iloc[0]["model"]
print("Mejor modelo (aleatorio):", best_name)
proba = art_rand[best_name]["proba"]
msel = art_rand[best_name]["msel"]
print("Umbral seleccionado:", msel)

from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(6, 5))
PrecisionRecallDisplay.from_predictions(y_te, proba, ax=ax, name=best_name)
ax.set_title("Curva Precision-Recall (holdout aleatorio)")
plt.tight_layout()
plt.savefig(FIGURES / "06_pr_curve_aleatorio.png", dpi=150)
plt.show()

y_pred = (proba >= msel["threshold"]).astype(int)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_te, y_pred, ax=ax, cmap="Blues")
ax.set_title(f"Matriz de confusión (thr={msel['threshold']:.2f})")
plt.tight_layout()
plt.savefig(FIGURES / "07_confusion_aleatorio.png", dpi=150)
plt.show()
""")

md("## 8. Métrica de negocio — Captura Top-K")

code("""cap = art_rand[best_name]["cap"]
display(cap)

ks = list(range(50, 1001, 50))
found = [int(y_te.to_numpy()[np.argsort(-proba)[:k]].sum()) for k in ks]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ks, found, marker="o", color="#E45756")
ax.axvline(100, ls="--", color="gray", label="K=100")
ax.axvline(500, ls=":", color="gray", label="K=500")
ax.set_xlabel("Operaciones revisadas (Top-K)")
ax.set_ylabel("Fraudes encontrados")
ax.set_title("Si revisamos las K operaciones de mayor riesgo…")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "08_captura_topk_aleatorio.png", dpi=150)
plt.show()
""")

code("""wt = threshold_workload_table(y_te, proba)
display(wt.head(10))
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(wt["alerts"], wt["precision"], label="Precision")
ax.plot(wt["alerts"], wt["recall"], label="Recall")
ax.set_xlabel("Alertas a revisar")
ax.set_title("Precision / Recall vs carga operativa del equipo")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "09_workload_aleatorio.png", dpi=150)
plt.show()
""")

md("""### Interpretación de negocio
Con pocos fraudes reales, un FPR del 5 % sobre ~57.000 legítimas del holdout generaría **miles de alertas falsas**.  
Por eso el umbral se elige mirando **Precision** y el **número absoluto de alertas**, no solo FPR.
""")

md("## 9. Split temporal — robustez")

code("""best_t = df_temp.sort_values("PR_AUC", ascending=False).iloc[0]["model"]
print("Mejor modelo (temporal):", best_t)
display(df_temp.sort_values("PR_AUC", ascending=False)[["model", "PR_AUC", "Recall_0.5", "Precision_0.5", "Fraud@100", "Fraud@500"]])
print("Meta temporal:", tmeta)
""")

md("## 10. Importancia de variables y modelo final")

code("""imp = get_feature_importance(art_rand[best_name]["pipe"], list(X_tr.columns))
imp15 = dict(list(imp.items())[:15])
fig, ax = plt.subplots(figsize=(8, 5))
keys = list(imp15.keys())[::-1]
ax.barh(keys, [imp15[k] for k in keys], color="#54A24B")
ax.set_title(f"Importancia de variables — {best_name}")
plt.tight_layout()
plt.savefig(FIGURES / "10_importance_aleatorio.png", dpi=150)
plt.show()
""")

md("""## 11. Conclusiones y recomendaciones

1. El modelo seleccionado maximiza **PR-AUC** y ofrece un buen equilibrio Recall/Precision con un volumen de alertas operable.
2. La métrica **Top-K** traduce el score a una cola de revisión accionable para el equipo de fraude.
3. La validación **temporal** es más exigente y realista que el split aleatorio; se recomienda usarla para decidir el despliegue.
4. Limitaciones: dataset de 2 días (2013), variables PCA poco interpretables ante auditoría, sin datos en tiempo real.
5. Próximos pasos: monitoreo de drift, feedback del analista, calibración de scores y enriquecimiento con señales de dispositivo/geolocalización (si hay datos).
""")

md("""## 12. Consideraciones éticas y de privacidad
- Los datos están anonimizados vía PCA; no se manipulan PII.
- Un falso positivo puede bloquear a un cliente legítimo → costo de experiencia.
- Un falso negativo implica pérdida financiera → trade-off explícito de umbral.
- Uso académico / investigación bajo licencia ODbL.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print("Notebook escrito en", OUT)
