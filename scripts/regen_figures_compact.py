"""Regenera figuras clave con tamaño compacto y tipografía legible para slides."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    add_log_amount,
    clean_dataframe,
    feature_target_split,
    load_creditcard,
    random_split,
    temporal_split,
)
from src.metrics import select_threshold_by_workload
from src.modeling import build_pipeline, make_estimators, predict_proba_positive

FIGURES = ROOT / "reports" / "figures"
RESULTS = ROOT / "reports" / "results.json"
FIGURES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


def save(fig, name: str):
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("OK", path.name)


def main():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    best_r = data["random_split"]["best_model"]
    best_t = data["temporal_split"]["best_model"]

    df = add_log_amount(clean_dataframe(load_creditcard()))

    # --- EDA compactas ---
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    counts = df["Class"].value_counts().sort_index()
    bars = ax.bar(["Legítima (0)", "Fraude (1)"], counts.values, color=["#4C78A8", "#E45756"])
    ax.set_ylabel("Transacciones")
    ax.set_title("Distribución de clases (desbalance extremo)")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    save(fig, "01_class_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    sns.histplot(df.loc[df.Class == 0, "Amount"], bins=40, ax=axes[0], color="#4C78A8", log_scale=(False, True))
    axes[0].set_title("Amount — legítimas")
    sns.histplot(df.loc[df.Class == 1, "Amount"], bins=30, ax=axes[1], color="#E45756")
    axes[1].set_title("Amount — fraudes")
    save(fig, "02_amount_by_class.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    sns.kdeplot(df.loc[df.Class == 0, "Time"], ax=ax, label="Legítima", color="#4C78A8", fill=True, alpha=0.3)
    sns.kdeplot(df.loc[df.Class == 1, "Time"], ax=ax, label="Fraude", color="#E45756", fill=True, alpha=0.3)
    ax.set_title("Distribución temporal (Time) por clase")
    ax.legend(frameon=False)
    save(fig, "03_time_by_class.png")

    corr = df.corr(numeric_only=True)["Class"].drop("Class").abs().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    corr.sort_values().plot(kind="barh", ax=ax, color="#72B7B2")
    ax.set_title("|Correlación| con Class — top 10")
    ax.set_xlabel("|corr|")
    save(fig, "04_corr_with_class.png")

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.3))
    for ax, col in zip(axes, ["V14", "V17", "V12"]):
        sns.boxplot(data=df, x="Class", y=col, ax=ax, palette=["#4C78A8", "#E45756"], fliersize=1)
        ax.set_title(col)
    save(fig, "05_boxplots_by_class.png")

    # --- Reentrenar mejores modelos para curvas ---
    X, y = feature_target_split(df)
    X_tr, X_te, y_tr, y_te = random_split(X, y)
    X_tr_t, X_te_t, y_tr_t, y_te_t, _ = temporal_split(df)

    def fit_best(name, Xtr, ytr, Xte):
        est = make_estimators(ytr)[name if name != "XGBoost_SMOTE" else "XGBoost"]
        use_smote = name == "XGBoost_SMOTE"
        pipe = build_pipeline(est, list(Xtr.columns), use_smote=use_smote)
        pipe.fit(Xtr, ytr)
        return predict_proba_positive(pipe, Xte)

    # Map display names that exist in make_estimators
    def resolve(name):
        if name in make_estimators(y_tr):
            return name
        return "XGBoost"

    proba_r = fit_best(resolve(best_r), X_tr, y_tr, X_te)
    proba_t = fit_best(resolve(best_t), X_tr_t, y_tr_t, X_te_t)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    PrecisionRecallDisplay.from_predictions(y_te, proba_r, ax=ax, name=best_r)
    ax.set_title(f"Curva Precision-Recall\n{best_r} — split aleatorio")
    save(fig, "06_pr_curve_aleatorio.png")

    msel = select_threshold_by_workload(y_te, proba_r, max_alerts=113, min_precision=0.5)
    y_pred = (proba_r >= msel["threshold"]).astype(int)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ConfusionMatrixDisplay.from_predictions(y_te, y_pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusión\n{best_r} (umbral={msel['threshold']:.2f})")
    save(fig, "07_confusion_aleatorio.png")

    ks = list(range(50, 1001, 50))
    found = [int(y_te.to_numpy()[np.argsort(-proba_r)[:k]].sum()) for k in ks]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(ks, found, marker="o", ms=4, color="#E45756")
    ax.axvline(100, color="gray", ls="--", lw=1, label="K=100")
    ax.axvline(500, color="gray", ls=":", lw=1, label="K=500")
    ax.set_xlabel("Operaciones revisadas (Top-K)")
    ax.set_ylabel("Fraudes encontrados")
    ax.set_title("Captura de fraude vs esfuerzo de revisión")
    ax.legend(frameon=False)
    save(fig, "08_captura_topk_aleatorio.png")

    from src.metrics import threshold_workload_table

    wt = threshold_workload_table(y_te, proba_r)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(wt["alerts"], wt["precision"], label="Precision", color="#4C78A8")
    ax.plot(wt["alerts"], wt["recall"], label="Recall", color="#E45756")
    ax.set_xlabel("Alertas a revisar")
    ax.set_ylabel("Métrica")
    ax.set_title("Precision / Recall vs carga de alertas")
    ax.legend(frameon=False)
    save(fig, "09_workload_aleatorio.png")

    # Importancias desde results.json
    imp = data["random_split"]["models"][best_r]["feature_importance_top15"]
    if imp:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        keys = list(imp.keys())[:12][::-1]
        ax.barh(keys, [imp[k] for k in keys], color="#54A24B")
        ax.set_title(f"Importancia de variables — {best_r}")
        save(fig, "10_importance_aleatorio.png")

    # Comparación PR-AUC
    rows = []
    for split_name, key in [("Aleatorio", "random_split"), ("Temporal", "temporal_split")]:
        for mname, m in data[key]["models"].items():
            rows.append({"split": split_name, "model": mname, "pr_auc": m["metrics_thr_0.5"]["pr_auc"]})
    cdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.barplot(data=cdf, x="model", y="pr_auc", hue="split", ax=ax)
    ax.set_title("PR-AUC: split aleatorio vs temporal")
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="", frameon=False)
    save(fig, "11_comparison_pr_auc.png")

    # Temporal captura
    ks = list(range(50, 1001, 50))
    found_t = [int(y_te_t.to_numpy()[np.argsort(-proba_t)[:k]].sum()) for k in ks]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(ks, found_t, marker="o", ms=4, color="#E45756")
    ax.axvline(100, color="gray", ls="--", lw=1, label="K=100")
    ax.axvline(500, color="gray", ls=":", lw=1, label="K=500")
    ax.set_xlabel("Operaciones revisadas (Top-K)")
    ax.set_ylabel("Fraudes encontrados")
    ax.set_title(f"Captura Top-K — validación temporal ({best_t})")
    ax.legend(frameon=False)
    save(fig, "08_captura_topk_temporal.png")

    print("Figuras regeneradas.")


if __name__ == "__main__":
    main()
