"""Ejecuta el pipeline completo: EDA, modelado, métricas y figuras.

Uso (desde la raíz del repo):
    python -m src.run_pipeline
"""

from __future__ import annotations

import json
import sys
import time
import warnings
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
from src.metrics import (  # noqa: E402
    captura_table,
    classification_metrics,
    select_threshold_by_workload,
    threshold_workload_table,
)
from src.modeling import (  # noqa: E402
    build_pipeline,
    get_feature_importance,
    make_estimators,
    predict_proba_positive,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")

FIGURES = ROOT / "reports" / "figures"
RESULTS = ROOT / "reports" / "results.json"
FIGURES.mkdir(parents=True, exist_ok=True)


def savefig(name: str):
    path = FIGURES / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def run_eda(df: pd.DataFrame) -> dict:
    summary = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_fraud": int(df["Class"].sum()),
        "fraud_rate": float(df["Class"].mean()),
        "n_nulls": int(df.isna().sum().sum()),
        "amount_mean": float(df["Amount"].mean()),
        "amount_median": float(df["Amount"].median()),
        "time_min": float(df["Time"].min()),
        "time_max": float(df["Time"].max()),
    }

    # Clases
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["Class"].value_counts().sort_index()
    bars = ax.bar(["Legítima (0)", "Fraude (1)"], counts.values, color=["#4C78A8", "#E45756"])
    ax.set_ylabel("Cantidad de transacciones")
    ax.set_title("Distribución de clases (desbalance extremo)")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=11)
    savefig("01_class_distribution.png")

    # Amount
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df.loc[df["Class"] == 0, "Amount"], bins=50, ax=axes[0], color="#4C78A8", log_scale=(False, True))
    axes[0].set_title("Amount — legítimas (eje Y log)")
    sns.histplot(df.loc[df["Class"] == 1, "Amount"], bins=40, ax=axes[1], color="#E45756")
    axes[1].set_title("Amount — fraudes")
    savefig("02_amount_by_class.png")

    # Time
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.kdeplot(df.loc[df["Class"] == 0, "Time"], ax=ax, label="Legítima", color="#4C78A8", fill=True, alpha=0.3)
    sns.kdeplot(df.loc[df["Class"] == 1, "Time"], ax=ax, label="Fraude", color="#E45756", fill=True, alpha=0.3)
    ax.set_title("Distribución temporal (Time) por clase")
    ax.legend()
    savefig("03_time_by_class.png")

    # Correlación con Class (top)
    corr = df.corr(numeric_only=True)["Class"].drop("Class").abs().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    corr.sort_values().plot(kind="barh", ax=ax, color="#72B7B2")
    ax.set_title("|Correlación| con Class — top 12 variables")
    ax.set_xlabel("|corr|")
    savefig("04_corr_with_class.png")

    # Boxplots V14, V17, V12, Amount
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, col in zip(axes, ["V14", "V17", "V12", "Amount"]):
        sns.boxplot(data=df, x="Class", y=col, ax=ax, palette=["#4C78A8", "#E45756"])
        ax.set_title(col)
    savefig("05_boxplots_by_class.png")

    summary["top_corr"] = {k: float(v) for k, v in corr.items()}
    return summary


def evaluate_split(name: str, X_train, X_test, y_train, y_test, meta: dict | None = None) -> dict:
    feature_names = list(X_train.columns)
    estimators = make_estimators(y_train)
    # Agregar variante SMOTE con el mejor candidato base (XGBoost) — se evalúa también LR+SMOTE
    results = {"split": name, "meta": meta or {}, "models": {}}

    model_specs = []
    for mname, est in estimators.items():
        model_specs.append((mname, est, False))
    # SMOTE + XGBoost (Exp. 5 del plan)
    model_specs.append(("XGBoost_SMOTE", make_estimators(y_train)["XGBoost"], True))

    for mname, est, use_smote in model_specs:
        t0 = time.time()
        pipe = build_pipeline(est, feature_names, use_smote=use_smote)
        pipe.fit(X_train, y_train)
        proba = predict_proba_positive(pipe, X_test)
        elapsed = time.time() - t0

        # Métricas a umbral 0.5 y umbral seleccionado por carga
        m50 = classification_metrics(y_test, proba, threshold=0.5)
        # max_alerts: ~ equivalente a revisar un volumen manejable del test (~0.5% o 500, lo que sea menor razonable)
        max_alerts = min(500, max(100, int(0.002 * len(y_test))))
        m_sel = select_threshold_by_workload(
            y_test, proba, max_alerts=max_alerts, min_precision=0.5, prefer_recall=True
        )
        cap = captura_table(y_test, proba, ks=(100, 500)).to_dict(orient="records")
        workload = threshold_workload_table(y_test, proba).to_dict(orient="records")
        importances = get_feature_importance(pipe, feature_names)

        results["models"][mname] = {
            "train_seconds": float(elapsed),
            "metrics_thr_0.5": m50,
            "metrics_selected": m_sel,
            "captura_at_k": cap,
            "workload_table": workload,
            "feature_importance_top15": dict(list(importances.items())[:15]),
            "y_proba": proba.tolist(),  # se limpia al serializar resumen
        }
        print(
            f"  [{name}] {mname}: PR-AUC={m50['pr_auc']:.4f} "
            f"Recall@0.5={m50['recall']:.3f} Prec@0.5={m50['precision']:.3f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    # Selección del mejor modelo por PR-AUC
    best_name = max(results["models"], key=lambda k: results["models"][k]["metrics_thr_0.5"]["pr_auc"])
    results["best_model"] = best_name

    # Figuras del mejor modelo
    best = results["models"][best_name]
    proba = np.array(best["y_proba"])
    y_true = np.asarray(y_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax, name=best_name)
    ax.set_title(f"Curva Precision-Recall — {name}\n{best_name}")
    savefig(f"06_pr_curve_{name}.png")

    thr = best["metrics_selected"]["threshold"]
    y_pred = (proba >= thr).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Matriz de confusión — {name}\n{best_name} (thr={thr:.2f})")
    savefig(f"07_confusion_{name}.png")

    # Captura @K
    ks = list(range(50, 1001, 50))
    found = []
    for k in ks:
        idx = np.argsort(-proba)[:k]
        found.append(int(y_true[idx].sum()))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, found, marker="o", color="#E45756")
    ax.axvline(100, color="gray", ls="--", label="K=100")
    ax.axvline(500, color="gray", ls=":", label="K=500")
    ax.set_xlabel("Operaciones revisadas (Top-K)")
    ax.set_ylabel("Fraudes encontrados")
    ax.set_title(f"Captura de fraude vs esfuerzo de revisión — {name}")
    ax.legend()
    savefig(f"08_captura_topk_{name}.png")

    # Workload tradeoff
    wt = pd.DataFrame(best["workload_table"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wt["alerts"], wt["precision"], label="Precision", color="#4C78A8")
    ax.plot(wt["alerts"], wt["recall"], label="Recall", color="#E45756")
    ax.set_xlabel("Alertas a revisar")
    ax.set_ylabel("Métrica")
    ax.set_title(f"Precision / Recall vs carga de alertas — {name}")
    ax.legend()
    savefig(f"09_workload_{name}.png")

    # Importancias
    imp = best["feature_importance_top15"]
    if imp:
        fig, ax = plt.subplots(figsize=(8, 5))
        keys = list(imp.keys())[::-1]
        vals = [imp[k] for k in keys]
        ax.barh(keys, vals, color="#54A24B")
        ax.set_title(f"Importancia de variables — {best_name} ({name})")
        savefig(f"10_importance_{name}.png")

    return results


def slim_for_json(results: dict) -> dict:
    """Quita arrays grandes (y_proba) para serializar."""
    out = {
        "split": results["split"],
        "meta": results["meta"],
        "best_model": results["best_model"],
        "models": {},
    }
    for name, m in results["models"].items():
        out["models"][name] = {
            "train_seconds": m["train_seconds"],
            "metrics_thr_0.5": m["metrics_thr_0.5"],
            "metrics_selected": m["metrics_selected"],
            "captura_at_k": m["captura_at_k"],
            "workload_table": m["workload_table"],
            "feature_importance_top15": m["feature_importance_top15"],
        }
    return out


def comparison_figure(random_res: dict, temporal_res: dict):
    rows = []
    for split_name, res in [("Aleatorio", random_res), ("Temporal", temporal_res)]:
        for mname, m in res["models"].items():
            rows.append(
                {
                    "split": split_name,
                    "model": mname,
                    "pr_auc": m["metrics_thr_0.5"]["pr_auc"],
                    "recall": m["metrics_thr_0.5"]["recall"],
                    "precision": m["metrics_thr_0.5"]["precision"],
                }
            )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="model", y="pr_auc", hue="split", ax=ax)
    ax.set_title("Comparación PR-AUC: split aleatorio vs temporal")
    ax.set_ylabel("PR-AUC")
    ax.tick_params(axis="x", rotation=20)
    savefig("11_comparison_pr_auc.png")

    # Tabla resumen
    pivot = df.pivot(index="model", columns="split", values="pr_auc")
    return df, pivot


def main():
    print("==> Cargando datos (cache CSV / mirror TensorFlow)...", flush=True)
    df_raw = load_creditcard()
    print(f"    Filas: {len(df_raw):,} | Fraudes: {df_raw['Class'].sum()}", flush=True)

    print("==> Limpieza...", flush=True)
    df = clean_dataframe(df_raw)
    print(f"    Duplicados eliminados: {df.attrs.get('n_duplicates_removed', 0)}", flush=True)
    df = add_log_amount(df)

    print("==> EDA...", flush=True)
    eda = run_eda(df)

    print("==> Split aleatorio estratificado...", flush=True)
    X, y = feature_target_split(df, use_log_amount=True)
    X_tr, X_te, y_tr, y_te = random_split(X, y)
    random_res = evaluate_split(
        "aleatorio",
        X_tr,
        X_te,
        y_tr,
        y_te,
        meta={
            "train_n": int(len(X_tr)),
            "test_n": int(len(X_te)),
            "train_fraud": int(y_tr.sum()),
            "test_fraud": int(y_te.sum()),
        },
    )

    print("==> Split temporal...", flush=True)
    X_tr_t, X_te_t, y_tr_t, y_te_t, tmeta = temporal_split(df, test_size=0.2)
    temporal_res = evaluate_split("temporal", X_tr_t, X_te_t, y_tr_t, y_te_t, meta=tmeta)

    print("==> Figura comparativa...", flush=True)
    comp_df, pivot = comparison_figure(random_res, temporal_res)

    payload = {
        "eda": eda,
        "duplicates_removed": int(df.attrs.get("n_duplicates_removed", 0)),
        "random_split": slim_for_json(random_res),
        "temporal_split": slim_for_json(temporal_res),
        "comparison_pr_auc": comp_df.to_dict(orient="records"),
    }
    RESULTS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"==> Resultados guardados en {RESULTS}", flush=True)
    print("==> Mejor modelo (aleatorio):", random_res["best_model"], flush=True)
    print("==> Mejor modelo (temporal):", temporal_res["best_model"], flush=True)
    print("Listo.", flush=True)


if __name__ == "__main__":
    main()
