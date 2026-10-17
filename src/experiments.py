"""Experimentos del TP Final: genera todas las tablas, figuras y cifras del informe.

Uso (desde la raíz del repo):
    python -m src.experiments
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data import add_features, clean_dataframe, feature_target_split, load_creditcard, temporal_split_index
from src.metrics import (
    at_threshold,
    best_f1_threshold,
    summary,
    threshold_for_fpr,
    top_k,
    top_k_table,
)
from src.modeling import MODEL_NAMES, build_pipeline, predict_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"

RANDOM_STATE = 42
REVIEW_COST_EUR = 5.0  # supuesto paramétrico: costo de una revisión manual
DAYS_IN_DATASET = 2

COLORS = {
    "Logística": "#4C78A8",
    "Random Forest": "#54A24B",
    "XGBoost": "#E45756",
    "LightGBM": "#F58518",
    "XGBoost + SMOTE": "#B279A2",
    "Regla: monto alto": "#9D755D",
    "Ranking aleatorio": "#999999",
}


def log(msg: str):
    print(msg, flush=True)


def set_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.dpi": 220,
            "figure.dpi": 110,
        }
    )


def save_fig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_table(df: pd.DataFrame, name: str):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / name, index=False)


# ---------------------------------------------------------------- datos y EDA
def prepare_data():
    raw = load_creditcard()
    clean = clean_dataframe(raw)
    df = add_features(clean)
    quality = {
        "n_raw": int(len(raw)),
        "fraud_raw": int(raw["Class"].sum()),
        "duplicados": int(clean.attrs["n_duplicates_removed"]),
        "duplicados_fraude": int(clean.attrs["n_fraud_duplicates_removed"]),
        "n": int(len(df)),
        "fraud": int(df["Class"].sum()),
        "prevalencia": float(df["Class"].mean()),
        "nulos": int(raw.isna().sum().sum()),
        "monto_cero": int((df["Amount"] == 0).sum()),
        "monto_cero_fraude": int(((df["Amount"] == 0) & (df["Class"] == 1)).sum()),
        "monto_mediana_legit": float(df.loc[df.Class == 0, "Amount"].median()),
        "monto_mediana_fraude": float(df.loc[df.Class == 1, "Amount"].median()),
        "monto_media_fraude": float(df.loc[df.Class == 1, "Amount"].mean()),
        "monto_max": float(df["Amount"].max()),
        "monto_max_fraude": float(df.loc[df.Class == 1, "Amount"].max()),
        "monto_p99": float(df["Amount"].quantile(0.99)),
        "monto_total_fraude": float(df.loc[df.Class == 1, "Amount"].sum()),
        "horas": float(df["Time"].max() / 3600),
    }
    return df, quality


def eda(df: pd.DataFrame) -> dict:
    hour = df["Hour"].astype(int)
    by_hour = df.groupby(hour).agg(transacciones=("Class", "size"), fraudes=("Class", "sum")).reset_index()
    by_hour["tasa"] = by_hour["fraudes"] / by_hour["transacciones"]
    save_table(by_hour.rename(columns={"Hour": "hora"}), "eda_por_hora.csv")

    ordered = df.sort_values("Time").reset_index(drop=True)
    ordered["tramo"] = pd.qcut(np.arange(len(ordered)), 10, labels=[f"T{i}" for i in range(1, 11)])
    by_tramo = ordered.groupby("tramo", observed=True).agg(
        transacciones=("Class", "size"), fraudes=("Class", "sum")
    ).reset_index()
    by_tramo["tasa"] = by_tramo["fraudes"] / by_tramo["transacciones"]
    save_table(by_tramo, "eda_por_tramo_temporal.csv")

    corr = df[[f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]].corr()["Class"].drop("Class")
    top_corr = corr.abs().sort_values(ascending=False).head(6)

    night = df["Hour"] < 6
    findings = {
        "tasa_noche": float(df.loc[night, "Class"].mean()),
        "tasa_dia": float(df.loc[~night, "Class"].mean()),
        "pct_tx_noche": float(night.mean()),
        "hora_max_tasa": int(by_hour.loc[by_hour["tasa"].idxmax(), "Hour"]),
        "tasa_max_hora": float(by_hour["tasa"].max()),
        "tramo_tasa_min": float(by_tramo["tasa"].min()),
        "tramo_tasa_max": float(by_tramo["tasa"].max()),
        "top_corr": {k: float(corr[k]) for k in top_corr.index},
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    ax = axes[0]
    ax.bar(by_hour["Hour"], by_hour["tasa"] * 100, color="#E45756")
    ax.axhline(df["Class"].mean() * 100, color="gray", ls="--", lw=0.8, label="Promedio")
    ax.set_title("Tasa de fraude por hora del día")
    ax.set_xlabel("Hora")
    ax.set_ylabel("% fraude")
    ax.legend(frameon=False)

    ax = axes[1]
    data = [df.loc[df.Class == 0, "Amount"] + 1, df.loc[df.Class == 1, "Amount"] + 1]
    ax.boxplot(data, labels=["Legítima", "Fraude"], showfliers=False, widths=0.5,
               medianprops={"color": "black"})
    ax.set_yscale("log")
    ax.set_title("Monto por clase (sin outliers)")
    ax.set_ylabel("Monto + 1 (€, escala log)")

    ax = axes[2]
    ax.plot(by_tramo["tramo"].astype(str), by_tramo["tasa"] * 100, marker="o", ms=3, color="#4C78A8")
    ax.set_title("Prevalencia por tramo temporal (10)")
    ax.set_xlabel("Tramo (orden de Time)")
    ax.set_ylabel("% fraude")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, "fig1_eda.png")
    return findings


# ---------------------------------------------------------------- selección
def cross_validate(X_dev, y_dev, n_splits: int = 5) -> tuple[pd.DataFrame, dict]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows, oof = [], {}
    for name in MODEL_NAMES:
        t0 = time.time()
        scores = np.zeros(len(y_dev))
        fold_rows = []
        for tr, va in skf.split(X_dev, y_dev):
            pipe = build_pipeline(name, list(X_dev.columns), y_dev.iloc[tr])
            pipe.fit(X_dev.iloc[tr], y_dev.iloc[tr])
            s = predict_score(pipe, X_dev.iloc[va])
            scores[va] = s
            m = summary(y_dev.iloc[va], s, ks=(100,))
            fold_rows.append(m)
        f = pd.DataFrame(fold_rows)
        rows.append(
            {
                "modelo": name,
                "pr_auc_media": f["pr_auc"].mean(),
                "pr_auc_desvio": f["pr_auc"].std(ddof=1),
                "roc_auc_media": f["roc_auc"].mean(),
                "precision@100_media": f["precision@100"].mean(),
                "segundos": time.time() - t0,
            }
        )
        oof[name] = scores
        log(f"  CV {name}: PR-AUC {f['pr_auc'].mean():.3f} ± {f['pr_auc'].std(ddof=1):.3f}")
    table = pd.DataFrame(rows)
    save_table(table, "cv_seleccion.csv")
    return table, oof


def select_model(cv: pd.DataFrame) -> dict:
    """Regla fijada antes de mirar los tests.

    1. Gana la mayor PR-AUC media en CV.
    2. Los modelos a menos de un desvío de la mejor PR-AUC quedan empatados.
    3. Si la logística está empatada se la elige (interpretabilidad);
       si no, desempata la mayor Precision@100 media en CV y, si persiste
       el empate, la mayor PR-AUC media.
    """
    best = cv.loc[cv["pr_auc_media"].idxmax()]
    cut = best["pr_auc_media"] - best["pr_auc_desvio"]
    tied = cv[cv["pr_auc_media"] >= cut]
    if "Logística" in tied["modelo"].values:
        chosen = "Logística"
    else:
        ranked = tied.assign(p100=tied["precision@100_media"].round(3)).sort_values(
            ["p100", "pr_auc_media"], ascending=False
        )
        chosen = ranked.iloc[0]["modelo"]
    return {"elegido": chosen, "empatados": tied["modelo"].tolist(), "corte_empate": float(cut)}


# ---------------------------------------------------------------- evaluación
def evaluate_split(X_tr, y_tr, X_te, y_te, amount_te, label: str):
    rng = np.random.default_rng(RANDOM_STATE)
    scores = {
        "Ranking aleatorio": rng.random(len(y_te)),
        "Regla: monto alto": amount_te.to_numpy(),
    }
    pipes = {}
    for name in MODEL_NAMES:
        t0 = time.time()
        pipe = build_pipeline(name, list(X_tr.columns), y_tr)
        pipe.fit(X_tr, y_tr)
        scores[name] = predict_score(pipe, X_te)
        pipes[name] = pipe
        log(f"  {label} {name}: PR-AUC {average_precision_score(y_te, scores[name]):.3f} ({time.time()-t0:.0f}s)")
    rows = [{"modelo": n, **summary(y_te, s)} for n, s in scores.items()]
    table = pd.DataFrame(rows)
    save_table(table, f"test_{label}.csv")
    return table, scores, pipes


# ---------------------------------------------------------------- interpretación
def shap_values(pipe, X: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """TreeSHAP del clasificador final. Devuelve contribuciones (+ columna base) y su unidad."""
    import shap

    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xt = pd.DataFrame(pre.transform(X), columns=pre.get_feature_names_out(), index=X.index)
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(Xt, check_additivity=False)
    base = explainer.expected_value
    if isinstance(values, list):
        values, base = values[1], base[1]
    elif np.ndim(values) == 3:
        values, base = values[:, :, 1], np.atleast_1d(base)[1]
    unit = "puntos de probabilidad" if clf.__class__.__name__ == "RandomForestClassifier" else "log-odds"
    out = pd.DataFrame(values, columns=Xt.columns, index=X.index)
    out["base"] = float(np.atleast_1d(base)[0])
    return out, unit


def interpretation(pipe, X_te, y_te, score, chosen):
    score = pd.Series(score, index=X_te.index)
    top100 = score.sort_values(ascending=False).index[:100]
    tp_top = [i for i in top100 if y_te.loc[i] == 1]
    case = tp_top[len(tp_top) // 2]

    sample = X_te.sample(600, random_state=RANDOM_STATE)
    sample = pd.concat([sample, X_te.loc[y_te == 1]])
    sample = sample[~sample.index.duplicated()]
    contrib, unit = shap_values(pipe, sample)

    glob = contrib.drop(columns="base").abs().mean().sort_values(ascending=False)
    row = contrib.loc[case]
    parts = row.drop("base").sort_values(key=np.abs, ascending=False)
    local = {
        "base": float(row["base"]),
        "score": float(score.loc[case]),
        "monto": float(np.expm1(X_te.loc[case, "Amount_log"])),
        "hora": float((np.degrees(np.arctan2(X_te.loc[case, "Hour_sin"], X_te.loc[case, "Hour_cos"])) % 360) / 15),
        "contribuciones": {k: float(v) for k, v in parts.head(8).items()},
        "resto": float(parts.iloc[8:].sum()),
    }

    top = glob.head(12)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    ax.barh(top.index[::-1], top.values[::-1], color="#4C78A8")
    ax.set_title(f"Importancia global SHAP — {chosen}")
    ax.set_xlabel(f"Media |SHAP| ({unit})")
    ax = axes[1]
    items = list(local["contribuciones"].items())[::-1]
    vals = [v for _, v in items]
    ax.barh([k for k, _ in items], vals, color=["#E45756" if v > 0 else "#4C78A8" for v in vals])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_title(f"Caso local: fraude dentro del Top-100 (score {local['score']:.2f})")
    ax.set_xlabel(f"Contribución ({unit})")
    save_fig(fig, "fig4_interpretacion.png")
    return {"metodo": f"SHAP TreeExplainer ({unit})", "n_muestra": int(len(sample)),
            "global_top": {k: float(v) for k, v in top.items()}, "local": local}


def ablation(df, idx_dev, idx_test, idx_tr_t, idx_te_t, chosen):
    rows = []
    for fs, desc in [
        ("completo", "Modelo completo"),
        ("sin_monto", "Sin monto (Amount_log)"),
        ("sin_hora", "Sin hora del día"),
        ("solo_pca", "Solo V1–V28"),
        ("con_time_crudo", "Time crudo en lugar de hora"),
    ]:
        X, y = feature_target_split(df, fs)
        out = {"variante": desc}
        for label, tr, te in [("estable", idx_dev, idx_test), ("temporal", idx_tr_t, idx_te_t)]:
            pipe = build_pipeline(chosen, list(X.columns), y.iloc[tr])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            s = predict_score(pipe, X.iloc[te])
            out[f"pr_auc_{label}"] = average_precision_score(y.iloc[te], s)
            out[f"fraudes@100_{label}"] = top_k(y.iloc[te], s, 100)["fraudes"]
        rows.append(out)
        log(f"  Ablación {desc}: {out['pr_auc_estable']:.3f} / {out['pr_auc_temporal']:.3f}")
    table = pd.DataFrame(rows)
    save_table(table, "ablaciones.csv")
    return table


def segment_audit(df_te: pd.DataFrame, score, threshold) -> pd.DataFrame:
    d = df_te[["Amount", "Hour", "Class"]].copy()
    d["alerta"] = score >= threshold
    d["tramo_monto"] = pd.cut(d["Amount"], [-0.01, 1, 10, 100, np.inf],
                              labels=["≤ €1", "€1–10", "€10–100", "> €100"])
    d["franja"] = pd.cut(d["Hour"], [-0.01, 6, 18, 24], labels=["Noche (0–6 h)", "Día (6–18 h)", "Tarde/noche (18–24 h)"])
    rows = []
    for col, title in [("tramo_monto", "Monto"), ("franja", "Franja horaria")]:
        for g, sub in d.groupby(col, observed=True):
            fr = int(sub["Class"].sum())
            al = int(sub["alerta"].sum())
            tp = int((sub["alerta"] & (sub["Class"] == 1)).sum())
            rows.append({
                "atributo": title, "grupo": str(g), "transacciones": len(sub), "fraudes": fr,
                "tasa": fr / len(sub), "alertas": al,
                "recall": tp / fr if fr else np.nan, "precision": tp / al if al else np.nan,
            })
    table = pd.DataFrame(rows)
    save_table(table, "auditoria_segmentos.csv")
    return table


# ---------------------------------------------------------------- negocio
def economic_scenario(y_te, score, rule_score, amount_te, threshold, n_total, n_test):
    y = np.asarray(y_te).astype(int)
    a = np.asarray(amount_te, dtype=float)
    per_day = (n_total / DAYS_IN_DATASET) / n_test

    def row(name, selected):
        alerts = int(selected.sum())
        tp = int((selected & (y == 1)).sum())
        detected = float(a[selected & (y == 1)].sum())
        cost = alerts * REVIEW_COST_EUR
        return {
            "escenario": name, "alertas": alerts, "fraudes": tp, "recall": tp / y.sum(),
            "precision": tp / alerts if alerts else 0.0, "monto_detectado": detected,
            "costo_revision": cost, "neto": detected - cost,
            "alertas_dia": alerts * per_day, "neto_dia": (detected - cost) * per_day,
        }

    def top_mask(s, k):
        m = np.zeros(len(s), dtype=bool)
        m[np.argsort(-np.asarray(s), kind="stable")[:k]] = True
        return m

    fpr_thr = threshold_for_fpr(y, score, 0.05)
    rows = [
        row("Regla estática: Top-500 por monto", top_mask(rule_score, 500)),
        row("Modelo: Top-100", top_mask(score, 100)),
        row("Modelo: Top-500", top_mask(score, 500)),
        row("Modelo: umbral operativo", np.asarray(score) >= threshold),
        row("Modelo: umbral con FPR = 5 %", np.asarray(score) >= fpr_thr),
    ]
    table = pd.DataFrame(rows)
    save_table(table, "escenario_economico.csv")
    return table, {"factor_dia": per_day, "umbral_fpr5": fpr_thr,
                   "monto_fraude_test": float(a[y == 1].sum())}


# ---------------------------------------------------------------- figuras de evaluación
def fig_test(y_te, scores, chosen):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    for name in MODEL_NAMES:
        p, r, _ = precision_recall_curve(y_te, scores[name])
        ap = average_precision_score(y_te, scores[name])
        ax.plot(r, p, lw=1.6 if name == chosen else 0.9, color=COLORS[name], label=f"{name} ({ap:.3f})")
    ax.axhline(np.mean(y_te), color="gray", ls="--", lw=0.8, label=f"Aleatorio ({np.mean(y_te):.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall (test estable)")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    ks = np.arange(10, 1001, 10)
    y = np.asarray(y_te)
    for name in [chosen, "Logística", "Regla: monto alto", "Ranking aleatorio"]:
        order = np.argsort(-np.asarray(scores[name]), kind="stable")
        found = np.cumsum(y[order])[ks - 1]
        ax.plot(ks, found, color=COLORS[name], lw=1.6 if name == chosen else 1.0, label=name)
    ax.axvline(100, color="gray", ls="--", lw=0.8)
    ax.axvline(500, color="gray", ls=":", lw=0.8)
    ax.axhline(y.sum(), color="black", lw=0.6, ls="-.", label=f"Total fraudes ({int(y.sum())})")
    ax.set_xlabel("Operaciones revisadas (Top-K)")
    ax.set_ylabel("Fraudes encontrados")
    ax.set_title("Captura de fraude según esfuerzo de revisión")
    ax.legend(frameon=False, loc="center right")
    save_fig(fig, "fig2_test_estable.png")


def fig_temporal(test_stable, test_temporal, y_st, s_st, y_tm, s_tm, chosen):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    names = MODEL_NAMES
    x = np.arange(len(names))
    st = test_stable.set_index("modelo").loc[names, "pr_auc"]
    tm = test_temporal.set_index("modelo").loc[names, "pr_auc"]
    ax.bar(x - 0.2, st, 0.4, label="Test estable", color="#4C78A8")
    ax.bar(x + 0.2, tm, 0.4, label="Test temporal", color="#F58518")
    ax.set_xticks(x, [n.replace(" + ", "+\n") for n in names])
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("PR-AUC: split aleatorio vs temporal")
    ax.legend(frameon=False)

    ax = axes[1]
    ks = np.arange(10, 1001, 10)
    for label, y, s, c in [("Test estable", y_st, s_st, "#4C78A8"), ("Test temporal", y_tm, s_tm, "#F58518")]:
        y = np.asarray(y)
        order = np.argsort(-np.asarray(s), kind="stable")
        ax.plot(ks, np.cumsum(y[order])[ks - 1] / y.sum() * 100, color=c, label=label)
    ax.axvline(100, color="gray", ls="--", lw=0.8)
    ax.axvline(500, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("Operaciones revisadas (Top-K)")
    ax.set_ylabel("% de fraudes capturados")
    ax.set_title(f"Recall@K — {chosen}")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "fig3_temporal.png")


def fig_threshold(y_te, score, threshold, fpr_thr):
    p, r, t = precision_recall_curve(y_te, score)
    alerts = np.array([(np.asarray(score) >= th).sum() for th in t])
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.plot(alerts, p[:-1] * 100, color="#4C78A8", label="Precision")
    ax.plot(alerts, r[:-1] * 100, color="#E45756", label="Recall")
    for th, lab, ls in [(threshold, "Umbral operativo", "--"), (fpr_thr, "FPR = 5 %", ":")]:
        n = int((np.asarray(score) >= th).sum())
        ax.axvline(n, color="black", ls=ls, lw=0.8)
        ax.text(n * 1.08, 55, f"{lab}\n{n:,} alertas".replace(",", "."), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Alertas a revisar (escala log)")
    ax.set_ylabel("%")
    ax.set_title("Precision y Recall según la cantidad de alertas (test estable)")
    ax.legend(frameon=False, loc="lower left")
    save_fig(fig, "fig5_umbral_alertas.png")


# ---------------------------------------------------------------- orquestación
def run_all() -> dict:
    set_style()
    t0 = time.time()
    log("1/8 Datos y calidad")
    df, quality = prepare_data()

    log("2/8 EDA")
    findings = eda(df)

    X, y = feature_target_split(df, "completo")
    idx = np.arange(len(df))
    idx_dev, idx_test = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    idx_tr_t, idx_te_t = temporal_split_index(df, 0.2)
    X_dev, y_dev, X_te, y_te = X.iloc[idx_dev], y.iloc[idx_dev], X.iloc[idx_test], y.iloc[idx_test]
    amount_te = df["Amount"].iloc[idx_test]

    log("3/8 Validación cruzada (selección)")
    cache = RESULTS_DIR / "cv_cache.pkl"
    if cache.exists():
        cv, oof = pd.read_pickle(cache)
        log("  CV leída de caché")
    else:
        cv, oof = cross_validate(X_dev, y_dev)
        pd.to_pickle((cv, oof), cache)
    selection = select_model(cv)
    chosen = selection["elegido"]
    threshold = best_f1_threshold(y_dev, oof[chosen])
    oof_at_thr = at_threshold(y_dev, oof[chosen], threshold)
    log(f"  Elegido: {chosen} | umbral OOF {threshold:.3f}")

    log("4/8 Test estable")
    test_st, scores_st, pipes_st = evaluate_split(X_dev, y_dev, X_te, y_te, amount_te, "estable")
    op_st = at_threshold(y_te, scores_st[chosen], threshold)
    topk_st = top_k_table(y_te, scores_st[chosen], amount_te)
    save_table(topk_st, "topk_estable.csv")
    fig_test(y_te, scores_st, chosen)

    log("5/8 Test temporal")
    X_tr_t, y_tr_t = X.iloc[idx_tr_t], y.iloc[idx_tr_t]
    X_te_t, y_te_t = X.iloc[idx_te_t], y.iloc[idx_te_t]
    amount_te_t = df["Amount"].iloc[idx_te_t]
    test_tm, scores_tm, _ = evaluate_split(X_tr_t, y_tr_t, X_te_t, y_te_t, amount_te_t, "temporal")
    op_tm = at_threshold(y_te_t, scores_tm[chosen], threshold)
    topk_tm = top_k_table(y_te_t, scores_tm[chosen], amount_te_t)
    save_table(topk_tm, "topk_temporal.csv")
    fig_temporal(test_st, test_tm, y_te, scores_st[chosen], y_te_t, scores_tm[chosen], chosen)
    temporal_meta = {
        "corte_horas": float(df["Time"].iloc[idx_te_t].min() / 3600),
        "n_train": int(len(idx_tr_t)), "n_test": int(len(idx_te_t)),
        "fraude_train": int(y_tr_t.sum()), "fraude_test": int(y_te_t.sum()),
        "prev_train": float(y_tr_t.mean()), "prev_test": float(y_te_t.mean()),
    }

    log("6/8 Negocio")
    econ, econ_meta = economic_scenario(y_te, scores_st[chosen], scores_st["Regla: monto alto"],
                                        amount_te, threshold, len(df), len(idx_test))
    fpr5 = at_threshold(y_te, scores_st[chosen], econ_meta["umbral_fpr5"])
    fig_threshold(y_te, scores_st[chosen], threshold, econ_meta["umbral_fpr5"])
    audit = segment_audit(df.iloc[idx_test], scores_st[chosen], threshold)

    log("7/8 Interpretación")
    interp = interpretation(pipes_st[chosen], X_te, y_te, scores_st[chosen], chosen)

    log("8/8 Ablaciones")
    abl = ablation(df, idx_dev, idx_test, idx_tr_t, idx_te_t, chosen)

    results = {
        "calidad": quality,
        "eda": findings,
        "split": {"n_dev": int(len(idx_dev)), "n_test": int(len(idx_test)),
                  "fraude_dev": int(y_dev.sum()), "fraude_test": int(y_te.sum())},
        "cv": cv.to_dict(orient="records"),
        "seleccion": selection,
        "umbral": {"valor": threshold, "regla": "máximo F1 sobre predicciones out-of-fold del desarrollo",
                   "oof": oof_at_thr, "test_estable": op_st, "test_temporal": op_tm, "fpr5_test": fpr5},
        "test_estable": test_st.to_dict(orient="records"),
        "test_temporal": test_tm.to_dict(orient="records"),
        "temporal_meta": temporal_meta,
        "topk_estable": topk_st.to_dict(orient="records"),
        "topk_temporal": topk_tm.to_dict(orient="records"),
        "economico": econ.to_dict(orient="records"),
        "economico_meta": {**econ_meta, "costo_revision": REVIEW_COST_EUR},
        "auditoria": audit.to_dict(orient="records"),
        "interpretacion": interp,
        "ablaciones": abl.to_dict(orient="records"),
        "segundos": time.time() - t0,
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float),
                                              encoding="utf-8")
    log(f"Listo en {results['segundos'] / 60:.1f} min → {RESULTS_DIR}")
    return results


if __name__ == "__main__":
    run_all()
