"""Métricas técnicas y de negocio para detección de fraude."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """PR-AUC, Recall, Precision (principales) + F1, ROC-AUC y matriz de confusión."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    alerts = int(y_pred.sum())

    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / n_neg) if n_neg else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "alerts": alerts,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "alert_rate": float(alerts / len(y_true)) if len(y_true) else 0.0,
    }


def captura_at_k(y_true, y_proba, k: int) -> dict:
    """Si revisamos las K operaciones de mayor riesgo, ¿cuántos fraudes encontramos?"""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    k = min(int(k), len(y_true))
    if k <= 0:
        return {"k": 0, "frauds_found": 0, "precision_at_k": 0.0, "recall_at_k": 0.0, "lift": 0.0}

    idx = np.argsort(-y_proba)[:k]
    frauds_found = int(y_true[idx].sum())
    n_pos = int(y_true.sum())
    base_rate = n_pos / len(y_true) if len(y_true) else 0.0
    precision_at_k = frauds_found / k
    recall_at_k = frauds_found / n_pos if n_pos else 0.0
    lift = precision_at_k / base_rate if base_rate > 0 else 0.0
    return {
        "k": k,
        "frauds_found": frauds_found,
        "precision_at_k": float(precision_at_k),
        "recall_at_k": float(recall_at_k),
        "lift": float(lift),
        "base_rate": float(base_rate),
    }


def captura_table(y_true, y_proba, ks: Iterable[int] = (100, 500)) -> pd.DataFrame:
    rows = [captura_at_k(y_true, y_proba, k) for k in ks]
    return pd.DataFrame(rows)


def threshold_workload_table(
    y_true,
    y_proba,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Tabla umbral → Precision, Recall, alertas a revisar (carga operativa)."""
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 19), 2)
    rows = []
    for t in thresholds:
        m = classification_metrics(y_true, y_proba, threshold=float(t))
        rows.append(
            {
                "threshold": m["threshold"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "fpr": m["fpr"],
                "alerts": m["alerts"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"],
            }
        )
    return pd.DataFrame(rows)


def select_threshold_by_workload(
    y_true,
    y_proba,
    max_alerts: int | None = None,
    min_precision: float = 0.5,
    prefer_recall: bool = True,
) -> dict:
    """Elige umbral priorizando Precision y carga de alertas (no solo FPR).

    Estrategia:
    - Si max_alerts está definido, restringe a umbrales con alerts <= max_alerts.
    - Entre candidatos con Precision >= min_precision, maximiza Recall (o F1).
    - Fallback: maximiza F1.
    """
    table = threshold_workload_table(y_true, y_proba)
    candidates = table.copy()
    if max_alerts is not None:
        restricted = candidates[candidates["alerts"] <= max_alerts]
        if len(restricted):
            candidates = restricted

    high_prec = candidates[candidates["precision"] >= min_precision]
    if len(high_prec):
        if prefer_recall:
            best = high_prec.sort_values(["recall", "precision", "f1"], ascending=False).iloc[0]
        else:
            best = high_prec.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    else:
        best = candidates.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]

    thr = float(best["threshold"])
    metrics = classification_metrics(y_true, y_proba, threshold=thr)
    metrics["selection_rule"] = (
        f"max_alerts={max_alerts}, min_precision={min_precision}, prefer_recall={prefer_recall}"
    )
    return metrics


def pr_curve_arrays(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return precision, recall, thresholds
