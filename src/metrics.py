"""Métricas técnicas y de negocio para detección de fraude."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def top_k(y_true, score, k: int) -> dict:
    """Si revisamos las K operaciones de mayor riesgo, ¿cuántos fraudes encontramos?"""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    k = min(int(k), len(y))
    idx = np.argsort(-s, kind="stable")[:k]
    found = int(y[idx].sum())
    n_pos = int(y.sum())
    prevalence = n_pos / len(y)
    precision = found / k
    return {
        "k": k,
        "fraudes": found,
        "precision_k": precision,
        "recall_k": found / n_pos if n_pos else 0.0,
        "lift_k": precision / prevalence if prevalence else 0.0,
    }


def at_threshold(y_true, score, threshold: float) -> dict:
    y = np.asarray(y_true).astype(int)
    pred = np.asarray(score) >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    alerts = tp + fp
    return {
        "umbral": float(threshold),
        "alertas": alerts,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / alerts if alerts else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "f1": f1_score(y, pred.astype(int), zero_division=0),
        "fpr": fp / int((y == 0).sum()),
    }


def threshold_for_fpr(y_true, score, fpr: float) -> float:
    """Umbral que deja pasar como alerta una fracción `fpr` de las legítimas."""
    y = np.asarray(y_true).astype(int)
    neg = np.asarray(score)[y == 0]
    return float(np.quantile(neg, 1 - fpr))


def best_f1_threshold(y_true, score) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def summary(y_true, score, ks=(100, 500)) -> dict:
    row = {
        "pr_auc": average_precision_score(y_true, score),
        "roc_auc": roc_auc_score(y_true, score),
        "prevalencia": float(np.mean(y_true)),
    }
    for k in ks:
        t = top_k(y_true, score, k)
        row[f"fraudes@{k}"] = t["fraudes"]
        row[f"precision@{k}"] = t["precision_k"]
        row[f"recall@{k}"] = t["recall_k"]
        row[f"lift@{k}"] = t["lift_k"]
    return row


def top_k_table(y_true, score, amount, ks=(50, 100, 200, 500, 1000)) -> pd.DataFrame:
    """Tabla Top-K con monto de fraude capturado."""
    y = np.asarray(y_true).astype(int)
    a = np.asarray(amount, dtype=float)
    order = np.argsort(-np.asarray(score), kind="stable")
    total_amount = a[y == 1].sum()
    rows = []
    for k in ks:
        idx = order[:k]
        t = top_k(y, score, k)
        t["monto_capturado"] = float(a[idx][y[idx] == 1].sum())
        t["pct_monto"] = t["monto_capturado"] / total_amount if total_amount else 0.0
        rows.append(t)
    return pd.DataFrame(rows)
