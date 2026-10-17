"""Carga y preparación del dataset Credit Card Fraud (ULB)."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV = DATA_DIR / "creditcard.csv"

TF_MIRROR_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

PCA_COLS = [f"V{i}" for i in range(1, 29)]
EXPECTED_COLS = ["Time", *PCA_COLS, "Amount", "Class"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}
    for expected in ("time", "amount", "class"):
        if expected in cols_lower and cols_lower[expected] != expected.capitalize():
            df = df.rename(columns={cols_lower[expected]: expected.capitalize()})
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Esquema inesperado, faltan columnas: {missing}")
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce").astype(int)
    return df[EXPECTED_COLS]


def load_creditcard(cache_path: Path | None = None, force_download: bool = False) -> pd.DataFrame:
    """Carga el dataset ULB con cache local y valida su esquema.

    Fuentes: cache `data/creditcard.csv`, mirror público de TensorFlow y, como
    último recurso, OpenML (data_id=1597).
    """
    cache_path = Path(cache_path) if cache_path else DEFAULT_CSV
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_download:
        return _normalize_columns(pd.read_csv(cache_path))

    try:
        urlretrieve(TF_MIRROR_URL, cache_path)
        return _normalize_columns(pd.read_csv(cache_path))
    except Exception as exc:
        print(f"Mirror TensorFlow no disponible ({exc}); usando OpenML.", flush=True)

    from sklearn.datasets import fetch_openml

    bundle = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    df = _normalize_columns(bundle.frame.copy())
    df.to_csv(cache_path, index=False)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados exactos y registra cuántos se quitaron."""
    before = len(df)
    before_fraud = int(df["Class"].sum())
    out = df.drop_duplicates().reset_index(drop=True)
    out.attrs["n_duplicates_removed"] = before - len(out)
    out.attrs["n_fraud_duplicates_removed"] = before_fraud - int(out["Class"].sum())
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ingeniería de variables.

    - Amount_log = log1p(Amount): reduce la asimetría del monto.
    - Hour_sin / Hour_cos: hora del día en forma cíclica. Reemplaza a Time crudo,
      que en una validación temporal toma en test valores fuera del rango visto
      en entrenamiento.
    """
    out = df.copy()
    out["Amount_log"] = np.log1p(out["Amount"].astype(float))
    hour = (out["Time"] % 86400) / 3600.0
    out["Hour"] = hour
    out["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


FEATURE_SETS = {
    "completo": [*PCA_COLS, "Amount_log", "Hour_sin", "Hour_cos"],
    "sin_monto": [*PCA_COLS, "Hour_sin", "Hour_cos"],
    "sin_hora": [*PCA_COLS, "Amount_log"],
    "solo_pca": list(PCA_COLS),
    "con_time_crudo": [*PCA_COLS, "Amount_log", "Time"],
}


def feature_target_split(df: pd.DataFrame, feature_set: str = "completo"):
    cols = FEATURE_SETS[feature_set]
    return df[cols], df["Class"].astype(int)


def random_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """Holdout estratificado 80/20 (desarrollo / test estable)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def temporal_split_index(df: pd.DataFrame, test_size: float = 0.2):
    """Índices de un split temporal: 80 % inicial por Time para train, 20 % final para test."""
    order = np.argsort(df["Time"].to_numpy(), kind="stable")
    cut = int(len(order) * (1 - test_size))
    return order[:cut], order[cut:]
