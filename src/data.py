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

# Mirror público del dataset ULB (mismo CSV de Kaggle / TensorFlow)
TF_MIRROR_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c.lower(): c for c in df.columns}
    for expected in ("time", "amount", "class"):
        if expected in cols_lower and cols_lower[expected] != expected.capitalize():
            df = df.rename(columns={cols_lower[expected]: expected.capitalize()})
    if "Class" not in df.columns:
        raise ValueError(f"No se encontró columna Class. Columnas: {list(df.columns)}")
    df["Class"] = pd.to_numeric(df["Class"], errors="coerce").astype(int)
    return df


def load_creditcard(cache_path: Path | None = None, force_download: bool = False) -> pd.DataFrame:
    """Carga el dataset ULB con cache local en CSV.

    Orden de fuentes:
    1) Cache local `data/creditcard.csv`
    2) Mirror TensorFlow (sin credenciales)
    3) Fallback OpenML data_id=1597
    """
    cache_path = Path(cache_path) if cache_path else DEFAULT_CSV
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_download:
        print(f"Cargando cache local: {cache_path}", flush=True)
        return _normalize_columns(pd.read_csv(cache_path))

    print(f"Descargando dataset desde mirror TensorFlow...\n{TF_MIRROR_URL}", flush=True)
    try:
        urlretrieve(TF_MIRROR_URL, cache_path)
        df = pd.read_csv(cache_path)
        return _normalize_columns(df)
    except Exception as exc:
        print(f"Mirror TensorFlow falló ({exc}). Intentando OpenML...", flush=True)

    from sklearn.datasets import fetch_openml

    bundle = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    df = _normalize_columns(bundle.frame.copy())
    df.to_csv(cache_path, index=False)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados exactos y deja una copia limpia."""
    out = df.copy()
    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    out.attrs["n_duplicates_removed"] = before - len(out)
    return out


def add_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega Amount_log = log1p(Amount) sin modificar Amount original."""
    out = df.copy()
    out["Amount_log"] = np.log1p(out["Amount"].astype(float))
    return out


def feature_target_split(df: pd.DataFrame, use_log_amount: bool = True):
    """Separa X e y. Usa Amount_log + Time + V1..V28 (sin Amount crudo si use_log_amount)."""
    y = df["Class"].astype(int)
    drop_cols = ["Class"]
    if use_log_amount and "Amount_log" in df.columns:
        drop_cols.append("Amount")
    elif "Amount_log" in df.columns:
        drop_cols.append("Amount_log")
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y


def random_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """Holdout estratificado 80/20."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def temporal_split(df: pd.DataFrame, test_size: float = 0.2, use_log_amount: bool = True):
    """Split temporal: entrena con operaciones anteriores y prueba con posteriores.

    Ordena por Time y corta en el percentil (1 - test_size).
    """
    ordered = df.sort_values("Time").reset_index(drop=True)
    cut = int(len(ordered) * (1 - test_size))
    train_df = ordered.iloc[:cut].copy()
    test_df = ordered.iloc[cut:].copy()
    X_train, y_train = feature_target_split(train_df, use_log_amount=use_log_amount)
    X_test, y_test = feature_target_split(test_df, use_log_amount=use_log_amount)
    meta = {
        "train_time_max": float(train_df["Time"].max()),
        "test_time_min": float(test_df["Time"].min()),
        "train_n": len(train_df),
        "test_n": len(test_df),
        "train_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
    }
    return X_train, X_test, y_train, y_test, meta
