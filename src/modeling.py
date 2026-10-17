"""Construcción de pipelines y modelos para detección de fraude."""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


SCALE_COLS = ["Time", "Amount_log"]


def _identity(X):
    return X


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    """Escala Time y Amount_log; pasa V* sin cambios."""
    scale_cols = [c for c in SCALE_COLS if c in feature_names]
    passthrough = [c for c in feature_names if c not in scale_cols]
    return ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scale_cols),
            ("pass", FunctionTransformer(_identity, validate=False), passthrough),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def scale_pos_weight(y) -> float:
    y = np.asarray(y).astype(int)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = int((y == 0).sum())
    return float(n_neg / n_pos)


def make_estimators(y_train, random_state: int = 42) -> dict[str, Any]:
    """Diccionario de estimadores listos (sin preprocessor)."""
    spw = scale_pos_weight(y_train)
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            is_unbalance=True,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        ),
    }


def build_pipeline(estimator, feature_names: list[str], use_smote: bool = False, random_state: int = 42):
    pre = build_preprocessor(feature_names)
    if use_smote:
        return ImbPipeline(
            steps=[
                ("pre", pre),
                ("smote", SMOTE(random_state=random_state, k_neighbors=5)),
                ("clf", estimator),
            ]
        )
    return Pipeline(steps=[("pre", pre), ("clf", estimator)])


def predict_proba_positive(pipeline, X) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    # clase positiva = 1
    classes = list(pipeline.named_steps["clf"].classes_)
    pos_idx = classes.index(1) if 1 in classes else -1
    return proba[:, pos_idx]


def get_feature_importance(pipeline, feature_names: list[str]) -> dict[str, float]:
    """Importancias o coeficientes absolutos del estimador final."""
    clf = pipeline.named_steps["clf"]
    pre = pipeline.named_steps["pre"]
    # Orden de features tras ColumnTransformer
    try:
        out_names = list(pre.get_feature_names_out())
    except Exception:
        out_names = feature_names

    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_).ravel()
    else:
        return {}

    if len(vals) != len(out_names):
        # fallback: mapear por posición de feature_names originales
        n = min(len(vals), len(feature_names))
        return {feature_names[i]: float(vals[i]) for i in range(n)}

    pairs = sorted(zip(out_names, vals), key=lambda x: -float(x[1]))
    return {k: float(v) for k, v in pairs}
