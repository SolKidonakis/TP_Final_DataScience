"""Pipelines y modelos para detección de fraude."""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SCALE_COLS = ["Amount_log", "Time"]

MODEL_NAMES = ["Logística", "Random Forest", "XGBoost", "LightGBM", "XGBoost + SMOTE"]


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    """Escala Amount_log (y Time si está); el resto pasa sin cambios."""
    scale_cols = [c for c in SCALE_COLS if c in feature_names]
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), scale_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def scale_pos_weight(y) -> float:
    y = np.asarray(y).astype(int)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def make_estimator(name: str, y_train, random_state: int = 42):
    spw = scale_pos_weight(y_train)
    if name == "Logística":
        return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if name in ("XGBoost", "XGBoost + SMOTE"):
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0 if name == "XGBoost + SMOTE" else spw,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "LightGBM":
        # is_unbalance=True resultó inestable (PR-AUC ~0); se usa peso moderado y regularización.
        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=float(np.sqrt(spw)),
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )
    raise ValueError(name)


def build_pipeline(name: str, feature_names: list[str], y_train, random_state: int = 42):
    pre = build_preprocessor(feature_names)
    clf = make_estimator(name, y_train, random_state)
    if name == "XGBoost + SMOTE":
        return ImbPipeline(
            [("pre", pre), ("smote", SMOTE(random_state=random_state)), ("clf", clf)]
        )
    return Pipeline([("pre", pre), ("clf", clf)])


def predict_score(pipeline, X) -> np.ndarray:
    return pipeline.predict_proba(X)[:, 1]
