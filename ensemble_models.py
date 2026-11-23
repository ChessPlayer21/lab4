"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1,
                )),
            ]), cat_cols),
        ],
        remainder="drop",
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test."""
    X_train = pd.DataFrame(data.train_features).copy()
    X_test = pd.DataFrame(data.test_features).copy()
    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()

    X_test = X_test[X_train.columns]

    pre = _build_preprocessor(X_train)

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    n_num = X_train.shape[1] - len(cat_cols)
    cat_positions = list(range(n_num, n_num + len(cat_cols)))

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_depth=3,
        l2_regularization=1.0,
        min_samples_leaf=20,
        categorical_features=cat_positions if cat_positions else None,
        early_stopping=True,
        validation_fraction=0.15,
        scoring="roc_auc",
        n_iter_no_change=30,
        random_state=data.random_seed,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return EnsembleResult(
        model_name="HGB(lr=0.05, depth=3, l2=1.0) + OrdinalEncoder",
        probabilities=proba,
        predictions=pred,
        roc_auc=float(roc_auc_score(y_test, proba)),
        accuracy=float(accuracy_score(y_test, pred)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
    )
