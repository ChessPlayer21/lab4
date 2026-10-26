"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    X_train = pd.DataFrame(data.train_features)
    X_test = pd.DataFrame(data.test_features)
    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_mask = [c in cat_cols for c in X_train.columns]

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_depth=3,
        categorical_features=cat_mask,
        early_stopping=True,
        random_state=data.random_seed,
    )
    model.fit(X_train, y_train)            

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return EnsembleResult(
        model_name="HistGradientBoosting",
        probabilities=proba,
        predictions=pred,
        roc_auc=float(roc_auc_score(y_test, proba)),
        accuracy=float(accuracy_score(y_test, pred)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
    )
