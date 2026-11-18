import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import metrics, select_threshold


def fit_linear(train, val):
    """Select regularization on validation ROC-AUC; scaler sees training only."""
    candidates = []
    for c in (0.01, 0.1, 1.0):
        estimator = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        )
        estimator.fit(train[0], train[1])
        p = estimator.predict_proba(val[0])[:, 1]
        score = metrics(val[1], p, 0.5)["roc_auc"]
        candidates.append((score, c, estimator, p))
    _, c, estimator, p = max(candidates, key=lambda item: item[0])
    threshold = select_threshold(val[1], p)
    return (
        estimator,
        threshold,
        {
            "C": c,
            "candidates": [
                {"C": item[1], "validation_roc_auc": float(item[0])}
                for item in candidates
            ],
            "validation": metrics(val[1], p, threshold),
        },
    )


def install_head(model, estimator):
    scaler, linear = estimator.steps[0][1], estimator.steps[1][1]
    weight = linear.coef_[0] / scaler.scale_
    bias = linear.intercept_[0] - np.dot(weight, scaler.mean_)
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor(weight[None], dtype=torch.float32))
        model.head.bias.copy_(torch.tensor([bias], dtype=torch.float32))
    model.eval()
