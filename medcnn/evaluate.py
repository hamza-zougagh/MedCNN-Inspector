import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def select_threshold(labels, probabilities):
    """Maximize validation Youden J; deterministic tie break closest to 0.5."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    choices = np.flatnonzero(valid)
    best = choices[np.isclose((tpr - fpr)[choices], (tpr - fpr)[choices].max())]
    return float(thresholds[best[np.argmin(abs(thresholds[best] - 0.5))]])


def metrics(labels, probabilities, threshold):
    predicted = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    both = len(np.unique(labels)) == 2
    return {
        "n": len(labels),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if both else None,
        "pr_auc_average_precision": float(
            average_precision_score(labels, probabilities)
        )
        if both
        else None,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


@torch.inference_mode()
def predict(model, batches):
    model.eval()
    logits, labels, indices = [], [], []
    for x, y, ids in batches:
        logits.append(model(x).cpu().numpy())
        labels.append(np.asarray(y))
        indices.append(np.asarray(ids))
    z = np.concatenate(logits)
    return {
        "logits": z,
        "probabilities": torch.from_numpy(z).sigmoid().numpy(),
        "labels": np.concatenate(labels),
        "ids": np.concatenate(indices),
    }
