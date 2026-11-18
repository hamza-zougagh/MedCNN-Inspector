"""User-defined concept probes and controlled direction interventions."""

import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from .feature_discovery import extract_bank
from .workbench import device_of, edit_map, get_layer, logits, probabilities


def _digests(items):
    return {hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for _, x in items}


def fit_concept(model, spec, layer_name, train_positive, train_negative,
                eval_positive, eval_negative, regularization=1.0,
                cache_dir=None):
    """Fit on concept-training examples only; evaluate on disjoint examples."""
    groups = (train_positive, train_negative, eval_positive, eval_negative)
    if any(len(group) < (2 if i < 2 else 1) for i, group in enumerate(groups)):
        raise ValueError("Need >=2 training and >=1 evaluation image per concept class")
    if regularization <= 0:
        raise ValueError("Regularization C must be positive")
    train_items = train_positive + train_negative
    eval_items = eval_positive + eval_negative
    if (_digests(train_positive) & _digests(train_negative) or
        _digests(eval_positive) & _digests(eval_negative)):
        raise ValueError("The same image cannot have both concept labels")
    if (set(item[0] for item in train_items) & set(item[0] for item in eval_items) or
        _digests(train_items) & _digests(eval_items)):
        raise ValueError("Concept fit and evaluation images must be disjoint")
    train_bank = extract_bank(model, spec, layer_name, train_items, cache_dir=cache_dir)
    eval_bank = extract_bank(model, spec, layer_name, eval_items, cache_dir=cache_dir)
    y_train = np.r_[np.ones(len(train_positive)), np.zeros(len(train_negative))]
    y_eval = np.r_[np.ones(len(eval_positive)), np.zeros(len(eval_negative))]
    estimator = make_pipeline(StandardScaler(),
                              LogisticRegression(C=regularization, max_iter=1000))
    estimator.fit(train_bank["mean"], y_train)
    score = estimator.predict_proba(eval_bank["mean"])[:, 1]
    scaler, classifier = estimator.steps[0][1], estimator.steps[1][1]
    direction = classifier.coef_[0] / scaler.scale_
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        raise ValueError("Fitted concept direction is degenerate")
    direction = direction / norm
    return {"direction": direction.astype(np.float32),
            "evaluation_accuracy": float(accuracy_score(y_eval, score >= .5)),
            "evaluation_roc_auc": float(roc_auc_score(y_eval, score)),
            "evaluation_probabilities": score,
            "evaluation_labels": y_eval.astype(int),
            "evaluation_ids": eval_bank["ids"],
            "train_n": len(train_items), "eval_n": len(eval_items),
            "regularization_C": regularization, "layer": layer_name}


@torch.inference_mode()
def concept_intervention(model, spec, image, layer_name, direction,
                         target_class, alpha=1.0, seed=42):
    """Compare signed concept shifts with a seeded matched-norm random control."""
    if alpha <= 0 or not 0 <= target_class < len(spec.class_names):
        raise ValueError("Choose positive alpha and a valid target class")
    layer = get_layer(model, layer_name)
    x = image[None].to(device_of(model))
    direction = np.asarray(direction, dtype=np.float32)
    random = np.random.default_rng(seed).normal(size=direction.shape).astype(np.float32)
    random /= max(float(np.linalg.norm(random)), 1e-12)
    def score(vector=None, scale=0):
        if vector is None:
            return float(probabilities(logits(model, x), spec)[0, target_class])
        def edit(output):
            if len(vector) != output.shape[1]:
                raise ValueError("Concept direction width differs from layer")
            values = torch.as_tensor(vector, device=output.device, dtype=output.dtype)
            output += scale * values[None, :, None, None]
        with edit_map(layer, edit):
            return float(probabilities(logits(model, x), spec)[0, target_class])
    baseline = score()
    positive, negative = score(direction, alpha), score(direction, -alpha)
    random_control, zero_control = score(random, alpha), score(direction, 0)
    if zero_control != baseline or layer._forward_hooks:
        raise AssertionError("Zero-shift control failed or hook leaked")
    return {"baseline": baseline, "positive_shift": positive,
            "negative_shift": negative, "random_control": random_control,
            "zero_control": zero_control,
            "directional_sensitivity": (positive - negative) / (2 * alpha),
            "target_class": spec.class_names[target_class], "alpha": alpha,
            "seed": seed, "layer": layer_name}
