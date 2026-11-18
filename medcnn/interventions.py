from contextlib import contextmanager

import numpy as np
import pandas as pd

from .evaluate import metrics, predict


@contextmanager
def ablate(layer, channels):
    channels = sorted({int(c) for c in channels})
    if any(c < 0 for c in channels):
        raise ValueError("Channel indices must be nonnegative")

    def hook(module, inputs, output):
        if output.ndim != 4:
            raise ValueError("Expected N,C,H,W activations")
        if channels and max(channels) >= output.shape[1]:
            raise ValueError("Channel index exceeds layer width")
        modified = output.clone()
        modified[:, channels, :, :] = 0
        return modified

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def select_channels(ranking, strategy, budget, seed=42):
    order = ranking["channel"].to_numpy(dtype=int)
    if not 0 <= budget <= len(order):
        raise ValueError("Budget outside channel count")
    if strategy == "top":
        return order[:budget].tolist()
    if strategy == "low":
        # Bottom of signed ranking: often strongly normal associated, not unimportant.
        return order[len(order) - budget :].tolist() if budget else []
    if strategy == "random":
        return (
            np.random.default_rng(seed)
            .choice(np.sort(order), budget, replace=False)
            .tolist()
        )
    raise ValueError(strategy)


def compare(baseline, changed, threshold):
    if not np.array_equal(baseline["ids"], changed["ids"]):
        raise ValueError("Interventions must use identical ordered samples")
    delta = changed["probabilities"] - baseline["probabilities"]
    a, b = (
        metrics(baseline["labels"], baseline["probabilities"], threshold),
        metrics(changed["labels"], changed["probabilities"], threshold),
    )
    return {
        "mean_signed_probability_change": float(delta.mean()),
        "mean_absolute_probability_change": float(abs(delta).mean()),
        "mean_logit_change": float((changed["logits"] - baseline["logits"]).mean()),
        "class_flip_percent": float(
            100
            * np.mean(
                (changed["probabilities"] >= threshold)
                != (baseline["probabilities"] >= threshold)
            )
        ),
        "delta_roc_auc": b["roc_auc"] - a["roc_auc"],
        "delta_balanced_accuracy": b["balanced_accuracy"] - a["balanced_accuracy"],
        **b,
    }


def run_ablations(
    model, batches, ranking, threshold, budgets=(1, 4, 8), repeats=5, seed=42
):
    baseline = predict(model, batches)
    rows, raw = [], []
    conditions = [("baseline", 0, 0, [])]
    for budget in budgets:
        for strategy in ("top", "low", "random"):
            for repetition in range(repeats if strategy == "random" else 1):
                channels = select_channels(ranking, strategy, budget, seed + repetition)
                conditions.append((strategy, budget, repetition, channels))
    for strategy, budget, repetition, channels in conditions:
        print(
            f"Ablation: {strategy}, budget={budget}, repetition={repetition}",
            flush=True,
        )
        with ablate(model.target_layer, channels):
            changed = predict(model, batches)
        rows.append(
            {
                "strategy": strategy,
                "budget": budget,
                "repetition": repetition,
                "channels": str(channels),
                "selection_seed": seed + repetition,
                **compare(baseline, changed, threshold),
            }
        )
        for i, idx in enumerate(baseline["ids"]):
            raw.append(
                {
                    "strategy": strategy,
                    "budget": budget,
                    "repetition": repetition,
                    "id": int(idx),
                    "label": int(baseline["labels"][i]),
                    "baseline_probability": float(baseline["probabilities"][i]),
                    "probability": float(changed["probabilities"][i]),
                    "baseline_logit": float(baseline["logits"][i]),
                    "logit": float(changed["logits"][i]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(raw)
