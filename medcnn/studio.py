"""Controlled dataset interventions and portable experiment exports."""

import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import predict
from .interventions import compare, select_channels
from .core import intervene, layer_for


def run_studio(model, batches, layer_name, ranking, threshold, training_mean,
               budgets=(1, 4, 8), repeats=3, seed=42, methods=("zero", "training_mean")):
    if repeats < 2 or not budgets or any(b < 1 or b > len(ranking) for b in budgets):
        raise ValueError("Use valid positive budgets and at least two random controls")
    baseline = predict(model, batches)
    summary, individual = [], []
    layer = layer_for(model, layer_name)
    for method in methods:
        for budget in budgets:
            for strategy in ("top", "low", "random"):
                for repeat in range(repeats if strategy == "random" else 1):
                    channels = select_channels(ranking, strategy, budget, seed + repeat)
                    with intervene(layer, channels, method, training_mean):
                        changed = predict(model, batches)
                    condition = dict(method=method, layer=layer_name, strategy=strategy,
                                     budget=budget, repetition=repeat, seed=seed + repeat,
                                     channels=json.dumps(channels))
                    summary.append({**condition, **compare(baseline, changed, threshold)})
                    for i, image_id in enumerate(baseline["ids"]):
                        individual.append({**condition, "id": int(image_id),
                                           "label": int(baseline["labels"][i]),
                                           "baseline_probability": float(baseline["probabilities"][i]),
                                           "probability": float(changed["probabilities"][i])})
    return pd.DataFrame(summary), pd.DataFrame(individual)


def plot_summary(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, metric, title in zip(axes,
        ("mean_absolute_probability_change", "delta_balanced_accuracy"),
        ("Mean absolute probability change", "Change in balanced accuracy")):
        for (method, strategy), group in summary.groupby(["method", "strategy"]):
            grouped = group.groupby("budget")[metric]
            mean, sd = grouped.mean(), grouped.std().fillna(0)
            ax.errorbar(mean.index, mean, yerr=sd, marker="o", capsize=3,
                        label=f"{method} / {strategy}")
        ax.set(xlabel="Channels intervened", ylabel=title)
        ax.axhline(0, color="gray", linewidth=.7)
    axes[1].legend(fontsize=7, loc="best")
    return fig


def png_bytes(summary):
    fig = plot_summary(summary)
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=160)
    plt.close(fig)
    return stream.getvalue()
