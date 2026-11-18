"""Individual and paired channel effects with a small interaction graph."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .workbench import device_of, edit_map, get_layer, logits, probabilities


@torch.inference_mode()
def interaction_experiment(model, spec, image, layer_name, channels, target_class,
                           method="zero", training_mean=None):
    channels = sorted(set(int(v) for v in channels))
    if not 2 <= len(channels) <= 8 or method not in ("zero", "training_mean"):
        raise ValueError("Choose 2–8 channels and a supported replacement")
    if not 0 <= target_class < len(spec.class_names):
        raise ValueError("Invalid target class")
    layer = get_layer(model, layer_name)
    x = image[None].to(device_of(model))
    def score(selected):
        if not selected:
            return float(probabilities(logits(model, x), spec)[0, target_class])
        def replace(output):
            if max(selected) >= output.shape[1] or min(selected) < 0:
                raise ValueError("Channel outside layer width")
            if method == "zero":
                output[:, selected] = 0
            else:
                if training_mean is None or len(training_mean) != output.shape[1]:
                    raise ValueError("Training mean must match layer width")
                values = torch.as_tensor(np.asarray(training_mean)[selected],
                                         device=output.device, dtype=output.dtype)
                output[:, selected] = values[None, :, None, None]
        with edit_map(layer, replace):
            return float(probabilities(logits(model, x), spec)[0, target_class])
    baseline = score([])
    singles = {channel: score([channel]) for channel in channels}
    rows = []
    for i, first in enumerate(channels):
        for second in channels[i + 1:]:
            pair = score([first, second])
            effect_first = singles[first] - baseline
            effect_second = singles[second] - baseline
            rows.append({"first": first, "second": second,
                         "baseline_probability": baseline,
                         "first_probability": singles[first],
                         "second_probability": singles[second],
                         "pair_probability": pair,
                         "first_effect": effect_first,
                         "second_effect": effect_second,
                         "pair_effect": pair - baseline,
                         "nonadditivity": pair - singles[first] - singles[second] + baseline,
                         "target_class": spec.class_names[target_class],
                         "method": method, "layer": layer_name})
    restored = score([])
    if restored != baseline or layer._forward_hooks:
        raise AssertionError("Model output or hooks were not restored")
    return pd.DataFrame(rows)


def network_png(table):
    channels = sorted(set(table["first"]) | set(table["second"]))
    theta = np.linspace(0, 2 * np.pi, len(channels), endpoint=False)
    position = {channel: np.array([np.cos(t), np.sin(t)]) for channel, t in zip(channels, theta)}
    fig, ax = plt.subplots(figsize=(5, 5))
    maximum = max(float(table["nonadditivity"].abs().max()), 1e-9)
    for row in table.itertuples():
        start, end = position[row.first], position[row.second]
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color="tab:red" if row.nonadditivity > 0 else "tab:blue",
                alpha=.25 + .7 * abs(row.nonadditivity) / maximum,
                linewidth=.5 + 4 * abs(row.nonadditivity) / maximum)
    for channel, point in position.items():
        ax.scatter(*point, s=450, color="white", edgecolors="black", zorder=3)
        ax.text(*point, str(channel), ha="center", va="center", zorder=4)
    ax.set(aspect="equal", title="Pairwise nonadditivity for selected channels")
    ax.axis("off")
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stream.getvalue()
