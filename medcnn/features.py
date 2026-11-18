from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch


@contextmanager
def capture(layer):
    """Own a detached clone: downstream in-place operations cannot corrupt it."""
    captured = {}

    def hook(module, inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Expected N,C,H,W intermediate activations")
        captured["activation"] = output.detach().clone()

    handle = layer.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@torch.inference_mode()
def extract(model, batches):
    model.eval()
    pooled, intermediate, labels, ids = [], [], [], []
    with capture(model.target_layer) as captured:
        for x, y, index in batches:
            representation = model.backbone(x)
            pooled.append(representation.cpu().numpy())
            intermediate.append(captured["activation"].mean((2, 3)).cpu().numpy())
            labels.append(np.asarray(y))
            ids.append(np.asarray(index))
    return {
        "pooled": np.concatenate(pooled),
        "intermediate": np.concatenate(intermediate),
        "labels": np.concatenate(labels),
        "ids": np.concatenate(ids),
    }


def rank_channels(features, labels):
    """Signed standardized mean difference: positive means pneumonia associated."""
    features = np.asarray(features, dtype=np.float64)
    a, b = features[labels == 1], features[labels == 0]
    if min(len(a), len(b)) < 2:
        raise ValueError("Need >=2 training examples per class for channel ranking")
    variance = ((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (
        len(a) + len(b) - 2
    )
    floor = max(float(np.sqrt(variance).max()) * 1e-6, 1e-8)
    score = (a.mean(0) - b.mean(0)) / np.maximum(np.sqrt(variance), floor)
    table = pd.DataFrame(
        {
            "channel": np.arange(features.shape[1]),
            "association": score,
            "normal_mean": b.mean(0),
            "pneumonia_mean": a.mean(0),
            "std": features.std(0),
            "normal_std": b.std(0),
            "pneumonia_std": a.std(0),
        }
    )
    table = table.sort_values(
        ["association", "channel"], ascending=[False, True]
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table
