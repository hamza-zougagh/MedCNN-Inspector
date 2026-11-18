"""Dataset-level feature extraction, rankings, spatial maxima, and clustering."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch

from .utils import file_hash
from .workbench import capture_map, device_of, get_layer, logits, probabilities


def _cache_path(spec, layer_name, items, cache_dir):
    digest = hashlib.sha256()
    digest.update(str(spec.checkpoint.resolve()).encode())
    digest.update(file_hash(spec.checkpoint).encode())
    digest.update(layer_name.encode())
    digest.update(json.dumps([item[0] for item in items]).encode())
    for _id, tensor in items:
        digest.update(tensor.contiguous().numpy().tobytes())
    return Path(cache_dir) / f"{digest.hexdigest()[:24]}.npz"


@torch.inference_mode()
def extract_bank(model, spec, layer_name, items, batch_size=8, cache_dir=None, progress=None):
    """Extract real 4D maps and compact descriptors from ordered image tensors."""
    if not items:
        raise ValueError("Add at least one image")
    if len({item[0] for item in items}) != len(items):
        raise ValueError("Image IDs must be unique")
    cache = _cache_path(spec, layer_name, items, cache_dir) if cache_dir else None
    if cache and cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}
    layer = get_layer(model, layer_name)
    means, maxima, positions, scores = [], [], [], []
    with capture_map(layer) as captured:
        for start in range(0, len(items), batch_size):
            batch = torch.stack([x for _id, x in items[start:start + batch_size]])
            output = logits(model, batch.to(device_of(model)))
            maps = captured["map"].cpu()
            n, c, h, w = maps.shape
            flat = maps.flatten(2)
            peak, offset = flat.max(2)
            means.append(maps.mean((2, 3)).numpy())
            maxima.append(peak.numpy())
            positions.append(torch.stack((offset // w, offset % w), -1).numpy())
            scores.append(probabilities(output, spec).cpu().numpy())
            if progress:
                progress(min(start + batch_size, len(items)), len(items))
    result = {"ids": np.asarray([str(item[0]) for item in items]),
              "mean": np.concatenate(means), "maximum": np.concatenate(maxima),
              "peak_yx": np.concatenate(positions), "probabilities": np.concatenate(scores)}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **result)
    return result


def top_examples(bank, channel, count=8, statistic="maximum"):
    if statistic not in ("maximum", "mean"):
        raise ValueError("Choose maximum or mean")
    values = bank[statistic][:, channel]
    order = np.argsort(-values, kind="stable")[:count]
    return [{"id": str(bank["ids"][i]), "activation": float(values[i]),
             "peak_y": int(bank["peak_yx"][i, channel, 0]),
             "peak_x": int(bank["peak_yx"][i, channel, 1])} for i in order]


def cluster_features(bank, clusters=3, seed=42):
    if not 2 <= clusters <= len(bank["ids"]):
        raise ValueError("Cluster count must be between 2 and image count")
    normalized = StandardScaler().fit_transform(bank["mean"])
    assignment = KMeans(n_clusters=clusters, random_state=seed, n_init=10).fit_predict(normalized)
    return assignment
