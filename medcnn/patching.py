"""Activation transfer between two preprocessed images."""

import numpy as np
import pandas as pd
import torch

from .workbench import capture_map, device_of, edit_map, get_layer, logits, probabilities


def _slices(shape, channels, region):
    _n, c, h, w = shape
    chosen = list(range(c)) if channels is None else sorted(set(int(v) for v in channels))
    if not chosen or chosen[0] < 0 or chosen[-1] >= c:
        raise ValueError("Select valid channels")
    if region is None:
        return chosen, slice(0, h), slice(0, w)
    if len(region) != 4 or not (0 <= region[0] < region[2] <= 1 and
                                0 <= region[1] < region[3] <= 1):
        raise ValueError("Region must be normalized (x0,y0,x1,y1)")
    x0, y0, x1, y1 = region
    xs = slice(min(w - 1, int(x0 * w)), max(1, int(np.ceil(x1 * w))))
    ys = slice(min(h - 1, int(y0 * h)), max(1, int(np.ceil(y1 * h))))
    return chosen, ys, xs


@torch.inference_mode()
def patch_activations(model, spec, recipient, donor, layer_name,
                      channels=None, region=None):
    """Measure original, donor, same-image control, and patched predictions."""
    layer = get_layer(model, layer_name)
    device = device_of(model)
    recipient, donor = recipient[None].to(device), donor[None].to(device)
    with capture_map(layer) as captured:
        donor_output = logits(model, donor)
        donor_map = captured["map"].clone()
        original_output = logits(model, recipient)
        own_map = captured["map"].clone()
    if donor_map.shape != own_map.shape:
        raise ValueError("Donor and recipient activation shapes differ")
    chosen, ys, xs = _slices(own_map.shape, channels, region)
    def transfer(source):
        def transform(output):
            output[:, chosen, ys, xs] = source[:, chosen, ys, xs]
        return transform
    with edit_map(layer, transfer(own_map)):
        control_output = logits(model, recipient)
    with edit_map(layer, transfer(donor_map)):
        patched_output = logits(model, recipient)
    if not torch.allclose(control_output, original_output, rtol=0, atol=0):
        raise AssertionError("Same-image patch failed to restore baseline")
    return {"original": probabilities(original_output, spec)[0].cpu().numpy(),
            "donor": probabilities(donor_output, spec)[0].cpu().numpy(),
            "control": probabilities(control_output, spec)[0].cpu().numpy(),
            "patched": probabilities(patched_output, spec)[0].cpu().numpy(),
            "channels": chosen, "region": region, "layer": layer_name}


def sensitivity_scan(model, spec, recipient, donor, layer_names, target_class,
                     channels_by_layer=None, region=None):
    if not 0 <= target_class < len(spec.class_names):
        raise ValueError("Invalid target class")
    rows = []
    for name in layer_names:
        choices = (channels_by_layer or {}).get(name, [None])
        for channel in choices:
            result = patch_activations(model, spec, recipient, donor, name,
                                       None if channel is None else [channel], region)
            rows.append({"layer": name, "channel": "all" if channel is None else int(channel),
                         "target_class": spec.class_names[target_class],
                         "original_probability": float(result["original"][target_class]),
                         "donor_probability": float(result["donor"][target_class]),
                         "patched_probability": float(result["patched"][target_class]),
                         "delta_probability": float(result["patched"][target_class] -
                                                    result["original"][target_class]),
                         "control_delta": float(result["control"][target_class] -
                                                result["original"][target_class])})
    return pd.DataFrame(rows)
