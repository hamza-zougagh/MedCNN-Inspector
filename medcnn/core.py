"""Layer registry, hooks, and training-only channel statistics."""

from contextlib import contextmanager

import numpy as np
import torch

from .features import capture, rank_channels


LAYERS = {
    "resnet18": ("layer1", "layer2", "layer3", "layer4"),
    "densenet121": (
        "features.denseblock1", "features.denseblock2",
        "features.denseblock3", "features.denseblock4",
    ),
}


def layer_for(model, name):
    if name not in LAYERS.get(model.architecture, ()):
        raise ValueError(f"Unsupported layer {name!r} for {model.architecture}")
    return model.backbone.get_submodule(name)


@contextmanager
def intervene(layer, channels, method="zero", training_mean=None):
    """Replace selected channels without mutating module output or parameters."""
    channels = sorted(set(int(c) for c in channels))
    if any(c < 0 for c in channels) or method not in ("zero", "training_mean"):
        raise ValueError("Invalid channels or intervention method")

    def hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Expected N,C,H,W activations")
        if channels and channels[-1] >= output.shape[1]:
            raise ValueError("Channel outside layer width")
        changed = output.clone()
        if channels:
            if method == "zero":
                changed[:, channels] = 0
            else:
                if training_mean is None or len(training_mean) != output.shape[1]:
                    raise ValueError("Training mean must match layer width")
                values = torch.as_tensor(
                    np.asarray(training_mean)[channels],
                    dtype=output.dtype, device=output.device,
                )
                changed[:, channels] = values[None, :, None, None]
        return changed

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def channel_features(model, batches, layer_name, progress=None):
    model.eval()
    values, labels, ids = [], [], []
    with capture(layer_for(model, layer_name)) as captured:
        for number, (x, y, index) in enumerate(batches, 1):
            model(x)
            values.append(captured["activation"].mean((2, 3)).cpu().numpy())
            labels.append(np.asarray(y))
            ids.append(np.asarray(index))
            if progress:
                progress(number, len(batches))
    return np.concatenate(values), np.concatenate(labels), np.concatenate(ids)


def training_statistics(model, batches, layer_name, progress=None):
    values, labels, ids = channel_features(model, batches, layer_name, progress)
    return values.mean(0), rank_channels(values, labels), ids


@torch.inference_mode()
def inspect_image(model, image, layer_name):
    with capture(layer_for(model, layer_name)) as captured:
        logit = model(image.unsqueeze(0)).item()
    return logit, torch.sigmoid(torch.tensor(logit)).item(), captured["activation"][0]
