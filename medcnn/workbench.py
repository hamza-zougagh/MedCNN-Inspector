"""Safe local model registry and generic 2D CNN instrumentation.

Only registered Python factories are callable. JSON specs select a factory and a
local tensor state_dict; model files are never executed or unpickled as modules.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .models import MedicalCNN


FACTORIES = {}


def register_factory(name, factory):
    if not name or not callable(factory) or name in FACTORIES:
        raise ValueError("Factory needs a unique name and a callable")
    FACTORIES[name] = factory


def _torchvision_factory(architecture):
    def create(classes):
        from torchvision import models
        return getattr(models, architecture)(weights=None, num_classes=classes)
    return create


for _name in ("resnet18", "densenet121", "efficientnet_b0", "vgg16"):
    register_factory(_name, _torchvision_factory(_name))
register_factory("medical_resnet18", lambda _classes: MedicalCNN("resnet18", pretrained=False))
register_factory("medical_densenet121", lambda _classes: MedicalCNN("densenet121", pretrained=False))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: str
    checkpoint: Path
    class_names: tuple[str, ...]
    input_size: tuple[int, int]
    input_channels: int = 3
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    output_kind: str = "logits"
    dataset: str | None = None
    metadata: dict | None = None


def read_spec(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Model specification must be a JSON object")
    allowed = {"name", "factory", "checkpoint", "class_names", "input_size",
               "input_channels", "mean", "std", "output_kind", "dataset", "metadata"}
    required = {"name", "factory", "checkpoint", "class_names", "input_size"}
    if missing := required - set(raw):
        raise ValueError(f"Model specification is missing: {', '.join(sorted(missing))}")
    if set(raw) - allowed:
        raise ValueError(f"Unknown model spec fields: {sorted(set(raw) - allowed)}")
    if raw["factory"] not in FACTORIES:
        raise ValueError(f"Unknown model factory {raw['factory']!r}; choose a registered architecture")
    if not isinstance(raw.get("class_names"), list):
        raise ValueError("class_names must be a list")
    classes = tuple(raw["class_names"])
    size = tuple(int(v) for v in raw["input_size"])
    channels = int(raw.get("input_channels", 3))
    mean = tuple(float(v) for v in raw.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(float(v) for v in raw.get("std", [0.229, 0.224, 0.225]))
    if (not raw.get("name") or
        len(classes) < 2 or any(not isinstance(v, str) or not v.strip() for v in classes) or
        len(set(classes)) != len(classes) or
        len(size) != 2 or min(size) < 16 or max(size) > 1024 or
        channels not in (1, 3) or len(mean) != channels or len(std) != channels or
        not all(np.isfinite(mean)) or not all(np.isfinite(std)) or
        any(v <= 0 for v in std) or raw.get("output_kind", "logits") not in
        ("logits", "probabilities")):
        raise ValueError("Invalid or unsupported model specification")
    checkpoint = (path.parent / raw["checkpoint"]).resolve()
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict) or set(metadata) - {
        "training_dataset", "preprocessing_confirmed", "class_order_confirmed"}:
        raise ValueError("Invalid model provenance metadata")
    if ("training_dataset" in metadata and not isinstance(metadata["training_dataset"], str) or
        any(key in metadata and not isinstance(metadata[key], bool)
            for key in ("preprocessing_confirmed", "class_order_confirmed"))):
        raise ValueError("Model provenance must use a text dataset name and Boolean confirmations")
    return ModelSpec(raw["name"], raw["factory"], checkpoint, classes, size,
                     channels, mean, std, raw.get("output_kind", "logits"),
                     raw.get("dataset"), metadata)


def load_model(spec, device="cpu"):
    """Load only a tensor state_dict using PyTorch's restricted loader."""
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if device not in ("cpu", "cuda"):
        raise ValueError("Choose cpu or cuda")
    state = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("Checkpoint must contain a tensor state_dict")
    model = FACTORIES[spec.factory](1 if spec.factory.startswith("medical_") else len(spec.class_names))
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def preprocess(image, spec):
    from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
    mode = "L" if spec.input_channels == 1 else "RGB"
    transform = Compose([Resize(spec.input_size, interpolation=InterpolationMode.BILINEAR,
                                antialias=True), ToTensor(), Normalize(spec.mean, spec.std)])
    return transform(image.convert(mode))


def logits(model, x):
    output = model(x)
    if not isinstance(output, torch.Tensor):
        raise ValueError("Expected a tensor classification output")
    if output.ndim == 1:
        output = output[:, None]
    if output.ndim != 2 or output.shape[0] != x.shape[0]:
        raise ValueError("Expected N,C classification output")
    return output


def probabilities(output, spec):
    if not torch.isfinite(output).all():
        raise ValueError("Model output contains NaN or infinity")
    if output.shape[1] == 1:
        if len(spec.class_names) != 2:
            raise ValueError("Single output needs exactly two class names")
        positive = output[:, 0].sigmoid() if spec.output_kind == "logits" else output[:, 0]
        if spec.output_kind == "probabilities" and (positive.min() < 0 or positive.max() > 1):
            raise ValueError("Probability output outside [0,1]")
        return torch.stack((1 - positive, positive), 1)
    if output.shape[1] != len(spec.class_names):
        raise ValueError("Output width does not match class names")
    if spec.output_kind == "probabilities":
        if (output.min() < 0 or output.max() > 1 or
            not torch.allclose(output.sum(1), torch.ones_like(output[:, 0]), atol=1e-3)):
            raise ValueError("Invalid multiclass probabilities")
        return output
    return output.softmax(1)


def device_of(model):
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


@torch.inference_mode()
def predict_image(model, x, spec):
    if x.ndim != 3:
        raise ValueError("Expected a single C,H,W image")
    output = logits(model, x[None].to(device_of(model)))
    return probabilities(output, spec).detach().cpu()[0].numpy()


@torch.inference_mode()
def discover_layers(model, sample):
    """Probe named modules once and return 4D tensor outputs with exact shapes."""
    shapes, handles = {}, []
    def record(name):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                shapes[name] = tuple(int(v) for v in output.shape)
        return hook
    try:
        for name, module in model.named_modules():
            if name:
                handles.append(module.register_forward_hook(record(name)))
        logits(model, sample.to(device_of(model)))
    finally:
        for handle in handles:
            handle.remove()
    return shapes


def get_layer(model, name, compatible=None):
    if compatible is not None and name not in compatible:
        raise ValueError("Layer was not discovered as a 4D activation")
    if not name:
        raise ValueError("Choose a named 4D feature layer")
    try:
        return model.get_submodule(name)
    except AttributeError as exc:
        raise ValueError(f"Unknown layer {name!r}; choose a discovered 4D layer") from exc


@contextmanager
def capture_map(layer):
    captured = {}
    def hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Expected 4D activation")
        captured["map"] = output.detach().clone()
    handle = layer.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def edit_map(layer, transform):
    def hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Expected 4D activation")
        changed = output.clone()
        transform(changed)
        return changed
    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def image_digest(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def load_local_factories():
    """Import the explicitly editable local registry, never a spec-supplied module."""
    from . import local_models  # noqa: F401
