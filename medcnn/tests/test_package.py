"""Tests that run without the presentation project's src package."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from medcnn.core import intervene, layer_for, training_statistics
from medcnn.pneumoniamnist import preprocess_uploaded
from medcnn.studio import run_studio


class TinyMedicalCNN(nn.Module):
    architecture = "resnet18"

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential()
        self.backbone.add_module("layer1", nn.Conv2d(3, 4, 3, padding=1))
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(self.backbone(x).mean((2, 3))).squeeze(-1)


def batches():
    x = torch.linspace(0, 1, 20 * 3 * 8 * 8).reshape(20, 3, 8, 8)
    y = torch.tensor([0, 1] * 10)
    return DataLoader(TensorDataset(x, y, torch.arange(20)), batch_size=5)


def test_training_statistics_and_both_interventions():
    torch.manual_seed(42)
    model = TinyMedicalCNN().eval()
    layer = layer_for(model, "layer1")
    mean, ranking, ids = training_statistics(model, batches(), "layer1")
    assert len(mean) == len(ranking) == 4
    np.testing.assert_array_equal(ids, np.arange(20))
    x = next(iter(batches()))[0]
    with torch.inference_mode():
        original = model(x)
        with intervene(layer, [0], "zero"):
            zeroed = model(x)
        with intervene(layer, [0], "training_mean", mean):
            replaced = model(x)
        restored = model(x)
    assert not torch.equal(zeroed, original)
    assert not torch.equal(replaced, original)
    torch.testing.assert_close(restored, original, rtol=0, atol=0)
    assert not layer._forward_hooks


def test_studio_uses_real_paired_predictions():
    torch.manual_seed(42)
    model = TinyMedicalCNN().eval()
    mean, ranking, _ = training_statistics(model, batches(), "layer1")
    summary, raw = run_studio(model, batches(), "layer1", ranking, .5, mean,
        budgets=(1,), repeats=2, methods=("zero", "training_mean"))
    assert len(summary) == 8
    assert len(raw) == 160
    assert all(raw.groupby(["method", "strategy", "repetition"]).id.nunique() == 20)


def test_uploaded_image_uses_expected_preprocessing():
    image = Image.fromarray(np.full((64, 64), 128, dtype=np.uint8), mode="L")
    tensor = preprocess_uploaded(image, 128)
    assert tensor.shape == (3, 128, 128)
    assert torch.isfinite(tensor).all()
