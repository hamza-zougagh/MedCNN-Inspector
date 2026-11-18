import json

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from medcnn.concepts import concept_intervention, fit_concept
from medcnn.feature_discovery import cluster_features, extract_bank, top_examples
from medcnn.interactions import interaction_experiment, network_png
from medcnn.patching import patch_activations, sensitivity_scan
from medcnn.workbench import (FACTORIES, discover_layers, load_model, predict_image,
                              preprocess, probabilities, read_spec, register_factory,
                              get_layer,
                              ModelSpec)


class ToyCNN(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(4, 5, 3, padding=1), nn.ReLU())
        self.classifier = nn.Linear(5, classes)

    def forward(self, x):
        return self.classifier(self.features(x).mean((2, 3)))


@pytest.fixture
def setup_model(tmp_path):
    torch.manual_seed(7)
    if "toy_cnn_test" not in FACTORIES:
        register_factory("toy_cnn_test", ToyCNN)
    model = ToyCNN().eval()
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    spec_path = tmp_path / "model.json"
    spec_path.write_text(json.dumps({"name": "Toy", "factory": "toy_cnn_test",
        "checkpoint": "weights.pt", "class_names": ["a", "b", "c"],
        "input_size": [32, 32], "input_channels": 3,
        "mean": [0, 0, 0], "std": [1, 1, 1]}))
    spec = read_spec(spec_path)
    return model, load_model(spec), spec, spec_path


def test_safe_registry_multiclass_and_discovery(setup_model, tmp_path):
    original, model, spec, path = setup_model
    x = preprocess(Image.new("RGB", (40, 40), (100, 80, 60)), spec)
    assert x.shape == (3, 32, 32)
    probabilities = predict_image(model, x, spec)
    assert probabilities.shape == (3,)
    assert probabilities.sum() == pytest.approx(1)
    shapes = discover_layers(model, x[None])
    assert shapes["features.0"] == (1, 4, 32, 32)
    assert shapes["features.2"] == (1, 5, 32, 32)
    assert not model.features[0]._forward_hooks
    invalid = json.loads(path.read_text())
    (tmp_path / "missing.json").write_text("{}")
    with pytest.raises(ValueError, match="missing"):
        read_spec(tmp_path / "missing.json")
    invalid["factory"] = "uploaded_python_code"
    (tmp_path / "invalid.json").write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="Unknown model factory"):
        read_spec(tmp_path / "invalid.json")
    torch.save(original, tmp_path / "unsafe.pt")
    invalid["factory"] = "toy_cnn_test"
    invalid["checkpoint"] = "unsafe.pt"
    (tmp_path / "invalid.json").write_text(json.dumps(invalid))
    with pytest.raises(Exception):
        load_model(read_spec(tmp_path / "invalid.json"))


def test_binary_logits_and_probability_validation(tmp_path):
    spec = ModelSpec("Binary", "toy_cnn_test", tmp_path / "unused.pt",
        ("negative", "positive"), (32, 32))
    predicted = probabilities(torch.tensor([[0.0], [2.0]]), spec)
    assert predicted.shape == (2, 2)
    assert predicted[0].tolist() == pytest.approx([.5, .5])
    probability_spec = ModelSpec("Binary", "toy_cnn_test", tmp_path / "unused.pt",
        ("negative", "positive"), (32, 32), output_kind="probabilities")
    with pytest.raises(ValueError):
        probabilities(torch.tensor([[1.5]]), probability_spec)
    with pytest.raises(ValueError, match="NaN or infinity"):
        probabilities(torch.tensor([[float("nan")]]), spec)


def test_feature_bank_patching_and_interactions(setup_model, tmp_path):
    _, model, spec, _ = setup_model
    items = [(str(i), torch.rand(3, 32, 32)) for i in range(6)]
    bank = extract_bank(model, spec, "features.2", items, batch_size=2,
                        cache_dir=tmp_path / "cache")
    assert bank["mean"].shape == (6, 5)
    assert top_examples(bank, 0, count=2)[0]["activation"] >= top_examples(bank, 0, count=2)[1]["activation"]
    assert len(cluster_features(bank, 2)) == 6
    second = extract_bank(model, spec, "features.2", items, cache_dir=tmp_path / "cache")
    np.testing.assert_array_equal(second["mean"], bank["mean"])
    parameters = {key: value.clone() for key, value in model.state_dict().items()}
    patched = patch_activations(model, spec, items[0][1], items[1][1], "features.2",
                                channels=[0, 1], region=(.25, .25, .75, .75))
    np.testing.assert_array_equal(patched["original"], patched["control"])
    scan = sensitivity_scan(model, spec, items[0][1], items[1][1],
                            ["features.0", "features.2"], 1)
    assert len(scan) == 2
    table = interaction_experiment(model, spec, items[0][1], "features.2",
                                   [0, 1, 2], 1)
    assert len(table) == 3
    assert len(network_png(table)) > 100
    assert not model.features[2]._forward_hooks
    with pytest.raises(ValueError, match="Unknown layer"):
        get_layer(model, "features.no_such_layer")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, parameters[key])


def test_concept_split_and_direction_controls(setup_model):
    _, model, spec, _ = setup_model
    def make(prefix, brightness, count):
        return [(f"{prefix}{i}", torch.full((3, 32, 32), brightness + i * .02))
                for i in range(count)]
    fit_pos, fit_neg = make("fp", .8, 4), make("fn", .1, 4)
    eval_pos, eval_neg = make("ep", .7, 2), make("en", .2, 2)
    result = fit_concept(model, spec, "features.2", fit_pos, fit_neg, eval_pos, eval_neg)
    assert result["train_n"] == 8 and result["eval_n"] == 4
    assert np.isfinite(result["evaluation_roc_auc"])
    changed = concept_intervention(model, spec, eval_pos[0][1], "features.2",
                                   result["direction"], 1, alpha=.5)
    assert changed["baseline"] == changed["zero_control"]
    assert not model.features[2]._forward_hooks
    with pytest.raises(ValueError):
        fit_concept(model, spec, "features.2", fit_pos, fit_neg,
                    fit_pos[:1], eval_neg)


@pytest.mark.parametrize("architecture", ["resnet18", "densenet121",
                                           "efficientnet_b0", "vgg16"])
def test_torchvision_family_has_4d_layers(architecture):
    from medcnn.workbench import FACTORIES
    model = FACTORIES[architecture](3).eval()
    shapes = discover_layers(model, torch.zeros(1, 3, 64, 64))
    assert shapes
    assert all(len(shape) == 4 for shape in shapes.values())


@pytest.mark.parametrize("architecture", ["efficientnet_b0", "vgg16"])
def test_additional_torchvision_families_support_real_hooks(architecture, tmp_path):
    torch.manual_seed(11)
    model = FACTORIES[architecture](3).eval()
    spec = ModelSpec(architecture, architecture, tmp_path / "not_saved.pt",
                     ("a", "b", "c"), (64, 64))
    x, donor = torch.rand(3, 64, 64), torch.rand(3, 64, 64)
    shapes = discover_layers(model, x[None])
    layer_name = next(name for name, module in model.named_modules()
                      if isinstance(module, nn.Conv2d) and name in shapes and shapes[name][1] >= 2)
    layer = get_layer(model, layer_name)
    result = patch_activations(model, spec, x, donor, layer_name, channels=[0])
    np.testing.assert_array_equal(result["original"], result["control"])
    assert np.isfinite(result["patched"]).all()
    assert not layer._forward_hooks
    if architecture == "efficientnet_b0":
        table = interaction_experiment(model, spec, x, layer_name, [0, 1], 1)
        assert len(table) == 1
        assert table.iloc[0].nonadditivity == pytest.approx(
            table.iloc[0].pair_effect - table.iloc[0].first_effect -
            table.iloc[0].second_effect)
