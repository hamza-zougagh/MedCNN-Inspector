from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from medcnn.catalog import DatasetHandle, import_dataset, list_datasets, register_model
from medcnn.general_experiments import (bootstrap_ci, run_experiment,
                                        save_experiment, summarize, summary_png)
from medcnn.workbench import (FACTORIES, load_model, predict_image, preprocess,
                              read_spec, register_factory)


class ThreeClassCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Conv2d(3, 5, 3, padding=1)
        self.classifier = nn.Linear(5, classes)

    def forward(self, x):
        return self.classifier(self.features(x).relu().mean((2, 3)))


def _fixture(tmp_path, duplicate=False):
    if "experiment_test_cnn" not in FACTORIES:
        register_factory("experiment_test_cnn", ThreeClassCNN)
    weights = tmp_path / "weights.pt"
    torch.manual_seed(7)
    torch.save(ThreeClassCNN(3).state_dict(), weights)
    root = tmp_path / "registry"
    spec_path = register_model("Three class CNN", "experiment_test_cnn", weights,
        ["a", "b", "c"], (24, 24), mean=(0, 0, 0), std=(1, 1, 1), root=root)
    source = tmp_path / "images"
    for split, count in (("train", 9), ("val", 3), ("test", 4)):
        for class_index, label in enumerate(("a", "b", "c")):
            for index in range(count):
                number = {"train": 0, "val": 70, "test": 130}[split]
                image = np.full((24, 24, 3), number + class_index * 20 + index,
                                dtype=np.uint8)
                path = source / split / label / f"{index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(path)
    manifest = import_dataset("Three class images", "folder", source, root=root,
                              split_mode="existing")
    entry = next(row for row in list_datasets(root) if row["path"] == manifest)
    spec = read_spec(spec_path)
    dataset = DatasetHandle(entry)
    if duplicate:
        # A previously imported index without content hashes still receives
        # experiment-time split protection.
        dataset.frame = dataset.frame.drop(columns=["content_sha256"])
        dataset.by_split = {split: part.drop(columns=["content_sha256"])
                            for split, part in dataset.by_split.items()}
        (source / "test/a/0.png").write_bytes((source / "train/a/0.png").read_bytes())
    return load_model(spec), spec, dataset, root


def test_bootstrap_and_repetition_summary():
    assert bootstrap_ci([2, 2, 2, 2], 9, 200) == (2.0, 2.0)
    assert bootstrap_ci([0, 1, 2, 3], 9, 200) == bootstrap_ci([0, 1, 2, 3], 9, 200)
    rows = pd.DataFrame([{"image_id": image, "trial": trial,
        "effect": image / 10, "random_effect": 0, "control_effect": 0}
        for trial in range(3) for image in range(1, 21)])
    report = summarize(rows, seed=9, bootstrap_draws=200)
    assert report["unique_test_images"] == 20
    assert report["image_sign_consistency"] == 1
    assert report["evidence_label"] == "repeated test-image sensitivity"
    assert report["effect_ci_low"] > 0
    single = summarize(rows[rows.trial == 0], seed=9, bootstrap_draws=200)
    assert single["trial_sign_consistency"] is None
    assert single["between_trial_sd"] is None


@pytest.mark.parametrize("kind", ["ablation", "patching", "concept"])
def test_custom_multiclass_matched_deterministic_experiments(tmp_path, kind):
    model, spec, dataset, root = _fixture(tmp_path)
    initial_parameters = {name: value.detach().clone()
                          for name, value in model.state_dict().items()}
    _, sample_image, _ = dataset.item("test", 0)
    sample_tensor = preprocess(sample_image, spec)
    baseline_before = predict_image(model, sample_tensor, spec)
    arguments = dict(kind=kind, sample_size=4, repeats=2, seed=13,
                     reference_size=6, bootstrap_draws=200, cache_dir=None)
    first = run_experiment(model, spec, dataset, "features", 0, **arguments)
    second = run_experiment(model, spec, dataset, "features", 0, **arguments)
    raw, report, config = first
    pd.testing.assert_frame_equal(raw, second[0])
    assert report == second[1]
    assert config == second[2]
    assert len(raw) == 8 and raw.trial.nunique() == 2
    assert np.allclose(raw.baseline_control, raw.baseline, atol=0, rtol=0)
    assert config["test_ids_by_trial"] and report["unique_test_images"] <= 8
    assert set(config["reference_ids"]).isdisjoint(set(raw.image_id))
    if kind == "concept":
        assert config["probe_validation_ids"]
        assert set(config["probe_validation_ids"]).isdisjoint(set(raw.image_id))
    if kind == "patching":
        assert raw.donor_id.ne("").all() and raw.random_donor_id.ne("").all()
        assert (raw.donor_id != raw.random_donor_id).all()
        assert all(first.split("/")[-2] == second.split("/")[-2]
                   for first, second in zip(raw.donor_id, raw.random_donor_id))
    else:
        assert raw.random_control.notna().all()
    saved = save_experiment(raw, report, config, root)
    assert (saved / "config.json").exists() and (saved / "per_image.csv").exists()
    assert (saved / "comparison.png").read_bytes() == summary_png(report)
    assert save_experiment(raw, report, config, root) == saved
    changed_rows = raw.copy()
    changed_rows.loc[0, "effect"] += .001
    preserved = (saved / "per_image.csv").read_bytes()
    another = save_experiment(changed_rows, report, config, root)
    assert another != saved and (saved / "per_image.csv").read_bytes() == preserved
    assert not model.features._forward_hooks
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial_parameters[name], rtol=0, atol=0)
    np.testing.assert_array_equal(predict_image(model, sample_tensor, spec),
                                  baseline_before)


def test_experiment_rejects_cross_split_duplicate_content(tmp_path):
    model, spec, dataset, _ = _fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="Identical image content crosses"):
        run_experiment(model, spec, dataset, "features", 0, kind="ablation",
                       sample_size=dataset.count("test"), repeats=1,
                       reference_size=18, cache_dir=None, bootstrap_draws=200)


def test_patching_requires_distinct_matched_donors(tmp_path):
    model, spec, dataset, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="at least four"):
        run_experiment(model, spec, dataset, "features", 0, kind="patching",
                       sample_size=2, repeats=1, reference_size=4)


@pytest.mark.parametrize("architecture", ["resnet18", "densenet121"])
@pytest.mark.parametrize("kind", ["ablation", "patching", "concept"])
def test_saved_pneumonia_models_general_experiment(architecture, kind):
    project = Path(__file__).resolve().parents[2]
    spec_path = project / "model_specs" / f"pneumonia_{architecture}.json"
    if not spec_path.exists():
        pytest.skip("Example spec unavailable")
    spec = read_spec(spec_path)
    if not spec.checkpoint.exists() or not (project / "cache/data/pneumoniamnist_64.npz").exists():
        pytest.skip("Local example checkpoint or data unavailable")
    model = load_model(spec)
    dataset = DatasetHandle({"id": "builtin:pneumoniamnist", "name": "PneumoniaMNIST 64",
                             "group_separated": False})
    layer = "backbone.layer3" if architecture == "resnet18" else "backbone.features.denseblock3"
    sample_size = 4 if kind == "patching" else 2
    raw, report, config = run_experiment(model, spec, dataset, layer, 1,
        kind=kind, sample_size=sample_size, repeats=1, seed=5,
        reference_size=4, channel_count=1, channels=[0] if kind == "patching" else None,
        bootstrap_draws=200, cache_dir=None)
    assert len(raw) == sample_size and report["unique_test_images"] == sample_size
    assert config["factory"] == f"medical_{architecture}"
    assert not model.get_submodule(layer)._forward_hooks
