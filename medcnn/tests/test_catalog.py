import csv
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from medcnn.catalog import (DatasetHandle, import_dataset, list_datasets,
                            list_models, register_model, validate_pair)
from medcnn.workbench import FACTORIES, load_model, read_spec, register_factory


class CatalogCNN(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 3, padding=1)
        self.classifier = nn.Linear(4, classes)

    def forward(self, x):
        return self.classifier(self.features(x).mean((2, 3)))


@pytest.fixture
def factory():
    if "catalog_test_cnn" not in FACTORIES:
        register_factory("catalog_test_cnn", CatalogCNN)
    return "catalog_test_cnn"


def _image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((20, 20, 3), value, dtype=np.uint8)).save(path)


def test_model_registration_and_pair_validation(tmp_path, factory):
    checkpoint = tmp_path / "weights.pt"
    torch.save(CatalogCNN().state_dict(), checkpoint)
    root = tmp_path / "registry"
    spec_path = register_model("My classifier", factory, checkpoint, ["a", "b"],
        (32, 32), mean=(0, 0, 0), std=(1, 1, 1), root=root)
    spec = read_spec(spec_path)
    assert spec.checkpoint.parent == root / "checkpoints"
    assert list_models(root, examples=tmp_path / "empty")[0]["ready"]
    with pytest.raises(ValueError):
        register_model("My classifier", factory, checkpoint, ["a", "b"],
                       (32, 32), root=root)
    images = tmp_path / "images"
    for label, value in (("a", 40), ("b", 170)):
        for index in range(10):
            _image(images / label / f"{index}.png", value + index)
    manifest = import_dataset("Folder study", "folder", images, root=root)
    entry = next(e for e in list_datasets(root) if e["path"] == manifest)
    handle = DatasetHandle(entry)
    report = validate_pair(spec, handle, load_model(spec))
    assert report["compatible"] and sum(report["counts"].values()) == 20
    assert set(report["label_counts"]) == {"train", "val", "test"}
    assert {handle.item(split, 0)[2] for split in ("train", "val", "test")} <= {"a", "b"}
    assert report["technical_compatibility"].startswith("passed")
    assert report["scientific_checklist"]["preprocessing"] == "unknown"
    assert report["scientific_checklist"]["patient_group_separation"] == "unknown"
    other = register_model("Wrong classes", factory, checkpoint, ["normal", "pneumonia"],
        (32, 32), mean=(0, 0, 0), std=(1, 1, 1), root=root)
    with pytest.raises(ValueError, match="labels missing"):
        validate_pair(read_spec(other), handle, load_model(read_spec(other)))
    declared = register_model("Declared provenance", factory, checkpoint, ["a", "b"],
        (32, 32), mean=(0, 0, 0), std=(1, 1, 1), root=root,
        metadata={"training_dataset": "Local study X",
                  "preprocessing_confirmed": True, "class_order_confirmed": True})
    declared_spec = read_spec(declared)
    checklist = validate_pair(declared_spec, handle,
                              load_model(declared_spec))["scientific_checklist"]
    assert checklist["preprocessing"] == "user-declared"
    assert checklist["training_data_provenance"] == "Local study X"


def test_csv_group_split_and_existing_group_leakage(tmp_path):
    images = tmp_path / "images"
    rows = []
    for group in range(8):
        for label in ("a", "b"):
            name = f"g{group}_{label}.png"
            _image(images / name, group * 20)
            rows.append({"image": name, "label": label, "patient": f"p{group}",
                         "split": "train" if group < 6 else "test"})
    source = tmp_path / "labels.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    root = tmp_path / "registry"
    manifest = import_dataset("Grouped", "csv", source, root=root, image_root=images,
        group_column="patient", split_mode="generate", seed=19)
    entry = next(e for e in list_datasets(root) if e["path"] == manifest)
    assert entry["group_separated"]
    import pandas as pd
    frame = pd.read_csv(manifest.parent / "samples.csv")
    assert frame.groupby("group_id").split.nunique().max() == 1
    assert all(frame.split.value_counts().reindex(("train", "val", "test"), fill_value=0) > 0)
    with pytest.raises(ValueError, match="Existing splits"):
        import_dataset("Bad existing", "csv", source, root=root, image_root=images,
            split_column="split", group_column="patient", split_mode="existing")
    rows[0]["split"] = "val"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="cross dataset splits"):
        import_dataset("Leaky existing", "csv", source, root=root, image_root=images,
            split_column="split", group_column="patient", split_mode="existing")


def test_zip_import_and_path_traversal_rejection(tmp_path):
    source = tmp_path / "safe.zip"
    with zipfile.ZipFile(source, "w") as packed:
        for label, offset in (("a", 10), ("b", 100)):
            for number in range(10):
                image = tmp_path / f"{label}_{number}.png"
                _image(image, offset + number)
                packed.write(image, f"{label}/{number}.png")
    manifest = import_dataset("Archive", "zip", source, root=tmp_path / "registry")
    assert manifest.exists()
    assert (manifest.parent / "images/a/0.png").exists()
    entry = next(e for e in list_datasets(tmp_path / "registry") if e["path"] == manifest)
    assert entry["ready"]
    assert DatasetHandle(entry).item("test", 0)[1].size == (20, 20)
    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as packed:
        packed.writestr("../outside.png", image.read_bytes())
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        import_dataset("Unsafe", "zip", unsafe, root=tmp_path / "registry")
    assert not (tmp_path / "outside.png").exists()
    assert not (tmp_path / "registry/datasets/unsafe").exists()
    assert not list((tmp_path / "registry/datasets").glob(".unsafe-*"))
    backslash = tmp_path / "backslash.zip"
    with zipfile.ZipFile(backslash, "w") as packed:
        packed.writestr("..\\outside.png", image.read_bytes())
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        import_dataset("Backslash", "zip", backslash, root=tmp_path / "registry")


def test_csv_path_cannot_leave_image_root(tmp_path):
    import pandas as pd
    images = tmp_path / "images"
    images.mkdir()
    _image(tmp_path / "outside.png", 10)
    labels = tmp_path / "labels.csv"
    pd.DataFrame([{"image": "../outside.png", "label": "a"}]).to_csv(labels, index=False)
    with pytest.raises(ValueError, match="Invalid image path"):
        import_dataset("Outside", "csv", labels, root=tmp_path / "registry",
                       image_root=images)
    pd.DataFrame([{"image": "..\\outside.png", "label": "a"}]).to_csv(labels, index=False)
    with pytest.raises(ValueError, match="Invalid image path"):
        import_dataset("Outside backslash", "csv", labels,
                       root=tmp_path / "registry", image_root=images)


def test_import_size_limits_leave_no_registration(tmp_path, monkeypatch):
    import medcnn.catalog as catalog
    images = tmp_path / "images"
    for split in ("train", "val", "test"):
        for label in ("a", "b"):
            _image(images / split / label / "0.png", 42 if label == "a" else 120)
    monkeypatch.setattr(catalog, "MAX_IMAGE_PIXELS", 300)
    with pytest.raises(ValueError, match="25-megapixel import limit"):
        import_dataset("Oversized", "folder", images, root=tmp_path / "registry",
                       split_mode="existing")
    assert not (tmp_path / "registry/datasets/oversized").exists()
    archive = tmp_path / "count.zip"
    with zipfile.ZipFile(archive, "w") as packed:
        packed.writestr("a/0.png", (images / "train/a/0.png").read_bytes())
    original = zipfile.ZipFile.infolist
    monkeypatch.setattr(zipfile.ZipFile, "infolist",
                        lambda self: original(self) * 100_001)
    with pytest.raises(ValueError, match="file-count"):
        import_dataset("Too many files", "zip", archive, root=tmp_path / "registry")
    assert not (tmp_path / "registry/datasets/too-many-files").exists()


def test_import_rejects_duplicate_content_across_splits_and_late_file_changes(tmp_path):
    source = tmp_path / "images"
    for split, base in (("train", 10), ("val", 70), ("test", 130)):
        for label, offset in (("a", 0), ("b", 20)):
            for number in range(3):
                _image(source / split / label / f"{number}.png", base + offset + number)
    target = tmp_path / "registry"
    (source / "test/a/0.png").write_bytes((source / "train/a/0.png").read_bytes())
    with pytest.raises(ValueError, match="Identical image content crosses"):
        import_dataset("Duplicate", "folder", source, root=target, split_mode="existing")
    assert not (target / "datasets/duplicate").exists()
    _image(source / "test/a/0.png", 130)
    manifest = import_dataset("Valid", "folder", source, root=target, split_mode="existing")
    entry = next(row for row in list_datasets(target) if row["path"] == manifest)
    dataset = DatasetHandle(entry)
    assert dataset.item("test", 0)[1].size == (20, 20)
    changed = source / "test/a/0.png"
    _image(changed, 200)
    with pytest.raises(ValueError, match="changed since import"):
        dataset.item("test", 0)


def test_import_forms_and_paired_analysis_in_streamlit(tmp_path, monkeypatch, factory):
    from streamlit.testing.v1 import AppTest

    project = Path(__file__).resolve().parents[2]
    script = project / "medcnn_app.py"
    if not script.exists():
        script = project / "app.py"
    checkpoint = tmp_path / "toy.pt"
    torch.save(CatalogCNN().state_dict(), checkpoint)
    images = tmp_path / "images"
    for label, value in (("a", 40), ("b", 170)):
        for index in range(10):
            _image(images / label / f"{index}.png", value + index)
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(script, default_timeout=120).run()
    assert not app.exception
    next(w for w in app.text_input if w.label == "Model name").set_value("Toy CNN")
    factory_choice = next(w for w in app.selectbox if w.label.startswith("2D CNN architecture"))
    factory_choice.set_value(factory)
    next(w for w in app.text_input if w.label == "Local checkpoint path").set_value(str(checkpoint))
    next(w for w in app.text_input if w.label.startswith("Class names")).set_value("a, b")
    next(w for w in app.number_input if w.label == "Input height").set_value(32)
    next(w for w in app.number_input if w.label == "Input width").set_value(32)
    next(w for w in app.button if w.label == "Validate and add model").click().run()
    assert not app.exception
    assert (tmp_path / "workspace_registry/models/toy-cnn.json").exists()

    next(w for w in app.text_input if w.label == "Dataset name").set_value("Toy images")
    next(w for w in app.text_input if w.label == "Image folder path").set_value(str(images))
    next(w for w in app.button if w.label == "Validate and import dataset").click().run()
    assert not app.exception
    assert (tmp_path / "workspace_registry/datasets/toy-images/manifest.json").exists()
    assert any("Ready:" in item.value for item in app.success)
    for page in ("Model Inspector", "Feature Discovery", "Activation Patching",
                 "Concept Explorer", "Feature Interaction Lab", "Dataset Experiments"):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        action = {"Feature Discovery": "Extract activations",
                  "Activation Patching": "Patch and compare",
                  "Feature Interaction Lab": "Measure individual and paired effects",
                  "Dataset Experiments": "Run dataset experiment"}.get(page)
        if action:
            if page == "Dataset Experiments":
                next(w for w in app.slider if w.label == "Test images per trial").set_value(4).run()
                next(w for w in app.slider if w.label == "Independent seeded trials").set_value(2).run()
            next(w for w in app.button if w.label == action).click().run()
            assert not app.exception, (page, action)
            if page == "Feature Interaction Lab":
                next(w for w in app.radio if w.label == "Replacement").set_value(
                    "training_mean").run()
                assert not app.exception
                next(w for w in app.button if w.label == action).click().run()
                assert not app.exception
            if page == "Dataset Experiments":
                assert any("Measured" in w.value for w in app.success)
                for experiment_type in ("patching", "concept"):
                    next(w for w in app.selectbox if w.label == "Experiment type").set_value(
                        experiment_type).run()
                    assert not app.exception
                    next(w for w in app.button if w.label == action).click().run()
                    assert not app.exception, experiment_type
                    assert any("Measured" in w.value for w in app.success)
                saved = list((tmp_path / "workspace_registry/experiments").glob("*/config.json"))
                assert len(saved) == 3
                assert all((path.parent / "per_image.csv").exists() and
                           (path.parent / "comparison.png").exists() for path in saved)
