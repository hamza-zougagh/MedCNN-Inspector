"""Persistent local model and image-dataset catalog.

Only tensor state_dicts and image/CSV data are accepted. Model factories come
from the trusted local registry, never from imported files.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
import torch

from .data import get_dataset
from .utils import save_json
from .workbench import (FACTORIES, ModelSpec, discover_layers, load_local_factories,
                        load_model, logits, preprocess, probabilities, read_spec,
                        device_of)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")
MAX_IMAGE_PIXELS = 25_000_000


def slugify(name):
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise ValueError("Name needs letters or numbers")
    return slug[:64]


def catalog_root(root="workspace_registry"):
    root = Path(root).resolve()
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    return root


def list_models(root="workspace_registry", examples="model_specs"):
    catalog_root(root)
    paths = list(Path(examples).glob("*.json")) + list((Path(root) / "models").glob("*.json"))
    result = []
    for path in sorted(paths):
        try:
            spec = read_spec(path)
            result.append({"id": str(path.resolve()), "name": spec.name,
                           "factory": spec.factory, "classes": list(spec.class_names),
                           "path": path, "ready": spec.checkpoint.exists(),
                           "managed": path.parent.name == "models"})
        except Exception as exc:
            result.append({"id": str(path.resolve()), "name": path.stem,
                           "path": path, "ready": False, "error": str(exc),
                           "managed": path.parent.name == "models"})
    return result


def register_model(name, factory, checkpoint, class_names, input_size,
                   input_channels=3, mean=None, std=None, output_kind="logits",
                   root="workspace_registry", copy_checkpoint=True, metadata=None):
    """Validate a trusted factory and restricted checkpoint before publishing a spec."""
    load_local_factories()
    if factory not in FACTORIES:
        raise ValueError(f"Factory {factory!r} is not registered locally")
    root = catalog_root(root)
    slug = slugify(name)
    target = root / "models" / f"{slug}.json"
    if target.exists():
        raise ValueError("A model with this name is already registered")
    source = Path(checkpoint).resolve()
    if not source.is_file():
        raise ValueError("State-dict checkpoint file not found")
    if source.stat().st_size > 1_500_000_000:
        raise ValueError("Checkpoint exceeds the 1.5 GB local import limit")
    classes = tuple(part.strip() for part in class_names)
    channels = int(input_channels)
    mean = tuple(float(v) for v in (mean if mean is not None else ([.485, .456, .406] if channels == 3 else [0.])))
    std = tuple(float(v) for v in (std if std is not None else ([.229, .224, .225] if channels == 3 else [1.])))
    spec = ModelSpec(name, factory, source, classes, tuple(map(int, input_size)),
                     channels, mean, std, output_kind)
    # read_spec applies the same schema checks as dashboard imports.
    candidate = root / "models" / f".{slug}.candidate.json"
    payload = {"name": name, "factory": factory, "checkpoint": str(source),
               "class_names": list(classes), "input_size": list(spec.input_size),
               "input_channels": channels, "mean": list(mean), "std": list(std),
               "output_kind": output_kind, "metadata": metadata or {}}
    save_json(candidate, payload)
    try:
        spec = read_spec(candidate)
        model = load_model(spec)
        sample = torch.zeros((1, channels, *spec.input_size))
        with torch.inference_mode():
            prediction = probabilities(logits(model, sample), spec)
        if prediction.shape != (1, len(classes)) or not discover_layers(model, sample):
            raise ValueError("The checkpoint is not a compatible 2D classifier")
    finally:
        candidate.unlink(missing_ok=True)
    if copy_checkpoint:
        destination = root / "checkpoints" / f"{slug}{source.suffix or '.pt'}"
        if destination.exists():
            raise ValueError("Managed checkpoint target already exists")
        shutil.copy2(source, destination)
        payload["checkpoint"] = f"../checkpoints/{destination.name}"
    save_json(target, payload)
    return target


def _safe_extract(archive, destination):
    with zipfile.ZipFile(archive) as packed:
        entries = packed.infolist()
        if len(entries) > 100_000 or sum(item.file_size for item in entries) > 2_000_000_000:
            raise ValueError("ZIP exceeds local file-count or 2 GB expanded-size limit")
        for item in entries:
            path = PurePosixPath(item.filename)
            if ("\\" in item.filename or path.is_absolute() or ".." in path.parts or not path.parts or
                re.match(r"^[A-Za-z]:", item.filename) or
                stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError(f"Unsafe ZIP entry: {item.filename}")
            resolved = (destination / Path(*path.parts)).resolve()
            if destination.resolve() not in resolved.parents and resolved != destination.resolve():
                raise ValueError(f"ZIP entry leaves import directory: {item.filename}")
        packed.extractall(destination)


def _folder_rows(folder):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError("Image folder not found")
    rows = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = path.relative_to(folder).parts
        if len(parts) < 2:
            raise ValueError("Folder images must be under class-name subfolders")
        first = parts[0].lower()
        split = (("val" if first == "validation" else first)
                 if first in ("train", "val", "validation", "test") else "")
        if split and len(parts) < 3:
            raise ValueError("Split folders need class-name subfolders")
        label = parts[1] if split else parts[0]
        rows.append({"image_path": str(path), "label": label, "split": split,
                     "group_id": "", "image_id": path.relative_to(folder).as_posix()})
    if not rows:
        raise ValueError("No supported image files found")
    return pd.DataFrame(rows)


def _csv_rows(csv_path, image_root, image_column, label_column,
              split_column=None, group_column=None):
    image_root = Path(image_root).resolve()
    if not image_root.is_dir():
        raise ValueError("Image root folder not found")
    source = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required = [image_column, label_column]
    if split_column:
        required.append(split_column)
    if group_column:
        required.append(group_column)
    if any(column not in source for column in required):
        raise ValueError(f"CSV missing required columns: {required}")
    rows = []
    for row in source.itertuples(index=False, name=None):
        record = dict(zip(source.columns, row))
        if "\\" in record[image_column]:
            raise ValueError(f"Invalid image path in CSV: {record[image_column]}")
        relative = Path(record[image_column])
        path = (image_root / relative).resolve()
        if image_root not in path.parents or not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Invalid image path in CSV: {record[image_column]}")
        rows.append({"image_path": str(path), "label": record[label_column].strip(),
                     "split": record[split_column].strip().lower() if split_column else "",
                     "group_id": record[group_column].strip() if group_column else "",
                     "image_id": relative.as_posix()})
    if not rows:
        raise ValueError("CSV has no image rows")
    return pd.DataFrame(rows)


def _assign_splits(frame, mode, seed, train_fraction, val_fraction):
    frame = frame.copy()
    if frame.image_path.duplicated().any() or frame.image_id.duplicated().any():
        raise ValueError("Duplicate image paths or IDs are not allowed")
    if frame.label.eq("").any() or frame.label.nunique() < 2:
        raise ValueError("Dataset needs at least two nonempty labels")
    if len(frame) < 6:
        raise ValueError("Need at least six images for three splits")
    if mode == "existing":
        frame["split"] = frame["split"].replace({"validation": "val"})
        if set(frame["split"]) != set(SPLITS):
            raise ValueError("Existing splits must contain train, val, and test")
    elif mode == "generate":
        if not (0 < train_fraction < 1 and 0 < val_fraction < 1 - train_fraction):
            raise ValueError("Invalid train/validation fractions")
        positions = np.arange(len(frame))
        groups = frame.group_id.to_numpy()
        if np.any(groups != ""):
            if np.any(groups == ""):
                raise ValueError("Every image needs a group ID when group splitting is enabled")
            if len(set(groups)) < 3:
                raise ValueError("Need at least three distinct groups")
            try:
                splitter = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=seed)
                train, remainder = next(splitter.split(positions, groups=groups))
                second = GroupShuffleSplit(n_splits=1,
                    train_size=val_fraction / (1 - train_fraction), random_state=seed + 1)
                val_local, test_local = next(second.split(remainder, groups=groups[remainder]))
                val, test = remainder[val_local], remainder[test_local]
            except ValueError as exc:
                raise ValueError(f"Cannot form three group-disjoint splits: {exc}") from exc
        else:
            try:
                train, remainder = train_test_split(positions, train_size=train_fraction,
                    stratify=frame.label, random_state=seed)
                val, test = train_test_split(remainder,
                    train_size=val_fraction / (1 - train_fraction),
                    stratify=frame.label.iloc[remainder], random_state=seed + 1)
            except ValueError as exc:
                raise ValueError(f"Cannot generate stratified splits: {exc}") from exc
        frame["split"] = ""
        frame.loc[train, "split"] = "train"
        frame.loc[val, "split"] = "val"
        frame.loc[test, "split"] = "test"
    else:
        raise ValueError("Choose existing or generated splits")
    if frame.group_id.ne("").any():
        overlap = frame.groupby("group_id")["split"].nunique()
        if (overlap > 1).any():
            raise ValueError("Patient/group identifiers cross dataset splits")
    return frame


def _image_content_digest(image):
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("Image exceeds the 25-megapixel import limit")
    rgb = image.convert("RGB")
    return sha256(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()).hexdigest()


def import_dataset(name, kind, source, root="workspace_registry", image_root=None,
                   image_column="image", label_column="label", split_column=None,
                   group_column=None, split_mode="generate", seed=42,
                   train_fraction=.7, val_fraction=.15, progress=None):
    """Persist image references and leakage-checked splits. ZIP images are copied."""
    root = catalog_root(root)
    slug = slugify(name)
    dataset_root = (root / "datasets").resolve()
    target = dataset_root / slug
    if target.exists():
        raise ValueError("A dataset directory with this name already exists; use another name")
    stage = Path(tempfile.mkdtemp(prefix=f".{slug}-", dir=dataset_root)).resolve()
    try:
        if kind == "folder":
            frame = _folder_rows(source)
        elif kind == "zip":
            images = stage / "images"
            images.mkdir(parents=True, exist_ok=True)
            _safe_extract(source, images)
            csv_files = [p for p in images.rglob("*.csv")]
            if csv_files:
                if len(csv_files) > 1:
                    raise ValueError("ZIP contains multiple CSV manifests")
                frame = _csv_rows(csv_files[0], images, image_column, label_column,
                                  split_column, group_column)
            else:
                frame = _folder_rows(images)
        elif kind == "csv":
            if image_root is None:
                raise ValueError("CSV import needs an image root folder")
            frame = _csv_rows(source, image_root, image_column, label_column,
                              split_column, group_column)
        else:
            raise ValueError("Choose folder, zip, or csv")
        frame = _assign_splits(frame, split_mode, int(seed), train_fraction, val_fraction)
        digests = []
        for number, path in enumerate(frame.image_path, 1):
            try:
                with Image.open(path) as image:
                    digests.append(_image_content_digest(image))
            except Exception as exc:
                raise ValueError(f"Unreadable image {path}: {exc}") from exc
            if progress and (number % 10 == 0 or number == len(frame)):
                progress(number, len(frame))
        frame["content_sha256"] = digests
        if (frame.groupby("content_sha256").split.nunique() > 1).any():
            raise ValueError("Identical image content crosses dataset splits")
        frame = frame.sort_values(["split", "label", "image_id"]).reset_index(drop=True)
        if kind == "zip":
            frame["image_path"] = frame.image_path.map(
                lambda value: Path(value).relative_to(stage).as_posix())
        frame.to_csv(stage / "samples.csv", index=False)
        source_description = str(source) if isinstance(source, (str, Path)) else "uploaded ZIP"
        if kind == "csv" and Path(source).name == ".uploaded_labels.csv":
            shutil.copy2(source, stage / "source_labels.csv")
            source_description = str(target / "source_labels.csv")
        manifest = {"name": name, "slug": slug, "kind": kind,
            "source": source_description,
            "classes": sorted(frame.label.unique().tolist()),
            "counts": {split: int((frame.split == split).sum()) for split in SPLITS},
            "group_separated": bool(frame.group_id.ne("").any()),
            "split_mode": split_mode, "split_seed": int(seed) if split_mode == "generate" else None,
            "train_fraction": train_fraction if split_mode == "generate" else None,
            "val_fraction": val_fraction if split_mode == "generate" else None,
            "created_utc": datetime.now(timezone.utc).isoformat()}
        save_json(stage / "manifest.json", manifest)
        stage.rename(target)
        return target / "manifest.json"
    finally:
        if stage.exists():
            if stage.parent != dataset_root:
                raise RuntimeError("Import staging directory escaped the dataset registry")
            shutil.rmtree(stage)


def list_datasets(root="workspace_registry"):
    root = catalog_root(root)
    examples = [{"id": "builtin:pneumoniamnist", "name": "PneumoniaMNIST 64",
                 "classes": ["normal", "pneumonia"], "counts": {"train": 4708,
                 "val": 524, "test": 624}, "ready": True, "managed": False,
                 "path": None, "group_separated": False}]
    for path in sorted((root / "datasets").glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            frame = pd.read_csv(path.parent / "samples.csv", dtype=str, keep_default_na=False)
            ready = all((path.parent / p).is_file() for p in frame.image_path)
            examples.append({"id": str(path.resolve()), "name": manifest["name"],
                "classes": manifest["classes"], "counts": manifest["counts"],
                "ready": ready, "managed": True, "path": path,
                "group_separated": manifest["group_separated"]})
        except Exception as exc:
            examples.append({"id": str(path.resolve()), "name": path.parent.name,
                "ready": False, "managed": True, "path": path, "error": str(exc)})
    return examples


class DatasetHandle:
    def __init__(self, entry):
        self.entry = entry
        if entry["id"] == "builtin:pneumoniamnist":
            self.kind = "pneumoniamnist"
            self.datasets, _ = get_dataset(download=True)
            self.classes = ("normal", "pneumonia")
        else:
            self.kind = "imported"
            self.frame = pd.read_csv(entry["path"].parent / "samples.csv",
                                     dtype=str, keep_default_na=False)
            self.classes = tuple(entry["classes"])
            required = {"image_path", "image_id", "label", "split", "group_id"}
            if not required <= set(self.frame):
                raise ValueError("Imported sample index is missing required columns")
            if (set(self.frame.split) != set(SPLITS) or
                self.frame.image_path.duplicated().any() or
                self.frame.image_id.duplicated().any() or
                set(self.frame.label) != set(self.classes)):
                raise ValueError("Imported sample index no longer matches its manifest")
            if any(int((self.frame.split == split).sum()) != entry["counts"][split]
                   for split in SPLITS):
                raise ValueError("Imported split counts changed since registration")
            if entry.get("group_separated"):
                if (self.frame.group_id.eq("").any() or
                    (self.frame.groupby("group_id").split.nunique() > 1).any()):
                    raise ValueError("Patient/group identifiers now cross dataset splits")
            self.by_split = {split: self.frame[self.frame.split == split].reset_index(drop=True)
                             for split in SPLITS}

    def count(self, split):
        if split not in SPLITS:
            raise ValueError("Choose train, val, or test")
        return len(self.datasets[split]) if self.kind == "pneumoniamnist" else len(self.by_split[split])

    def labels(self, split):
        if split not in SPLITS:
            raise ValueError("Choose train, val, or test")
        if self.kind == "pneumoniamnist":
            return [self.classes[int(value)] for value in self.datasets[split].labels]
        return self.by_split[split].label.tolist()

    def item(self, split, index):
        if split not in SPLITS or not 0 <= index < self.count(split):
            raise IndexError("Image position outside split")
        if self.kind == "pneumoniamnist":
            dataset = self.datasets[split]
            return (f"{split}:{dataset.indices[index]}", dataset.original(index),
                    self.classes[int(dataset.labels[index])])
        row = self.by_split[split].iloc[index]
        with Image.open((self.entry["path"].parent / row.image_path).resolve()) as image:
            copy = image.copy()
        if "content_sha256" in self.frame and _image_content_digest(copy) != row.content_sha256:
            raise ValueError(f"Image content changed since import: {row.image_id}; reimport the dataset")
        return f"{split}:{row.image_id}", copy, row.label


def validate_pair(spec, dataset, model):
    unknown = set(dataset.classes) - set(spec.class_names)
    if unknown:
        raise ValueError(f"Dataset labels missing from model classes: {sorted(unknown)}")
    warnings = []
    if set(spec.class_names) - set(dataset.classes):
        warnings.append("Some model classes have no dataset examples")
    if not dataset.entry.get("group_separated", False):
        warnings.append("No patient/group IDs were provided for an independent leakage audit")
    checked = []
    label_counts = {}
    for split in SPLITS:
        if dataset.count(split) == 0:
            raise ValueError(f"Dataset split {split} is empty")
        labels = pd.Series(dataset.labels(split)).value_counts().to_dict()
        label_counts[split] = {str(label): int(count) for label, count in labels.items()}
        missing_labels = set(dataset.classes) - set(labels)
        if missing_labels:
            warnings.append(f"{split} split lacks dataset classes: {sorted(missing_labels)}")
        image_id, image, label = dataset.item(split, 0)
        tensor = preprocess(image, spec)
        if tuple(tensor.shape) != (spec.input_channels, *spec.input_size):
            raise ValueError(f"Preprocessed input shape mismatch in {split}")
        with torch.inference_mode():
            result = probabilities(logits(model, tensor[None].to(device_of(model))), spec)
        checked.append({"split": split, "image_id": image_id, "label": label,
                        "source_mode": image.mode, "source_size": image.size,
                        "model_input_shape": tuple(tensor.shape),
                        "output_classes": result.shape[1]})
        expected_mode = "L" if spec.input_channels == 1 else "RGB"
        if image.mode != expected_mode:
            warnings.append(f"Input images are converted to {expected_mode}")
        if image.size != (spec.input_size[1], spec.input_size[0]):
            warnings.append("Input images are resized to the model's configured dimensions")
    metadata = spec.metadata or {}
    training_dataset = str(metadata.get("training_dataset", "")).strip()
    preprocessing_confirmed = metadata.get("preprocessing_confirmed") is True
    class_order_confirmed = metadata.get("class_order_confirmed") is True
    if not preprocessing_confirmed:
        warnings.append("Training-time preprocessing is not confirmed")
    if not class_order_confirmed:
        warnings.append("Training-time class order is not confirmed")
    if not training_dataset:
        warnings.append("Training-data provenance was not supplied")
    relationship = ("dataset name appears in provenance declaration"
                    if training_dataset and dataset.entry["name"].lower() in training_dataset.lower()
                    else "not established")
    if relationship == "not established":
        warnings.append("Relation between this evaluation dataset and declared training data is not established")
    if not dataset.entry.get("group_separated", False):
        warnings.append("Patient-level separation cannot be audited without supplied group IDs")
    checklist = {
        "preprocessing": "user-declared" if preprocessing_confirmed else "unknown",
        "class_order": "user-declared" if class_order_confirmed else "unknown",
        "training_data_provenance": training_dataset or "unknown",
        "training_to_evaluation_relation": relationship,
        "patient_group_separation": ("supplied group IDs are split-disjoint"
            if dataset.entry.get("group_separated", False) else "unknown"),
        "external_scientific_validity": "not assessed"}
    return {"model": spec.name, "dataset": dataset.entry["name"],
            "counts": {split: dataset.count(split) for split in SPLITS},
            "label_counts": label_counts,
            "checked": checked, "warnings": sorted(set(warnings)), "compatible": True,
            "technical_compatibility": "passed runtime shape, label, and output checks",
            "scientific_checklist": checklist}


def forget_registration(path, root="workspace_registry"):
    """Remove only managed registration files; leave weights and image files."""
    root = catalog_root(root)
    path = Path(path).resolve()
    models = (root / "models").resolve()
    datasets = (root / "datasets").resolve()
    if path.parent == models and path.suffix == ".json":
        path.unlink()
    elif path.parent.parent == datasets and path.name == "manifest.json":
        path.unlink()
    else:
        raise ValueError("Only managed registrations can be forgotten")
