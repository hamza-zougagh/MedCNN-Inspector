from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocessing(image_size):
    return {
        "image_size": image_size,
        "dataset_size": 64,
        "channels": "grayscale replicated to RGB",
        "mean": MEAN,
        "std": STD,
        "resize": "bilinear, antialias=True",
        "augmentation": "none",
    }


def stratified_indices(labels, limit=None, seed=42):
    labels = np.asarray(labels).reshape(-1)
    indices = np.arange(len(labels))
    if limit is None or limit >= len(labels):
        return indices
    if limit < 2 or np.min(np.bincount(labels.astype(int))) < 2:
        raise ValueError(
            "Stratified sampling requires >=2 images and >=2 examples per class"
        )
    chosen, _ = train_test_split(
        indices, train_size=limit, stratify=labels, random_state=seed
    )
    return np.sort(chosen)


class MedicalImages(Dataset):
    def __init__(self, raw, positive_label, image_size=128, limit=None, seed=42):
        self.raw = raw
        self.indices = stratified_indices(raw.labels, limit, seed)
        self.labels = (raw.labels.reshape(-1)[self.indices] == positive_label).astype(
            np.int64
        )
        self.transform = image_transform(image_size)

    def __len__(self):
        return len(self.indices)

    def original(self, index):
        return self.raw[int(self.indices[index])][0].convert("L")

    def __getitem__(self, index):
        return (
            self.transform(self.original(index).convert("RGB")),
            int(self.labels[index]),
            int(self.indices[index]),
        )


def load_data(root="cache/data", image_size=128, limits=None, seed=42, download=True):
    from medmnist import INFO, PneumoniaMNIST

    metadata = INFO["pneumoniamnist"]
    mapping = {int(k): v for k, v in metadata["label"].items()}
    positives = [k for k, v in mapping.items() if v.lower() == "pneumonia"]
    if len(mapping) != 2 or len(positives) != 1:
        raise ValueError(f"Unexpected official label metadata: {mapping}")
    Path(root).mkdir(parents=True, exist_ok=True)
    datasets = {}
    for split in ("train", "val", "test"):
        # MedMNIST 3 supports 64. Network errors propagate; never silently substitute data.
        raw = PneumoniaMNIST(split=split, root=root, size=64, download=download)
        datasets[split] = MedicalImages(
            raw, positives[0], image_size, (limits or {}).get(split), seed
        )
    return datasets, {
        "official_labels": mapping,
        "positive_label": positives[0],
        "target_encoding": {"0": "normal", "1": "pneumonia"},
        "preprocessing": preprocessing(image_size),
    }


def loader(dataset, batch_size=16):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def image_transform(image_size):
    from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

    return Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR,
               antialias=True),
        ToTensor(),
        Normalize(MEAN, STD),
    ])


def preprocess_uploaded(image, image_size):
    """Use the exact dataset preprocessing on a user supplied PIL image."""
    return image_transform(image_size)(image.convert("RGB"))
