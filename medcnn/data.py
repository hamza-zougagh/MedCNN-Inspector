"""Dataset adapter registry. Adapters return official train/val/test splits plus metadata."""

from .pneumoniamnist import load_data


DATASETS = {"pneumoniamnist": load_data}


def register_dataset(name, adapter):
    if name in DATASETS:
        raise ValueError(f"Dataset already registered: {name}")
    DATASETS[name] = adapter


def get_dataset(name="pneumoniamnist", **kwargs):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASETS[name](**kwargs)
