"""Copy a verified presentation checkpoint into an independent MedCNN run.

The source is read only. Example:
python -m medcnn.import_legacy --source results/resnet18 --target medcnn_runs/resnet18_full
"""

import argparse
import json
import shutil
from pathlib import Path

from .models import load_checkpoint
from .utils import file_hash, save_json


def import_run(source, target):
    source, target = Path(source), Path(target)
    model, checkpoint = load_checkpoint(source / "model.pt")
    if model.architecture != "resnet18":
        raise ValueError("The verified presentation import must be ResNet-18")
    target.mkdir(parents=True, exist_ok=True)
    original_hash = file_hash(source / "model.pt")
    existing = target / "model.pt"
    if existing.exists() and file_hash(existing) != original_hash:
        raise ValueError("Target already contains a different checkpoint")
    for name in ("model.pt", "config.json", "metadata.json", "test_metrics.json",
                 "validation_metrics.json", "predictions.csv"):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    old_index = source / "feature_cache.json"
    if old_index.exists():
        entry = json.loads(old_index.read_text(encoding="utf-8"))["train"]
        original_cache = source.parents[1] / entry["path"]
        if original_cache.exists() and file_hash(original_cache) == entry["sha256"]:
            cache = Path("medcnn_cache/features")
            cache.mkdir(parents=True, exist_ok=True)
            copied = cache / f"imported_{original_hash[:16]}_train.npz"
            shutil.copy2(original_cache, copied)
            entry["path"] = copied.as_posix()
            save_json(target / "feature_cache.json", {"train": entry})
    save_json(target / "import.json", {"source_checkpoint_sha256": original_hash,
                                      "architecture": model.architecture,
                                      "source": str(source)})
    print(f"Imported verified checkpoint to {target}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="results/resnet18")
    p.add_argument("--target", default="medcnn_runs/resnet18_full")
    args = p.parse_args()
    import_run(args.source, args.target)


if __name__ == "__main__":
    main()
