"""Train a medical head and compute training-only layer statistics.

Example: python -m medcnn.prepare --architecture densenet121 --train-limit 256
"""

import argparse
from contextlib import ExitStack
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .pneumoniamnist import loader
from .data import get_dataset
from .models import load_checkpoint
from .utils import file_hash, save_json, seed_all
from .core import LAYERS, training_statistics
from .features import rank_channels
from .features import capture
from .core import layer_for
import torch


def analysis_dir(run, layer_name):
    run = Path(run)
    target_root = Path("medcnn_runs/analysis") / run.name if run.parent.name == "results" else run
    return target_root / "layers" / layer_name.replace(".", "_")


def prepare_layer(run, layer_name, batch_size=16, threads=4, progress=None):
    run = Path(run)
    model, checkpoint = load_checkpoint(run / "model.pt")
    config = checkpoint["config"]
    seed_all(config["seed"], threads)
    datasets, _ = get_dataset(image_size=config["image_size"],
                            limits=config["limits"], seed=config["seed"], download=True)
    cache_index = run / "feature_cache.json"
    cached = None
    if layer_name == model.layer_name and cache_index.exists():
        entry = json.loads(cache_index.read_text(encoding="utf-8"))["train"]
        cached_path = Path(entry["path"])
        if cached_path.exists() and file_hash(cached_path) == entry["sha256"]:
            with np.load(cached_path) as archive:
                if np.array_equal(archive["ids"], datasets["train"].indices):
                    cached = (archive["intermediate"], archive["labels"], archive["ids"])
    if cached is None:
        mean, ranking, ids = training_statistics(
            model, loader(datasets["train"], batch_size), layer_name, progress)
    else:
        values, labels, ids = cached
        mean, ranking = values.mean(0), rank_channels(values, labels)
        if progress:
            progress(1, 1)
    # Keep the original presentation run and all its artifacts untouched.
    target = analysis_dir(run, layer_name)
    target.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(target / "ranking.csv", index=False)
    np.save(target / "training_mean.npy", mean)
    save_json(target / "provenance.json", {
        "source_split": "train", "n": len(ids), "layer": layer_name,
        "checkpoint_sha256": file_hash(run / "model.pt"),
        "dataset": "pneumoniamnist", "seed": config["seed"],
    })
    return target


@torch.inference_mode()
def prepare_all_layers(run, batch_size=16, threads=4, progress=None):
    """Prepare every supported layer with a single pass over training images."""
    run = Path(run)
    model, checkpoint = load_checkpoint(run / "model.pt")
    config = checkpoint["config"]
    seed_all(config["seed"], threads)
    datasets, _ = get_dataset(image_size=config["image_size"],
                              limits=config["limits"], seed=config["seed"], download=True)
    batches = loader(datasets["train"], batch_size)
    names = LAYERS[model.architecture]
    values = {name: [] for name in names}
    labels, ids = [], []
    with ExitStack() as stack:
        captured = {name: stack.enter_context(capture(layer_for(model, name)))
                    for name in names}
        for number, (x, y, index) in enumerate(batches, 1):
            model(x)
            for name in names:
                values[name].append(captured[name]["activation"].mean((2, 3)).cpu().numpy())
            labels.append(np.asarray(y))
            ids.append(np.asarray(index))
            if progress:
                progress(number, len(batches))
    labels, ids = np.concatenate(labels), np.concatenate(ids)
    checkpoint_hash = file_hash(run / "model.pt")
    targets = []
    for name in names:
        features = np.concatenate(values[name])
        target = analysis_dir(run, name)
        target.mkdir(parents=True, exist_ok=True)
        rank_channels(features, labels).to_csv(target / "ranking.csv", index=False)
        np.save(target / "training_mean.npy", features.mean(0))
        save_json(target / "provenance.json", {
            "source_split": "train", "n": len(ids), "layer": name,
            "checkpoint_sha256": checkpoint_hash, "dataset": "pneumoniamnist",
            "seed": config["seed"],
        })
        targets.append(target)
    return targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=LAYERS, default="resnet18")
    p.add_argument("--output")
    p.add_argument("--layer")
    p.add_argument("--train-limit", type=int)
    p.add_argument("--val-limit", type=int)
    p.add_argument("--test-limit", type=int)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--existing", action="store_true", help="Use an existing checkpoint")
    p.add_argument("--all-layers", action="store_true", help="Prepare every supported layer in one pass")
    args = p.parse_args()
    run = Path(args.output or f"medcnn_runs/{args.architecture}")
    if not args.existing:
        command = [sys.executable, "-m", "medcnn.pipeline", "--stage", "train",
                   "--architecture", args.architecture, "--output", str(run),
                   "--batch-size", str(args.batch_size), "--threads", str(args.threads),
                   "--seed", str(args.seed)]
        for split in ("train", "val", "test"):
            limit = getattr(args, f"{split}_limit")
            if limit is not None:
                command.extend((f"--{split}-limit", str(limit)))
        subprocess.run(command, check=True)
    model, checkpoint = load_checkpoint(run / "model.pt")
    if model.architecture != args.architecture:
        raise ValueError("Checkpoint architecture does not match --architecture")
    layer = args.layer or model.layer_name
    if layer not in LAYERS[model.architecture]:
        raise ValueError(f"Choose one of {LAYERS[model.architecture]}")
    if args.all_layers:
        def show_progress(done, total):
            if done == 1 or done == total or done % 25 == 0:
                print(f"Training batches {done}/{total}", flush=True)
        targets = prepare_all_layers(run, args.batch_size, args.threads, show_progress)
    else:
        targets = [prepare_layer(run, layer, args.batch_size, args.threads)]
    print(f"Training-only statistics saved to {', '.join(map(str, targets))}")


if __name__ == "__main__":
    main()
