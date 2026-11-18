"""Run and export a reproducible held-out intervention study.

Example: python -m medcnn.experiment --run medcnn_runs/densenet121 --sample-size 32
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Subset

from .pneumoniamnist import loader, stratified_indices
from .data import get_dataset
from .models import load_checkpoint
from .utils import file_hash, save_json, seed_all
from .core import LAYERS
from .studio import png_bytes, run_studio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--layer")
    p.add_argument("--output", default="medcnn_runs/experiment")
    p.add_argument("--sample-size", type=int, default=64)
    p.add_argument("--budgets", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    run = Path(args.run)
    model, checkpoint = load_checkpoint(run / "model.pt")
    layer = args.layer or model.layer_name
    if layer not in LAYERS[model.architecture]:
        raise ValueError("Unsupported layer")
    analysis_root = (Path("medcnn_runs/analysis") / run.name
                     if run.parent.name == "results" else run)
    analysis = analysis_root / "layers" / layer.replace(".", "_")
    provenance = json.loads((analysis / "provenance.json").read_text(encoding="utf-8"))
    checkpoint_hash = file_hash(run / "model.pt")
    if provenance["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Training statistics belong to another checkpoint")
    ranking = pd.read_csv(analysis / "ranking.csv")
    training_mean = np.load(analysis / "training_mean.npy")
    seed_all(args.seed, args.threads)
    config = checkpoint["config"]
    datasets, _ = get_dataset(image_size=config["image_size"],
                            limits=config["limits"], seed=config["seed"], download=True)
    test = datasets["test"]
    ids = stratified_indices(test.labels, args.sample_size, args.seed)
    out = Path(args.output)
    configuration = {"architecture": model.architecture, "checkpoint_sha256": checkpoint_hash,
        "dataset": "pneumoniamnist", "split": "test", "layer": layer,
        "seed": args.seed, "sample_ids": test.indices[ids].tolist(), "budgets": args.budgets,
        "repeats": args.repeats, "methods": ["zero", "training_mean"],
        "threshold": checkpoint["threshold"]}
    old = out / "config.json"
    if old.exists() and json.loads(old.read_text(encoding="utf-8")) != configuration:
        raise ValueError("Output has a different configuration; choose another directory")
    batches = loader(Subset(test, ids.tolist()), args.batch_size)
    summary, raw = run_studio(model, batches, layer, ranking,
        checkpoint["threshold"], training_mean, args.budgets, args.repeats, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    save_json(old, configuration)
    summary.to_csv(out / "summary.csv", index=False)
    raw.to_csv(out / "predictions.csv", index=False)
    (out / "comparison.png").write_bytes(png_bytes(summary))
    print(f"Saved {len(summary)} conditions and {len(raw)} real predictions to {out}")


if __name__ == "__main__":
    main()
