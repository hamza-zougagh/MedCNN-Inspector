"""Train a medical head using official splits: python -m medcnn.pipeline --help."""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from .pneumoniamnist import load_data, loader, stratified_indices
from .evaluate import metrics, select_threshold
from .features import extract, rank_channels
from .interventions import run_ablations
from .models import MedicalCNN, load_checkpoint
from .probes import fit_probe
from .report import write_report
from .train import fit_linear, install_head
from .utils import file_hash, fingerprint, save_json, seed_all
from .visualization import save_analysis, save_curves, save_examples


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=["download", "train", "features", "ablate", "all"],
        default="all",
    )
    p.add_argument(
        "--architecture", choices=["resnet18", "densenet121"], default="resnet18"
    )
    p.add_argument("--output", default="medcnn_runs/resnet18")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--train-limit", type=int)
    p.add_argument("--val-limit", type=int)
    p.add_argument("--test-limit", type=int)
    p.add_argument("--ablation-size", type=int, default=128)
    p.add_argument("--budgets", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace checkpoint and cached results",
    )
    return p


def main():
    started = time.perf_counter()
    args = parser().parse_args()
    if args.repeats < 2 or args.image_size < 32 or args.ablation_size < 2:
        raise ValueError("Need >=2 random repeats, image size >=32, ablation size >=2")
    try:
        versions = {
            package: importlib.metadata.version(package)
            for package in ("torch", "torchvision", "medmnist", "scikit-learn", "numpy")
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install requirements.txt in this interpreter; "
            "see README.md for CPU setup. No medical experiment was run."
        ) from None
    seed_all(args.seed, args.threads)
    config = {
        "architecture": args.architecture,
        "seed": args.seed,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "limits": {
            "train": args.train_limit,
            "val": args.val_limit,
            "test": args.test_limit,
        },
        "ablation_size": args.ablation_size,
        "budgets": args.budgets,
        "repeats": args.repeats,
        "weights": "IMAGENET1K_V1",
        "versions": versions,
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "pipeline_version": 1,
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    old = out / "config.json"
    if old.exists() and json.loads(old.read_text()) != config and not args.force:
        raise ValueError(
            "Output configuration differs. Use a new --output or explicitly --force."
        )
    if args.force:
        for name in (
            "model.pt",
            "ranking.csv",
            "ablation_summary.csv",
            "ablation_predictions.csv",
            "report.md",
            "manifest.json",
        ):
            (out / name).unlink(missing_ok=True)
    save_json(old, config)
    datasets, metadata = load_data(
        image_size=args.image_size, limits=config["limits"], seed=args.seed
    )
    save_json(out / "metadata.json", metadata)
    save_json(
        out / "class_distribution.json",
        {
            s: {
                "normal": int((d.labels == 0).sum()),
                "pneumonia": int((d.labels == 1).sum()),
            }
            for s, d in datasets.items()
        },
    )
    if args.stage == "download":
        return
    checkpoint_path = out / "model.pt"
    model = (
        load_checkpoint(checkpoint_path)[0]
        if checkpoint_path.exists()
        else MedicalCNN(args.architecture)
    )
    cache = Path("medcnn_cache/features")
    cache.mkdir(parents=True, exist_ok=True)
    features = {}
    cache_index = {}
    for split, dataset in datasets.items():
        key = fingerprint(
            {
                "config": config,
                "metadata": metadata,
                "split": split,
                "ids": dataset.indices.tolist(),
            }
        )
        path = cache / f"{key}.npz"
        if path.exists() and not args.force:
            with np.load(path) as archive:
                features[split] = dict(archive)
        else:
            print(f"Extracting {split}: {len(dataset)} images", flush=True)
            features[split] = extract(model, loader(dataset, args.batch_size))
            np.savez_compressed(path, **features[split])
        cache_index[split] = {
            "path": path.as_posix(),
            "sha256": file_hash(path),
            "n": len(dataset),
            "pooled_shape": list(features[split]["pooled"].shape),
            "intermediate_shape": list(features[split]["intermediate"].shape),
        }
    save_json(out / "feature_cache.json", cache_index)
    train, val, test = (features[s] for s in ("train", "val", "test"))
    if not checkpoint_path.exists():
        estimator, threshold, selection = fit_linear(
            (train["pooled"], train["labels"]), (val["pooled"], val["labels"])
        )
        install_head(model, estimator)
        # Assert the sklearn-to-PyTorch fold preserves logits.
        with torch.inference_mode():
            folded = model.head(torch.from_numpy(val["pooled"])).squeeze(-1).numpy()
        np.testing.assert_allclose(
            folded, estimator.decision_function(val["pooled"]), atol=1e-4, rtol=1e-4
        )
        # Calibrate on the deployed float32 head to avoid a boundary flip from float64 rounding.
        validation_probabilities = torch.from_numpy(folded).sigmoid().numpy()
        threshold = select_threshold(val["labels"], validation_probabilities)
        selection["validation"] = metrics(
            val["labels"], validation_probabilities, threshold
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": config,
                "metadata": metadata,
                "threshold": threshold,
                "selection": selection,
            },
            checkpoint_path,
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    threshold = checkpoint["threshold"]
    save_json(out / "validation_metrics.json", checkpoint["selection"])
    # Final evaluation follows all baseline selection. Does not influence any selection.
    with torch.inference_mode():
        logits = model.head(torch.from_numpy(test["pooled"])).squeeze(-1)
        probabilities = logits.sigmoid().numpy()
    baseline = metrics(test["labels"], probabilities, threshold)
    save_json(out / "test_metrics.json", baseline)
    pd.DataFrame(
        {
            "id": test["ids"],
            "label": test["labels"],
            "logit": logits.numpy(),
            "probability": probabilities,
            "prediction": (probabilities >= threshold).astype(int),
        }
    ).to_csv(out / "predictions.csv", index=False)
    save_curves(test["labels"], probabilities, out / "curves.png")
    print(json.dumps(baseline, indent=2), flush=True)
    if args.stage == "train":
        return
    ranking = rank_channels(train["intermediate"], train["labels"])
    # Association with the trained model's predictions is observational, too.
    with torch.inference_mode():
        train_logits = model.head(torch.from_numpy(train["pooled"])).squeeze(-1).numpy()
    centered = train["intermediate"].astype(np.float64)
    centered = centered - centered.mean(axis=0)
    z = train_logits.astype(np.float64) - train_logits.mean()
    denominator = np.sqrt((centered**2).sum(axis=0) * (z**2).sum())
    correlations = np.divide(
        centered.T @ z,
        denominator,
        out=np.zeros(centered.shape[1]),
        where=denominator > 1e-12,
    )
    ranking["training_logit_correlation"] = correlations[ranking.channel.to_numpy()]
    ranking.to_csv(out / "ranking.csv", index=False)
    probe, probe_threshold, probe_selection = fit_probe(train, val)
    probe_metrics = metrics(
        test["labels"], probe.predict_proba(test["intermediate"])[:, 1], probe_threshold
    )
    save_json(
        out / "probe.json",
        {
            "selection": probe_selection,
            "test": probe_metrics,
            "scaler_mean": probe[0].mean_.tolist(),
            "scaler_scale": probe[0].scale_.tolist(),
            "coefficient": probe[1].coef_.tolist(),
            "intercept": probe[1].intercept_.tolist(),
        },
    )
    save_examples(model, datasets["train"], ranking, out / "activation_examples.png")
    if args.stage == "features":
        save_analysis(ranking, train["intermediate"], train["labels"], None, out)
        return
    if (out / "ablation_summary.csv").exists() and not args.force:
        summary = pd.read_csv(out / "ablation_summary.csv")
    else:
        indices = stratified_indices(
            datasets["test"].labels, args.ablation_size, args.seed
        )
        batches = loader(Subset(datasets["test"], indices.tolist()), args.batch_size)
        summary, raw = run_ablations(
            model, batches, ranking, threshold, args.budgets, args.repeats, args.seed
        )
        summary.to_csv(out / "ablation_summary.csv", index=False)
        raw.to_csv(out / "ablation_predictions.csv", index=False)
    save_analysis(ranking, train["intermediate"], train["labels"], summary, out)
    write_report(out, config, metadata, baseline, probe_metrics, summary)
    save_json(
        out / "manifest.json",
        {
            "checkpoint_sha256": file_hash(checkpoint_path),
            "config_fingerprint": fingerprint(config),
            "completed": ["baseline", "features", "probe", "ablations"],
            "elapsed_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )
    print(f"Saved experiment and report to {out}", flush=True)


if __name__ == "__main__":
    main()
