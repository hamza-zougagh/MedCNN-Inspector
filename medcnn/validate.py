"""Validate all prepared layers with real held-out inference and hook restoration."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .core import LAYERS, intervene, layer_for
from .pneumoniamnist import load_data
from .models import load_checkpoint
from .prepare import analysis_dir
from .utils import file_hash, save_json, seed_all


def validate_run(run):
    run = Path(run)
    model, checkpoint = load_checkpoint(run / "model.pt")
    config = checkpoint["config"]
    seed_all(config["seed"], config.get("threads", 4))
    datasets, _ = load_data(image_size=config["image_size"],
                            limits=config["limits"], seed=config["seed"], download=False)
    x = torch.stack([datasets["test"][i][0] for i in range(4)])
    checkpoint_hash = file_hash(run / "model.pt")
    parameters = {key: value.clone() for key, value in model.state_dict().items()}
    rows = []
    with torch.inference_mode():
        baseline = model(x)
        for name in LAYERS[model.architecture]:
            layer = layer_for(model, name)
            target = analysis_dir(run, name)
            provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
            ranking = pd.read_csv(target / "ranking.csv")
            mean = np.load(target / "training_mean.npy")
            if provenance["checkpoint_sha256"] != checkpoint_hash or provenance["n"] != len(datasets["train"]):
                raise AssertionError(f"Stale or incomplete training statistics: {name}")
            if len(ranking) != len(mean):
                raise AssertionError(f"Layer width mismatch: {name}")
            channel = int(ranking.iloc[0].channel)
            for method in ("zero", "training_mean"):
                with intervene(layer, [channel], method, mean):
                    changed = model(x)
                restored = model(x)
                torch.testing.assert_close(restored, baseline, rtol=0, atol=0)
                if layer._forward_hooks:
                    raise AssertionError(f"Hook leaked: {name}")
                if not torch.isfinite(changed).all():
                    raise AssertionError(f"Nonfinite prediction: {name}")
                rows.append({"layer": name, "method": method, "channel": channel,
                             "n_test_images": len(x),
                             "mean_absolute_logit_change": float((changed - baseline).abs().mean()),
                             "baseline_logits": baseline.tolist(), "changed_logits": changed.tolist()})
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, parameters[key], rtol=0, atol=0)
    report = {"architecture": model.architecture, "checkpoint_sha256": checkpoint_hash,
              "training_n": len(datasets["train"]), "validation_n": len(datasets["val"]),
              "test_n": len(datasets["test"]), "checks": rows,
              "parameters_unchanged": True, "hooks_cleaned": True}
    save_json(run / "layer_validation.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    report = validate_run(args.run)
    print(f"Validated {report['architecture']}: {len(report['checks'])} real layer interventions")


if __name__ == "__main__":
    main()
