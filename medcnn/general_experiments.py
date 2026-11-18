"""Small, paired dataset experiments for compatible 2D classifiers.

Reference statistics and concept probes use train/validation only. Every measured
condition in a trial uses the same held-out recipients. Intervals resample unique
recipient images, so repeated trials never pretend to be independent patients.
"""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from .concepts import concept_intervention, fit_concept
from .feature_discovery import extract_bank
from .patching import patch_activations
from .utils import file_hash
from .workbench import (device_of, edit_map, get_layer, logits, predict_image,
                        preprocess, probabilities)


def bootstrap_ci(values, seed=42, draws=500, confidence=.95):
    """Percentile CI for a mean, with deterministic image-level resampling."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Bootstrap needs at least two finite image effects")
    if draws < 100 or not 0 < confidence < 1:
        raise ValueError("Use at least 100 draws and a confidence in (0,1)")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def summarize(raw, seed=42, bootstrap_draws=500):
    """Summarize paired effects, aggregating repeated recipients before bootstrap."""
    required = {"image_id", "trial", "effect", "random_effect", "control_effect"}
    if raw.empty or not required <= set(raw):
        raise ValueError("Experiment rows are incomplete")
    per_image = raw.groupby("image_id", sort=True)[
        ["effect", "random_effect", "control_effect"]].mean()
    effects = per_image.effect.to_numpy()
    contrasts = (per_image.effect - per_image.random_effect).to_numpy()
    low, high = bootstrap_ci(effects, seed, bootstrap_draws)
    control_low, control_high = bootstrap_ci(contrasts, seed + 1, bootstrap_draws)
    trial_effects = raw.groupby("trial").effect.mean().to_numpy()
    mean = float(effects.mean())
    sign = np.sign(mean)
    consistent = float(np.mean(np.sign(effects) == sign)) if sign else None
    stable_trials = (float(np.mean(np.sign(trial_effects) == sign))
                     if sign and len(trial_effects) > 1 else None)
    repeated = (len(trial_effects) >= 3 and len(effects) >= 20 and
                (low > 0 or high < 0) and
                (control_low > 0 or control_high < 0))
    return {"mean_effect": mean, "mean_absolute_effect": float(np.abs(effects).mean()),
            "effect_ci_low": low, "effect_ci_high": high,
            "mean_random_effect": float(per_image.random_effect.mean()),
            "mean_paired_control_difference": float(contrasts.mean()),
            "control_difference_ci_low": control_low,
            "control_difference_ci_high": control_high,
            "mean_baseline_control_error": float(per_image.control_effect.abs().mean()),
            "between_image_sd": float(effects.std(ddof=1)),
            "between_trial_sd": float(trial_effects.std(ddof=1)) if len(trial_effects) > 1 else None,
            "image_sign_consistency": consistent, "trial_sign_consistency": stable_trials,
            "unique_test_images": len(effects), "trials": len(trial_effects),
            "evidence_label": "repeated test-image sensitivity" if repeated else "exploratory"}


def _digest(image):
    rgb = image.convert("RGB")
    return sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def _choose_balanced(dataset, split, target, count, seed, minimum):
    labels = np.asarray(dataset.labels(split))
    positive = np.flatnonzero(labels == target)
    negative = np.flatnonzero(labels != target)
    if len(positive) < minimum or len(negative) < minimum:
        raise ValueError(f"{split} needs at least {minimum} target and other-class images")
    rng = np.random.default_rng(seed)
    each = min(len(positive), len(negative), max(minimum, count // 2))
    return (rng.choice(positive, each, replace=False).tolist(),
            rng.choice(negative, each, replace=False).tolist())


def _load_items(dataset, spec, split, positions):
    items, digests = [], set()
    for position in positions:
        image_id, image, label = dataset.item(split, int(position))
        items.append((image_id, preprocess(image, spec), label))
        digests.add(_digest(image))
    return items, digests


def _check_disjoint(named_digests):
    names = list(named_digests)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if named_digests[first] & named_digests[second]:
                raise ValueError(f"Identical image content crosses {first} and {second} splits")


def _trial_positions(dataset, kind, target, sample_size, repeats, rng):
    labels = np.asarray(dataset.labels("test"))
    choices = np.arange(len(labels))
    if kind == "patching":
        if sample_size < 4:
            raise ValueError("Patching needs at least four test images per trial for distinct matched donors")
        if int(np.sum(labels == target)) < 2:
            raise ValueError("Patching needs at least two test images of the target class")
        other_labels = [value for value in sorted(set(labels))
                        if value != target and int(np.sum(labels == value)) >= 2]
        if not other_labels:
            raise ValueError("Patching needs at least two test images in another class")
    trials = []
    for _ in range(repeats):
        if kind == "patching":
            second = other_labels[0]
            anchors = rng.choice(choices[labels == target], 2, replace=False).astype(int).tolist()
            anchors += rng.choice(choices[labels == second], 2, replace=False).astype(int).tolist()
            others = rng.choice(choices[~np.isin(choices, anchors)],
                                sample_size - 4, replace=False).astype(int).tolist()
            trials.append(anchors + others)
        else:
            trials.append(rng.choice(choices, sample_size, replace=False).astype(int).tolist())
    return trials


@torch.inference_mode()
def _ablate(model, spec, image, layer, channels, method, train_mean):
    def replace(output):
        if method == "zero":
            output[:, channels] = 0
        else:
            values = torch.as_tensor(train_mean[channels], device=output.device,
                                     dtype=output.dtype)
            output[:, channels] = values[None, :, None, None]
    with edit_map(layer, replace):
        result = probabilities(logits(model, image[None].to(device_of(model))), spec)[0]
    return result.detach().cpu().numpy()


def run_experiment(model, spec, dataset, layer_name, target_class,
                   kind="ablation", sample_size=8, repeats=2, seed=42,
                   reference_size=32, channel_count=1, channels=None,
                   method="zero", alpha=.5, bootstrap_draws=500,
                   cache_dir="workbench_cache/features", progress=None):
    """Return image rows, reliability summary, and a reproducible configuration."""
    if kind not in ("ablation", "patching", "concept"):
        raise ValueError("Choose ablation, patching, or concept")
    if not 0 <= target_class < len(spec.class_names) or repeats < 1 or repeats > 20:
        raise ValueError("Invalid target class or trial count")
    if not 2 <= sample_size <= dataset.count("test") or not 4 <= reference_size <= 256:
        raise ValueError("Sample size or training-reference count is outside limits")
    if method not in ("zero", "training_mean") or alpha <= 0:
        raise ValueError("Invalid replacement or concept shift")
    layer = get_layer(model, layer_name)
    target = spec.class_names[target_class]
    rng = np.random.default_rng(seed)
    test_positions = _trial_positions(dataset, kind, target, sample_size, repeats, rng)
    unique_test = sorted(set(position for trial in test_positions for position in trial))
    test_items, test_digests = _load_items(dataset, spec, "test", unique_test)
    lookup = dict(zip(unique_test, test_items))
    named_digests = {"test": test_digests}
    reference_ids, probe_ids = [], []
    direction = train_mean = None
    selected = []
    probe_metrics = None
    if kind in ("ablation", "concept"):
        positive, negative = _choose_balanced(dataset, "train", target,
                                               reference_size, seed + 17, 2)
        reference, train_digests = _load_items(dataset, spec, "train", positive + negative)
        reference_ids = [row[0] for row in reference]
        named_digests["train"] = train_digests
        if kind == "ablation":
            bank = extract_bank(model, spec, layer_name,
                [(row[0], row[1]) for row in reference], cache_dir=cache_dir)
            values = bank["mean"]
            width = values.shape[1]
            if not 1 <= channel_count < width:
                raise ValueError("Random-channel control needs at least one unselected channel")
            train_mean = values.mean(axis=0)
            positive_values = values[:len(positive)]
            negative_values = values[len(positive):]
            scale = np.maximum(values.std(axis=0), 1e-6)
            association = (positive_values.mean(axis=0) -
                           negative_values.mean(axis=0)) / scale
            selected = np.argsort(-association)[:channel_count].astype(int).tolist()
        else:
            val_positive, val_negative = _choose_balanced(
                dataset, "val", target, min(reference_size, 16), seed + 19, 1)
            validation, val_digests = _load_items(
                dataset, spec, "val", val_positive + val_negative)
            named_digests["val"] = val_digests
            probe_ids = [row[0] for row in validation]
            _check_disjoint(named_digests)
            fitted = fit_concept(model, spec, layer_name,
                [(row[0], row[1]) for row in reference[:len(positive)]],
                [(row[0], row[1]) for row in reference[len(positive):]],
                [(row[0], row[1]) for row in validation[:len(val_positive)]],
                [(row[0], row[1]) for row in validation[len(val_positive):]],
                cache_dir=cache_dir)
            direction = fitted["direction"]
            probe_metrics = {"validation_accuracy": fitted["evaluation_accuracy"],
                             "validation_roc_auc": fitted["evaluation_roc_auc"]}
    if kind == "patching":
        with torch.inference_mode():
            width = None
            from .workbench import capture_map
            with capture_map(layer) as captured:
                logits(model, test_items[0][1][None].to(device_of(model)))
                width = int(captured["map"].shape[1])
        selected = list(range(width)) if channels is None else sorted(set(map(int, channels)))
        if not selected or selected[0] < 0 or selected[-1] >= width or len(selected) > 16:
            raise ValueError("Choose 1–16 valid patch channels")
    _check_disjoint(named_digests)
    rows = []
    labels = np.asarray(dataset.labels("test"))
    total = repeats * sample_size
    done = 0
    for trial, positions in enumerate(test_positions):
        random_channels = None
        if kind == "ablation":
            random_channels = np.random.default_rng(seed + 1000 + trial).choice(
                width, len(selected), replace=False).astype(int).tolist()
        for position in positions:
            image_id, image, label = lookup[position]
            baseline = predict_image(model, image, spec)[target_class]
            donor_id = random_donor_id = ""
            if kind == "ablation":
                measured = _ablate(model, spec, image, layer, selected,
                                   method, train_mean)[target_class]
                random_score = _ablate(model, spec, image, layer, random_channels,
                                       method, train_mean)[target_class]
                control = baseline
            elif kind == "patching":
                alternatives = [p for p in positions if p != position]
                opposite = [p for p in alternatives if labels[p] != label]
                if not opposite:
                    raise ValueError("Each patch recipient needs an opposite-label donor in its trial")
                donor_position = int(rng.choice(opposite))
                random_pool = [p for p in alternatives
                               if p != donor_position and labels[p] == labels[donor_position]]
                if not random_pool:
                    raise ValueError("No distinct donor with the same label is available")
                random_position = int(rng.choice(random_pool))
                donor_id, donor = lookup[donor_position][0:2]
                random_donor_id, random_donor = lookup[random_position][0:2]
                primary = patch_activations(model, spec, image, donor, layer_name,
                                            selected)
                random_result = patch_activations(model, spec, image, random_donor,
                                                  layer_name, selected)
                measured = primary["patched"][target_class]
                random_score = random_result["patched"][target_class]
                control = primary["control"][target_class]
            else:
                result = concept_intervention(model, spec, image, layer_name,
                    direction, target_class, alpha, seed + 1000 + trial)
                measured = result["positive_shift"]
                random_score = result["random_control"]
                control = result["zero_control"]
            rows.append({"trial": trial, "image_id": image_id, "label": label,
                         "baseline": float(baseline), "intervened": float(measured),
                         "random_control": float(random_score),
                         "baseline_control": float(control),
                         "effect": float(measured - baseline),
                         "random_effect": float(random_score - baseline),
                         "control_effect": float(control - baseline),
                         "donor_id": donor_id, "random_donor_id": random_donor_id,
                         "channels": json.dumps(selected),
                         "random_channels": json.dumps(random_channels) if random_channels else ""})
            done += 1
            if progress:
                progress(done, total)
    raw = pd.DataFrame(rows)
    report = summarize(raw, seed + 2000, bootstrap_draws)
    if probe_metrics:
        report.update(probe_metrics)
    configuration = {"model": spec.name, "factory": spec.factory,
        "checkpoint_sha256": file_hash(spec.checkpoint), "dataset": dataset.entry["id"],
        "layer": layer_name, "target_class": target, "kind": kind,
        "sample_size": sample_size, "repeats": repeats, "seed": seed,
        "reference_size": reference_size if kind != "patching" else None,
        "reference_ids": reference_ids, "probe_validation_ids": probe_ids,
        "selected_content_sha256": {name: sha256("".join(sorted(digests)).encode()).hexdigest()
                                    for name, digests in named_digests.items()},
        "test_ids_by_trial": [[lookup[p][0] for p in trial] for trial in test_positions],
        "channels": selected, "channel_count": len(selected), "method": method,
        "alpha": alpha if kind == "concept" else None,
        "bootstrap_draws": bootstrap_draws,
        "preprocessing": {"input_size": spec.input_size, "channels": spec.input_channels,
                          "mean": spec.mean, "std": spec.std}}
    return raw, report, configuration


def summary_png(report):
    """Small exportable figure showing the estimate and both bootstrap intervals."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    estimates = [report["mean_effect"], report["mean_paired_control_difference"]]
    lows = [report["effect_ci_low"], report["control_difference_ci_low"]]
    highs = [report["effect_ci_high"], report["control_difference_ci_high"]]
    ax.errorbar(estimates, [1, 0], xerr=[np.asarray(estimates) - lows,
                np.asarray(highs) - estimates], fmt="o", color="#176a84", capsize=4)
    ax.axvline(0, color="#777", linewidth=1)
    ax.set(yticks=[0, 1], yticklabels=["Intervention − random", "Intervention − baseline"],
           xlabel="Target-class probability change (image bootstrap 95% CI)")
    fig.tight_layout()
    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=150)
    plt.close(fig)
    return stream.getvalue()


def save_experiment(raw, report, configuration, root="workspace_registry"):
    """Keep completed results without overwriting a prior run."""
    fingerprint = sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:16]
    experiments = Path(root).resolve() / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    payloads = {"per_image.csv": raw.to_csv(index=False).encode("utf-8"),
                "summary.json": json.dumps(report, indent=2).encode("utf-8"),
                "comparison.png": summary_png(report),
                "config.json": json.dumps(configuration, indent=2).encode("utf-8")}
    for number in range(1, 10_000):
        name = fingerprint if number == 1 else f"{fingerprint}-{number}"
        target = experiments / name
        if target.exists():
            if all((target / filename).is_file() and
                   (target / filename).read_bytes() == content
                   for filename, content in payloads.items()):
                return target
            continue
        stage = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=experiments)).resolve()
        try:
            for filename, content in payloads.items():
                (stage / filename).write_bytes(content)
            try:
                stage.rename(target)
            except FileExistsError:
                continue
            return target
        finally:
            if stage.exists():
                if stage.parent != experiments:
                    raise RuntimeError("Experiment staging directory escaped the registry")
                shutil.rmtree(stage)
    raise RuntimeError("Too many saved runs share this configuration")
