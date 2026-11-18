"""Real generic-workbench validation on the two trained example checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .concepts import concept_intervention, fit_concept
from .data import get_dataset
from .feature_discovery import cluster_features, extract_bank, top_examples
from .interactions import interaction_experiment
from .patching import patch_activations, sensitivity_scan
from .utils import save_json
from .workbench import discover_layers, load_model, predict_image, preprocess, read_spec


def _labeled_items(dataset, split, label, count, spec):
    indices = np.flatnonzero(dataset.labels == label)[:count]
    return [(f"{split}:{dataset.indices[i]}", preprocess(dataset.original(int(i)), spec))
            for i in indices]


def validate(spec_path):
    spec = read_spec(spec_path)
    torch.set_num_threads(4)
    model = load_model(spec)
    datasets, _ = get_dataset(download=False)
    test = datasets["test"]
    first, second = (preprocess(test.original(i), spec) for i in (0, 1))
    layer_name = ("backbone.layer3" if spec.factory == "medical_resnet18"
                  else "backbone.features.denseblock3")
    parameters = {key: value.clone() for key, value in model.state_dict().items()}
    shapes = discover_layers(model, first[None])
    assert layer_name in shapes
    original_probability = predict_image(model, first, spec)
    patch = patch_activations(model, spec, first, second, layer_name,
                              channels=[0, 1], region=(.25, .25, .75, .75))
    scan = sensitivity_scan(model, spec, first, second, [layer_name], 1,
                            channels_by_layer={layer_name: [0, 1]})
    items = [(f"test:{test.indices[i]}", preprocess(test.original(i), spec))
             for i in range(8)]
    bank = extract_bank(model, spec, layer_name, items, batch_size=4)
    clusters = cluster_features(bank, 2)
    top = top_examples(bank, 0, 2)
    train, val = datasets["train"], datasets["val"]
    concept = fit_concept(model, spec, layer_name,
        _labeled_items(train, "train", 1, 4, spec),
        _labeled_items(train, "train", 0, 4, spec),
        _labeled_items(val, "val", 1, 2, spec),
        _labeled_items(val, "val", 0, 2, spec))
    direction = concept_intervention(model, spec, first, layer_name,
                                     concept["direction"], 1, alpha=.5)
    short = layer_name.removeprefix("backbone.").replace(".", "_")
    mean = np.load(spec.checkpoint.parent / "layers" / short / "training_mean.npy")
    interactions = interaction_experiment(model, spec, first, layer_name,
                                          [0, 1, 2], 1, "training_mean", mean)
    assert not any(module._forward_hooks for module in model.modules())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, parameters[key], rtol=0, atol=0)
    report = {"model": spec.name, "layer": layer_name,
              "discovered_4d_layers": len(shapes),
              "test_probability": original_probability.tolist(),
              "patch_target_probability": float(patch["patched"][1]),
              "patch_control_delta": float(patch["control"][1] - patch["original"][1]),
              "sensitivity_conditions": len(scan), "feature_images": len(bank["ids"]),
              "top_channel_examples": top,
              "cluster_counts": np.bincount(clusters).tolist(),
              "concept_fit_n": concept["train_n"], "concept_eval_n": concept["eval_n"],
              "concept_eval_auc": concept["evaluation_roc_auc"],
              "directional_sensitivity": direction["directional_sensitivity"],
              "interaction_pairs": len(interactions),
              "hooks_cleaned": True, "parameters_unchanged": True}
    save_json(spec.checkpoint.parent / "workbench_validation.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True)
    args = p.parse_args()
    result = validate(args.spec)
    print(f"Validated {result['model']}: {result['discovered_4d_layers']} 4D layers, "
          f"{result['sensitivity_conditions']} patches, {result['interaction_pairs']} pairs")


if __name__ == "__main__":
    main()
