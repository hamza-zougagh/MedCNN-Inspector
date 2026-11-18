"""Streamlit workspaces for trusted local 2D CNNs."""

from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
import streamlit as st
import torch
from torch.utils.data import Subset

from .concepts import concept_intervention, fit_concept
from .catalog import DatasetHandle
from .data import get_dataset
from .feature_discovery import cluster_features, extract_bank, top_examples
from .general_experiments import run_experiment, save_experiment, summary_png
from .interactions import interaction_experiment, network_png
from .patching import patch_activations, sensitivity_scan
from .studio import png_bytes as studio_png, run_studio
from .core import LAYERS
from .pneumoniamnist import loader, stratified_indices
from .utils import file_hash
from .workbench import (capture_map, device_of, discover_layers, get_layer,
                        edit_map, load_local_factories, load_model, logits, predict_image,
                        probabilities,
                        preprocess, read_spec)


@st.cache_resource
def example_dataset():
    return get_dataset(download=True)[0]


def _uploaded_items(files, spec):
    items, images = [], {}
    for uploaded in files:
        data = uploaded.getvalue()
        if len(data) > 50_000_000:
            raise ValueError(f"Uploaded image {uploaded.name} exceeds 50 MB")
        name = f"upload:{uploaded.name}:{__import__('hashlib').sha256(data).hexdigest()[:12]}"
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > 25_000_000:
                raise ValueError(f"Uploaded image {uploaded.name} exceeds 25 megapixels")
            image = source.convert("RGB")
        items.append((name, preprocess(image, spec)))
        images[name] = image
    return items, images


def _select_image(spec, key, default_index=0, dataset=None):
    source = "Upload"
    if dataset is not None:
        source = st.radio("Image source", ("Paired dataset test", "Upload"),
                          horizontal=True, key=f"{key}_source")
    if source == "Paired dataset test":
        index = st.number_input("Test image position", 0, dataset.count("test") - 1,
                                min(default_index, dataset.count("test") - 1), key=f"{key}_index")
        image_id, image, _label = dataset.item("test", index)
        return image_id, image, preprocess(image, spec)
    uploaded = st.file_uploader("Upload PNG/JPEG image", type=("png", "jpg", "jpeg"),
                                key=f"{key}_upload")
    if uploaded is None:
        st.info("Upload an image to continue.")
        st.stop()
    try:
        items, images = _uploaded_items([uploaded], spec)
    except Exception as exc:
        st.error(f"Could not preprocess image: {exc}")
        st.stop()
    return items[0][0], next(iter(images.values())), items[0][1]


def _overlay(original, activation, channel, peak=None):
    values = activation[channel].detach().cpu()[None, None]
    up = torch.nn.functional.interpolate(values, size=(256, 256),
        mode="bilinear", align_corners=False)[0, 0].numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(original.resize((256, 256)), cmap="gray")
    view = ax.imshow(up, cmap="magma", alpha=.55)
    if peak is not None:
        height, width = values.shape[-2:]
        ax.scatter([(peak["peak_x"] + .5) * 256 / width],
                   [(peak["peak_y"] + .5) * 256 / height],
                   marker="x", s=80, color="cyan", linewidths=2)
    ax.axis("off")
    fig.colorbar(view, ax=ax, fraction=.046)
    return fig


def _figure_png(fig):
    stream = BytesIO()
    fig.savefig(stream, format="png", dpi=150, bbox_inches="tight")
    return stream.getvalue()


def _config(spec, **kwargs):
    return {"model": spec.name, "factory": spec.factory,
            "checkpoint_sha256": file_hash(spec.checkpoint),
            "paired_dataset": st.session_state.get("paired_dataset_id"),
            "preprocessing": {"input_size": spec.input_size,
                "channels": spec.input_channels, "mean": spec.mean, "std": spec.std},
            **kwargs}


def _reference_mean(model, spec, layer_name, dataset=None):
    if (dataset is not None and dataset.kind == "pneumoniamnist" and
        spec.factory.startswith("medical_") and layer_name.startswith("backbone.")):
        short = layer_name.removeprefix("backbone.").replace(".", "_")
        run = spec.checkpoint.parent
        path = run / "layers" / short / "training_mean.npy"
        provenance = run / "layers" / short / "provenance.json"
        if path.exists() and provenance.exists():
            info = json.loads(provenance.read_text(encoding="utf-8"))
            if (info.get("checkpoint_sha256") == file_hash(spec.checkpoint) and
                info.get("dataset") == "pneumoniamnist"):
                return np.load(path), [f"saved_train_mean:{path.name}"]
    if dataset is not None:
        count = min(32, dataset.count("train"))
        if count < 2:
            st.error("Training-mean replacement needs at least two training images")
            return None, []
        positions = np.linspace(0, dataset.count("train") - 1, count, dtype=int)
        try:
            items = [(image_id, preprocess(image, spec))
                     for image_id, image, _ in
                     (dataset.item("train", int(index)) for index in positions)]
            bar = st.progress(0, text="Preparing training-only reference mean")
            try:
                with st.spinner("Measuring training activations…"):
                    bank = extract_bank(model, spec, layer_name, items,
                        cache_dir="workbench_cache/features",
                        progress=lambda done, total: bar.progress(done / total))
            finally:
                bar.empty()
            return bank["mean"].mean(0), [item[0] for item in items]
        except Exception as exc:
            st.error(f"Could not prepare training reference: {exc}")
            return None, []
    files = st.file_uploader("Training reference images for mean replacement",
        type=("png", "jpg", "jpeg"), accept_multiple_files=True,
        key="wb_training_reference")
    if not files:
        st.info("Supply training-split reference images for mean replacement. Uploaded split provenance cannot be verified.")
        return None, []
    items, _ = _uploaded_items(files, spec)
    bank = extract_bank(model, spec, layer_name, items,
                        cache_dir="workbench_cache/features")
    return bank["mean"].mean(0), [item[0] for item in items]


def _dataset_experiments(model, spec, dataset, layer_name, shape, target_index, identity):
    st.subheader("Dataset Experiments")
    st.write("Run matched interventions on held-out test images. Reference statistics use training images; concept probes are evaluated on validation images.")
    kind = st.selectbox("Experiment type", ("ablation", "patching", "concept"),
        format_func=lambda value: {"ablation": "Channel ablation",
            "patching": "Activation patching", "concept": "Class-label concept direction"}[value])
    maximum = min(64, dataset.count("test"))
    minimum = 4 if kind == "patching" else 2
    if maximum < minimum:
        st.error(f"{kind.title()} needs at least {minimum} test images")
        return
    sample_size = (minimum if maximum == minimum else
                   st.slider("Test images per trial", minimum, maximum, min(8, maximum)))
    if maximum == minimum:
        st.caption(f"Test images per trial: {sample_size} (all available)")
    repeats = st.slider("Independent seeded trials", 1, 10, 3)
    seed = st.number_input("Dataset experiment seed", 0, 999999, 42)
    reference_size = (st.slider("Training reference images", 4, 128, 32)
                      if kind != "patching" else 32)
    bootstrap_draws = st.select_slider("Image bootstrap draws", (200, 500, 1000), value=500)
    channel_count, channels, method, alpha = 1, None, "zero", .5
    if kind == "ablation":
        if shape[1] < 2:
            st.error("A matched random-channel control needs a layer with at least two channels")
            return
        channel_count = st.slider("Target-associated channels", 1,
            min(16, shape[1] - 1), min(2, shape[1] - 1))
        method = st.radio("Dataset replacement", ("zero", "training_mean"), horizontal=True)
    elif kind == "patching":
        channels = st.multiselect("Patch channels (1–16)", range(shape[1]), default=[0],
            key="dataset_patch_channels")
        st.caption("Each recipient receives an opposite-label donor patch. The random control uses a distinct donor with the same label and the same channels; a same-image patch checks baseline restoration.")
    else:
        alpha = st.number_input("Concept shift magnitude", min_value=.01,
                                 max_value=5.0, value=.5)
        st.warning("The direction is fitted from dataset class labels. It is a label-associated direction, not an automatically identified clinical concept.")
    key = (identity, layer_name, target_index, kind, sample_size, repeats, seed,
           reference_size, channel_count, tuple(channels or ()), method, alpha,
           bootstrap_draws)
    if st.button("Run dataset experiment", type="primary"):
        try:
            if kind == "patching" and not channels:
                raise ValueError("Select at least one channel to patch")
            bar = st.progress(0, text="Running matched test-image conditions")
            try:
                raw, report, config = run_experiment(model, spec, dataset,
                    layer_name, target_index, kind=kind, sample_size=sample_size,
                    repeats=repeats, seed=int(seed), reference_size=reference_size,
                    channel_count=channel_count, channels=channels, method=method,
                    alpha=alpha, bootstrap_draws=bootstrap_draws,
                    progress=lambda done, total: bar.progress(done / total,
                        text=f"Measured {done}/{total} image conditions"))
            finally:
                bar.empty()
            saved = save_experiment(raw, report, config)
            st.session_state.wb_dataset_study = (key, raw, report, config, saved)
        except Exception as exc:
            st.error(f"Dataset experiment failed: {exc}")
    if ("wb_dataset_study" in st.session_state and
        st.session_state.wb_dataset_study[0] == key):
        _, raw, report, config, saved = st.session_state.wb_dataset_study
        st.success(f"Measured {report['unique_test_images']} distinct test images across {report['trials']} trials. Saved locally to {saved}")
        st.metric("Mean target probability change", f"{report['mean_effect']:+.4f}",
                  help="Compared with the unchanged baseline on the same images")
        st.write(f"95% image-bootstrap interval: [{report['effect_ci_low']:+.4f}, {report['effect_ci_high']:+.4f}]")
        st.write(f"Intervention minus matched random: {report['mean_paired_control_difference']:+.4f} "
                 f"[{report['control_difference_ci_low']:+.4f}, {report['control_difference_ci_high']:+.4f}]")
        image_consistency = (f"{report['image_sign_consistency']:.0%}"
                             if report['image_sign_consistency'] is not None else "not defined")
        trial_consistency = (f"{report['trial_sign_consistency']:.0%}"
                             if report['trial_sign_consistency'] is not None else "not measurable")
        st.caption(f"{report['evidence_label'].capitalize()}. Image sign consistency {image_consistency}; "
                   f"trial sign consistency {trial_consistency}. Intervals resample distinct images after averaging repeated appearances.")
        if report["unique_test_images"] < 20:
            st.warning("Fewer than 20 distinct test images: bootstrap intervals and stability estimates may be unstable.")
        st.dataframe(pd.DataFrame([report]), hide_index=True, width="stretch")
        st.dataframe(raw, hide_index=True, width="stretch")
        st.image(summary_png(report), caption="Paired probability effects with image-bootstrap intervals")
        st.download_button("Experiment per-image CSV", raw.to_csv(index=False),
                           "per_image.csv")
        st.download_button("Experiment summary CSV", pd.DataFrame([report]).to_csv(index=False),
                           "summary.csv")
        st.download_button("Experiment comparison PNG", summary_png(report),
                           "comparison.png")
        st.download_button("Experiment configuration JSON", json.dumps(config, indent=2),
                           "config.json")
    st.caption("Repeated test-image sensitivity is still exploratory science. Internal edits can be out of distribution and cannot establish a clinical mechanism.")


def render(spec_path, dataset=None):
    load_local_factories()
    try:
        spec = read_spec(spec_path)
        if not spec.checkpoint.exists():
            st.error(f"Local state_dict checkpoint not found: {spec.checkpoint}")
            st.stop()
    except Exception as exc:
        st.error(f"Model specification error: {exc}")
        st.stop()
    if dataset is None and spec.dataset == "pneumoniamnist":
        try:
            dataset = DatasetHandle({"id": "builtin:pneumoniamnist",
                "name": "PneumoniaMNIST 64", "group_separated": False})
        except Exception as exc:
            st.error(f"Could not load the example dataset: {exc}")
            st.stop()
    device = st.sidebar.selectbox("Inference device", ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
    identity = (str(Path(spec_path).resolve()), file_hash(spec_path),
                file_hash(spec.checkpoint), device,
                dataset.entry["id"] if dataset is not None else None)
    if st.session_state.get("wb_model_identity") != identity:
        try:
            st.session_state.wb_model = load_model(spec, device)
            st.session_state.wb_model_identity = identity
            st.session_state.pop("wb_shapes", None)
        except Exception as exc:
            st.error(f"Could not load trusted local state_dict: {exc}")
            st.stop()
    model = st.session_state.wb_model
    if "wb_shapes" not in st.session_state:
        if dataset is not None:
            sample = preprocess(dataset.item("test", 0)[1], spec)[None]
        else:
            sample = torch.zeros((1, spec.input_channels, *spec.input_size))
        try:
            st.session_state.wb_shapes = discover_layers(model, sample)
        except Exception as exc:
            st.error(f"Layer discovery failed: {exc}")
            st.stop()
    shapes = st.session_state.wb_shapes
    if not shapes:
        st.error("This classifier exposed no compatible 4D feature maps.")
        st.stop()
    st.sidebar.success(f"{len(shapes)} compatible 4D layers")
    layer_name = st.sidebar.selectbox("Feature layer", list(shapes),
        index=list(shapes).index("backbone.layer3") if "backbone.layer3" in shapes else 0)
    target = st.sidebar.selectbox("Target class", spec.class_names,
        index=min(1, len(spec.class_names) - 1))
    target_index = spec.class_names.index(target)
    pages = ["Model Inspector", "Feature Discovery", "Activation Patching",
             "Concept Explorer", "Feature Interaction Lab", "Dataset Experiments"]
    research_example = spec.factory.startswith("medical_") and dataset is not None and dataset.kind == "pneumoniamnist"
    if research_example:
        pages.append("Controlled Dataset Experiments")
    page = st.sidebar.radio("Analysis", pages)
    st.sidebar.caption(f"{spec.name} · {device}")
    st.caption("Model → feature maps → controlled interventions → exported evidence")
    st.warning("Internal edits can be out of distribution. Association and prediction sensitivity do not establish clinical meaning or a complete circuit.")

    if page == "Model Inspector":
        st.subheader("Model Inspector")
        st.write(f"**Model:** {spec.name}  |  **Factory:** `{spec.factory}`  |  **Classes:** {', '.join(spec.class_names)}")
        st.write(f"**Input:** {spec.input_channels} × {spec.input_size[0]} × {spec.input_size[1]}  |  **Checkpoint:** `{spec.checkpoint.name}`")
        table = pd.DataFrame([{"layer": name, "channels": shape[1],
                               "height": shape[2], "width": shape[3]}
                              for name, shape in shapes.items()])
        st.dataframe(table, hide_index=True, width="stretch")
        with st.expander("Module structure"):
            st.code(str(model))
        image_id, original, image = _select_image(spec, "inspector", dataset=dataset)
        scores = predict_image(model, image, spec)
        st.write(f"Prediction from **{image_id}**")
        st.bar_chart(pd.DataFrame({"probability": scores}, index=spec.class_names))
        st.download_button("Prediction CSV", pd.DataFrame({"class": spec.class_names,
            "probability": scores}).to_csv(index=False), "prediction.csv")

    elif page == "Feature Discovery":
        st.subheader("Feature Discovery")
        if dataset is not None:
            maximum = min(64, dataset.count("test"))
            if maximum < 2:
                st.info("Feature discovery needs at least two paired test images.")
                st.stop()
            count = st.slider("Paired test images to inspect", 2, maximum,
                              min(24, maximum))
            positions = np.linspace(0, dataset.count("test") - 1, count, dtype=int)
            records = [dataset.item("test", int(i)) for i in positions]
            items = [(image_id, preprocess(image, spec)) for image_id, image, _ in records]
            images = {image_id: image for image_id, image, _ in records}
        else:
            files = st.file_uploader("Upload an image set", type=("png", "jpg", "jpeg"),
                accept_multiple_files=True, key="wb_feature_files")
            if not files:
                st.info("Upload at least two images for feature discovery.")
                st.stop()
            items, images = _uploaded_items(files, spec)
        if len(items) < 2:
            st.info("Use at least two images.")
            st.stop()
        key = (identity, layer_name, tuple(i for i, _ in items))
        if st.button("Extract activations", type="primary"):
            bar = st.progress(0)
            with st.spinner("Running real inference…"):
                bank = extract_bank(model, spec, layer_name, items,
                    cache_dir="workbench_cache/features",
                    progress=lambda done, total: bar.progress(done / total))
            bar.empty()
            st.session_state.wb_bank = (key, bank)
        if "wb_bank" in st.session_state and st.session_state.wb_bank[0] == key:
            bank = st.session_state.wb_bank[1]
            channel = st.number_input("Channel", 0, bank["mean"].shape[1] - 1, 0)
            statistic = st.selectbox("Rank examples by", ("maximum", "mean"))
            tops = top_examples(bank, channel, count=min(8, len(items)), statistic=statistic)
            st.dataframe(pd.DataFrame(tops), hide_index=True, width="stretch")
            chosen = st.selectbox("Show activation map for", [row["id"] for row in tops])
            x = next(x for image_id, x in items if image_id == chosen)
            with torch.inference_mode(), capture_map(get_layer(model, layer_name)) as captured:
                logits(model, x[None].to(device_of(model)))
            fig = _overlay(images[chosen], captured["map"][0], channel,
                           next(row for row in tops if row["id"] == chosen))
            st.pyplot(fig)
            st.download_button("Activation map PNG", _figure_png(fig), "activation_map.png")
            plt.close(fig)
            clusters = st.slider("Representation clusters", 2, min(8, len(items)), min(3, len(items)))
            labels = cluster_features(bank, clusters)
            coordinates = PCA(n_components=2).fit_transform(bank["mean"])
            scatter = pd.DataFrame({"PC1": coordinates[:, 0], "PC2": coordinates[:, 1],
                                    "cluster": labels.astype(str), "image_id": bank["ids"]})
            st.scatter_chart(scatter, x="PC1", y="PC2", color="cluster")
            st.download_button("Feature and cluster CSV", scatter.to_csv(index=False), "features.csv")
            st.download_button("Feature config JSON", json.dumps(_config(spec,
                layer=layer_name, image_ids=bank["ids"].tolist(), channel=channel,
                ranking_statistic=statistic, clusters=clusters, cluster_seed=42), indent=2),
                "feature_config.json")
            if research_example:
                short = layer_name.removeprefix("backbone.")
                rank_path = (spec.checkpoint.parent / "layers" /
                             short.replace(".", "_") / "ranking.csv")
                if rank_path.exists():
                    st.subheader("Training-label associations")
                    st.dataframe(pd.read_csv(rank_path).head(30), hide_index=True,
                                 width="stretch")
                    st.caption("Association with dataset labels is observational and is separate from the activation ranking above.")
            st.caption("Clusters describe similarity in this layer's pooled activations; they are not clinical categories.")

    elif page == "Activation Patching":
        st.subheader("Activation Patching")
        left, right = st.columns(2)
        with left:
            recipient_id, recipient_original, recipient = _select_image(spec, "patch_recipient", dataset=dataset)
            st.image(recipient_original, caption="Recipient", width=220)
        with right:
            donor_id, donor_original, donor = _select_image(spec, "patch_donor", 1, dataset)
            st.image(donor_original, caption="Donor", width=220)
        all_channels = st.checkbox("Transfer entire layer", value=True)
        channel = None if all_channels else st.number_input("Channel to transfer", 0, shapes[layer_name][1] - 1, 0)
        spatial = st.checkbox("Limit to spatial region")
        region = None
        if spatial:
            x0, x1 = st.slider("Horizontal fraction", 0.0, 1.0, (.25, .75))
            y0, y1 = st.slider("Vertical fraction", 0.0, 1.0, (.25, .75))
            region = (x0, y0, x1, y1)
        patch_key = (identity, layer_name, recipient_id, donor_id, channel, region)
        if st.button("Patch and compare", type="primary"):
            result = patch_activations(model, spec, recipient, donor, layer_name,
                None if channel is None else [channel], region)
            st.session_state.wb_patch = (patch_key, result)
        if "wb_patch" in st.session_state and st.session_state.wb_patch[0] == patch_key:
            result = st.session_state.wb_patch[1]
            comparison = pd.DataFrame({name: result[name] for name in
                ("original", "donor", "control", "patched")}, index=spec.class_names)
            st.dataframe(comparison, width="stretch")
            st.bar_chart(comparison.loc[[target]].T)
            st.metric("Patched target probability change",
                f"{result['patched'][target_index] - result['original'][target_index]:+.5f}")
            st.download_button("Patching CSV", comparison.to_csv(), "patching.csv")
            fig, ax = plt.subplots(figsize=(5, 3))
            comparison.loc[target].plot.bar(ax=ax, color=["#456", "#999", "#8aa", "#c45"])
            ax.set(ylabel=f"P({target})", ylim=(0, 1))
            st.download_button("Patching PNG", _figure_png(fig), "patching.png")
            plt.close(fig)
            st.download_button("Patching config JSON", json.dumps(_config(spec,
                layer=result["layer"], channels=result["channels"], region=result["region"],
                recipient=recipient_id, donor=donor_id, target_class=target), indent=2),
                "patching_config.json")
        scan_layers = st.multiselect("Sensitivity comparison layers", list(shapes), default=[layer_name])
        scan_key = (identity, tuple(scan_layers), recipient_id, donor_id, region, target_index)
        if st.button("Compare selected layers") and scan_layers:
            if len(scan_layers) > 8:
                st.error("Select at most eight layers for a CPU-friendly scan.")
            else:
                scan = sensitivity_scan(model, spec, recipient, donor, scan_layers, target_index,
                                        region=region)
                st.session_state.wb_scan = (scan_key, scan)
        channel_scan = st.multiselect("Sensitivity comparison channels in current layer",
            range(shapes[layer_name][1]), default=list(range(min(4, shapes[layer_name][1]))))
        channel_scan_key = (identity, layer_name, tuple(channel_scan),
                            recipient_id, donor_id, region, target_index)
        if st.button("Compare selected channels") and channel_scan:
            if len(channel_scan) > 8:
                st.error("Select at most eight channels for a CPU-friendly scan.")
            else:
                scan = sensitivity_scan(model, spec, recipient, donor, [layer_name],
                    target_index, channels_by_layer={layer_name: channel_scan}, region=region)
                st.session_state.wb_channel_scan = (channel_scan_key, scan)
        if "wb_scan" in st.session_state and st.session_state.wb_scan[0] == scan_key:
            st.dataframe(st.session_state.wb_scan[1], hide_index=True, width="stretch")
            st.download_button("Sensitivity CSV", st.session_state.wb_scan[1].to_csv(index=False),
                               "patch_sensitivity.csv")
        if ("wb_channel_scan" in st.session_state and
            st.session_state.wb_channel_scan[0] == channel_scan_key):
            st.dataframe(st.session_state.wb_channel_scan[1], hide_index=True, width="stretch")
            st.download_button("Channel sensitivity CSV",
                st.session_state.wb_channel_scan[1].to_csv(index=False),
                "channel_sensitivity.csv")
        st.caption("The same-image control must reproduce the recipient. A patch effect is checkpoint sensitivity, not proof of a biological mechanism.")

    elif page == "Concept Explorer":
        st.subheader("Concept Explorer")
        st.write("Provide disjoint positive and negative examples for fitting and evaluation. Labels are supplied by you; no clinical concept is inferred automatically.")
        groups = {}
        for name in ("fit_positive", "fit_negative", "eval_positive", "eval_negative"):
            files = st.file_uploader(name.replace("_", " ").title(),
                type=("png", "jpg", "jpeg"), accept_multiple_files=True, key=f"wb_{name}")
            groups[name] = _uploaded_items(files, spec)[0] if files else []
        regularization = st.number_input("Probe regularization C", min_value=.001,
                                         max_value=100.0, value=1.0)
        concept_key = (identity, layer_name, regularization,
            tuple((name, tuple(i for i, _ in groups[name])) for name in groups))
        if st.button("Fit and evaluate concept", type="primary"):
            try:
                result = fit_concept(model, spec, layer_name,
                    groups["fit_positive"], groups["fit_negative"],
                    groups["eval_positive"], groups["eval_negative"],
                    regularization, cache_dir="workbench_cache/features")
                st.session_state.wb_concept = (concept_key, result)
            except Exception as exc:
                st.error(str(exc))
        if "wb_concept" in st.session_state and st.session_state.wb_concept[0] == concept_key:
            result = st.session_state.wb_concept[1]
            a, b = st.columns(2)
            a.metric("Held-out concept accuracy", f"{result['evaluation_accuracy']:.3f}")
            b.metric("Held-out concept ROC-AUC", f"{result['evaluation_roc_auc']:.3f}")
            evaluation = pd.DataFrame({"image_id": result["evaluation_ids"],
                "concept_label": result["evaluation_labels"],
                "concept_probability": result["evaluation_probabilities"]})
            st.dataframe(evaluation, hide_index=True, width="stretch")
            st.download_button("Concept evaluation CSV", evaluation.to_csv(index=False), "concept_evaluation.csv")
            st.download_button("Concept direction CSV", pd.DataFrame({
                "channel": np.arange(len(result["direction"])),
                "direction": result["direction"]}).to_csv(index=False),
                "concept_direction.csv")
            st.download_button("Concept fit config JSON", json.dumps(_config(spec,
                layer=layer_name, fit_ids=[i for group in ("fit_positive", "fit_negative")
                    for i, _ in groups[group]], eval_ids=result["evaluation_ids"].tolist(),
                regularization_C=regularization), indent=2), "concept_fit_config.json")
            image_id, original, image = _select_image(spec, "concept_target", dataset=dataset)
            alpha = st.number_input("Direction shift magnitude", min_value=.01,
                                     max_value=10.0, value=1.0)
            seed = st.number_input("Random control seed", min_value=0, value=42)
            effect_key = (concept_key, image_id, target_index, alpha, seed)
            if st.button("Intervene along direction"):
                measured = concept_intervention(model, spec, image, layer_name,
                    result["direction"], target_index, alpha, seed)
                st.session_state.wb_concept_effect = (effect_key, measured)
            if ("wb_concept_effect" in st.session_state and
                st.session_state.wb_concept_effect[0] == effect_key):
                measured = st.session_state.wb_concept_effect[1]
                effects = pd.DataFrame({"condition": ["baseline", "positive shift",
                    "negative shift", "random control", "zero control"],
                    "target_probability": [measured[k] for k in ("baseline",
                    "positive_shift", "negative_shift", "random_control", "zero_control")]})
                st.dataframe(effects, hide_index=True, width="stretch")
                st.download_button("Concept intervention CSV", effects.to_csv(index=False),
                                   "concept_intervention.csv")
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(effects["condition"], effects["target_probability"])
                ax.set(ylabel=f"P({target})", ylim=(0, 1))
                ax.tick_params(axis="x", rotation=30)
                st.download_button("Concept intervention PNG", _figure_png(fig),
                                   "concept_intervention.png")
                plt.close(fig)
                st.download_button("Concept config JSON", json.dumps(_config(spec,
                    layer=layer_name, fit_ids=[i for group in ("fit_positive", "fit_negative")
                        for i, _ in groups[group]], eval_ids=result["evaluation_ids"].tolist(),
                    target_image=image_id, target_class=target,
                    alpha=alpha, seed=seed, regularization_C=regularization), indent=2),
                    "concept_config.json")
            st.caption("A concept probe measures association in supplied examples. Direction shifts are controlled model interventions; neither proves a clinical concept.")

    elif page == "Feature Interaction Lab":
        st.subheader("Feature Interaction Lab")
        image_id, original, image = _select_image(spec, "interaction_image", dataset=dataset)
        st.image(original, caption=image_id, width=240)
        channels = st.multiselect("Channels (2–8)", range(shapes[layer_name][1]), default=[0, 1])
        method = st.radio("Replacement", ("zero", "training_mean"), horizontal=True)
        training_mean, reference_ids = (_reference_mean(model, spec, layer_name, dataset)
                                        if method == "training_mean" else (None, []))
        interaction_key = (identity, layer_name, image_id, tuple(sorted(channels)),
                           method, target_index, tuple(reference_ids))
        if st.button("Measure individual and paired effects", type="primary"):
            if method == "training_mean" and training_mean is None:
                st.error("Supply training reference images for mean replacement.")
            else:
                try:
                    table = interaction_experiment(model, spec, image, layer_name,
                        channels, target_index, method, training_mean)
                    st.session_state.wb_interactions = (interaction_key, table)
                except Exception as exc:
                    st.error(str(exc))
        if ("wb_interactions" in st.session_state and
            st.session_state.wb_interactions[0] == interaction_key):
            table = st.session_state.wb_interactions[1]
            st.dataframe(table, hide_index=True, width="stretch")
            st.image(network_png(table), caption="Red: positive nonadditivity; blue: negative. Line strength shows magnitude.")
            st.download_button("Interaction CSV", table.to_csv(index=False), "interactions.csv")
            st.download_button("Interaction network PNG", network_png(table), "interaction_network.png")
            st.download_button("Interaction config JSON", json.dumps(_config(spec,
                layer=layer_name, channels=channels, method=method,
                target_class=target, image_id=image_id,
                training_reference_ids=reference_ids), indent=2),
                "interaction_config.json")
        st.caption("Nonadditivity is pair effect minus both individual effects. This small graph does not identify a complete neural circuit.")
        with st.expander("Manual channel ablation"):
            manual = st.multiselect("Channels to replace", range(shapes[layer_name][1]),
                                    key="wb_manual_channels")
            if st.button("Compare manual replacement"):
                if method == "training_mean" and training_mean is None:
                    st.error("Supply training reference images for mean replacement.")
                else:
                    with torch.inference_mode():
                        baseline = predict_image(model, image, spec)
                        def replace(output):
                            if method == "zero":
                                output[:, manual] = 0
                            else:
                                values = torch.as_tensor(np.asarray(training_mean)[manual],
                                    device=output.device, dtype=output.dtype)
                                output[:, manual] = values[None, :, None, None]
                        with edit_map(get_layer(model, layer_name), replace):
                            changed = probabilities(logits(model,
                                image[None].to(device_of(model))), spec)[0].cpu().numpy()
                    st.session_state.wb_manual = ((identity, layer_name, image_id,
                        tuple(manual), method, target_index), baseline, changed)
            manual_key = (identity, layer_name, image_id, tuple(manual), method, target_index)
            if "wb_manual" in st.session_state and st.session_state.wb_manual[0] == manual_key:
                baseline, changed = st.session_state.wb_manual[1:]
                table = pd.DataFrame({"class": spec.class_names,
                    "baseline": baseline, "intervened": changed,
                    "change": changed - baseline})
                st.dataframe(table, hide_index=True, width="stretch")
                st.download_button("Manual ablation CSV", table.to_csv(index=False),
                                   "manual_ablation.csv")

    elif page == "Dataset Experiments":
        if dataset is None:
            st.info("Pair a dataset to run controlled experiments.")
        else:
            _dataset_experiments(model, spec, dataset, layer_name,
                                 shapes[layer_name], target_index, identity)

    else:
        st.subheader("Controlled Dataset Experiments")
        st.write("Compare training-ranked channels with repeated random controls on identical held-out images.")
        stages = LAYERS[model.architecture]
        stage = st.selectbox("Experiment stage", stages,
            index=stages.index(model.layer_name))
        target_dir = spec.checkpoint.parent / "layers" / stage.replace(".", "_")
        if not all((target_dir / name).exists() for name in ("ranking.csv", "training_mean.npy")):
            st.error("Training statistics are missing for this stage. Prepare the saved model's layers first.")
            st.stop()
        ranking = pd.read_csv(target_dir / "ranking.csv")
        mean = np.load(target_dir / "training_mean.npy")
        sample_size = st.slider("Stratified test images", 2, dataset.count("test"),
                                min(64, dataset.count("test")))
        budgets = st.multiselect("Channel budgets", (1, 2, 4, 8, 16), default=(1, 4, 8))
        repeats = st.slider("Random controls per budget", 2, 10, 3)
        seed = st.number_input("Experiment seed", 0, 999999, 42)
        methods = st.multiselect("Replacement methods", ("zero", "training_mean"),
                                 default=("zero", "training_mean"))
        config_key = (identity, stage, sample_size, tuple(budgets), repeats, seed,
                      tuple(methods))
        if st.button("Run controlled experiment", type="primary"):
            if not budgets or not methods:
                st.error("Choose at least one budget and replacement method.")
            else:
                checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
                test_set = dataset.datasets["test"]
                indices = stratified_indices(test_set.labels, sample_size, seed)
                batches = loader(Subset(test_set, indices.tolist()), batch_size=16)
                with st.spinner("Running real interventions on paired test images…"):
                    summary, raw = run_studio(model, batches, stage, ranking,
                        checkpoint["threshold"], mean, budgets, repeats, seed, methods)
                configuration = _config(spec, dataset=dataset.entry["name"],
                    split="test", official_ids=test_set.indices[indices].tolist(),
                    layer=stage, budgets=budgets, repeats=repeats, seed=seed,
                    methods=methods, threshold=checkpoint["threshold"])
                st.session_state.wb_studio = (config_key, summary, raw, configuration)
        if "wb_studio" in st.session_state and st.session_state.wb_studio[0] == config_key:
            _, summary, raw, configuration = st.session_state.wb_studio
            st.dataframe(summary, hide_index=True, width="stretch")
            st.image(studio_png(summary))
            st.download_button("Summary CSV", summary.to_csv(index=False), "summary.csv")
            st.download_button("Per-image CSV", raw.to_csv(index=False), "predictions.csv")
            st.download_button("Comparison PNG", studio_png(summary), "comparison.png")
            st.download_button("Configuration JSON", json.dumps(configuration, indent=2),
                               "configuration.json")
        st.caption("The test subset is never used to fit the head or rank channels. Random error bars are SD across channel selections.")
