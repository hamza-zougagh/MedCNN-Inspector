"""Unified model, dataset, and pairing workspace for Streamlit."""

from io import BytesIO
from pathlib import Path

import streamlit as st
import torch

from .catalog import (DatasetHandle, catalog_root, forget_registration,
                      import_dataset, list_datasets, list_models,
                      register_model, validate_pair)
from .workbench import FACTORIES, load_local_factories, load_model, read_spec
from .workbench_ui import render as render_analysis


def _floats(value):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _cards(title, entries, kind):
    st.subheader(title)
    if not entries:
        st.info(f"No {kind.lower()} registered yet.")
        return
    columns = st.columns(2)
    for number, entry in enumerate(entries):
        with columns[number % 2].container(border=True):
            st.markdown(f"**{entry['name']}**")
            st.caption("Ready" if entry["ready"] else "Needs attention")
            if "classes" in entry:
                st.write("Classes: " + ", ".join(entry["classes"]))
            if kind == "Models" and "factory" in entry:
                st.caption(f"Factory: {entry['factory']}")
            if kind == "Datasets" and "counts" in entry:
                counts = entry["counts"]
                st.caption(f"Train {counts['train']} · Val {counts['val']} · Test {counts['test']}")
                st.caption("Group separated" if entry.get("group_separated") else "No group identifiers recorded")
            if entry.get("error"):
                st.error(entry["error"])


def _model_form(root):
    st.subheader("Add Model")
    st.write("Import a trusted local tensor `state_dict`. Model architecture comes from a registered factory; uploaded Python code is never run.")
    with st.form("add_model_form"):
        name = st.text_input("Model name", placeholder="My trained classifier")
        choices = [name for name in FACTORIES if not name.startswith("medical_")]
        factory = st.selectbox("2D CNN architecture or trusted local factory", choices)
        source_kind = st.radio("Checkpoint source", ("Local file path", "Upload state_dict"), horizontal=True)
        checkpoint_path = st.text_input("Local checkpoint path") if source_kind == "Local file path" else None
        uploaded = st.file_uploader("Tensor state_dict (.pt or .pth)", type=("pt", "pth")) if source_kind != "Local file path" else None
        copy_checkpoint = st.checkbox("Keep a managed local copy", value=True)
        classes_text = st.text_input("Class names, comma separated", value="class 0, class 1")
        col1, col2, col3 = st.columns(3)
        height = col1.number_input("Input height", 16, 1024, 224)
        width = col2.number_input("Input width", 16, 1024, 224)
        channels = col3.selectbox("Input channels", (3, 1))
        mean = st.text_input("Channel means", value="0.485, 0.456, 0.406" if channels == 3 else "0")
        std = st.text_input("Channel standard deviations", value="0.229, 0.224, 0.225" if channels == 3 else "1")
        output_kind = st.selectbox("Model output format", ("logits", "probabilities"))
        training_dataset = st.text_input("Training dataset or provenance (optional)")
        preprocessing_confirmed = st.checkbox("I checked these preprocessing values against training records")
        class_order_confirmed = st.checkbox("I checked class order against the trained head")
        submitted = st.form_submit_button("Validate and add model", type="primary")
    if submitted:
        staged = None
        try:
            if not name.strip():
                raise ValueError("Enter a model name")
            source = checkpoint_path
            if uploaded is not None:
                if uploaded.size > 1_500_000_000:
                    raise ValueError("Checkpoint exceeds the 1.5 GB local import limit")
                stage = catalog_root(root) / "checkpoints" / ".uploaded_state_dict.pt"
                if stage.exists():
                    raise ValueError("Another model upload is in progress; retry shortly")
                stage.write_bytes(uploaded.getvalue())
                staged, source = stage, stage
            if not source:
                raise ValueError("Choose a local checkpoint path or upload a state_dict")
            with st.spinner("Checking restricted checkpoint loading, output classes, and 4D layers…"):
                path = register_model(name.strip(), factory, source,
                    [part.strip() for part in classes_text.split(",")],
                    (height, width), channels, _floats(mean), _floats(std),
                    output_kind, root, copy_checkpoint or staged is not None,
                    metadata={"training_dataset": training_dataset.strip(),
                              "preprocessing_confirmed": preprocessing_confirmed,
                              "class_order_confirmed": class_order_confirmed})
            st.success(f"Model registered: {path.name}")
            st.rerun()
        except Exception as exc:
            st.error(f"Model import failed: {exc}")
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)


def _dataset_form(root):
    st.subheader("Import Dataset")
    st.write("Use class folders, a ZIP of class folders, or a CSV with image paths and labels. Images stay local.")
    with st.form("import_dataset_form"):
        name = st.text_input("Dataset name", placeholder="My image study")
        kind = st.selectbox("Dataset format", ("folder", "zip", "csv"),
            format_func=lambda value: {"folder": "Image folders", "zip": "ZIP archive",
                                       "csv": "CSV labels + image folder"}[value])
        folder_path = st.text_input("Image folder path") if kind == "folder" else None
        zip_path = st.text_input("Local ZIP path") if kind == "zip" else None
        zip_upload = st.file_uploader("Or upload a ZIP", type=("zip",)) if kind == "zip" else None
        csv_path = st.text_input("CSV path") if kind == "csv" else None
        csv_upload = st.file_uploader("Or upload a CSV", type=("csv",)) if kind == "csv" else None
        image_root = st.text_input("Image root folder", help="Paths in the CSV are relative to this folder") if kind == "csv" else None
        if kind in ("csv", "zip"):
            image_column = st.text_input("CSV image-path column", value="image")
            label_column = st.text_input("CSV label column", value="label")
            split_column = st.text_input("CSV split column (optional)")
            group_column = st.text_input("Patient/group ID column (optional)")
        else:
            image_column, label_column, split_column, group_column = "image", "label", "", ""
        split_mode = st.radio("Splits", ("generate", "existing"), horizontal=True,
            format_func=lambda value: "Generate with fixed seed" if value == "generate" else "Use assigned train/val/test")
        if split_mode == "generate":
            seed = st.number_input("Split seed", 0, 999999, 42)
            train_fraction = st.slider("Training fraction", .50, .85, .70, .05)
            val_fraction = st.slider("Validation fraction", .05, .30, .15, .05)
        else:
            seed, train_fraction, val_fraction = 42, .7, .15
        submitted = st.form_submit_button("Validate and import dataset", type="primary")
    if submitted:
        staged = None
        try:
            if not name.strip():
                raise ValueError("Enter a dataset name")
            source = folder_path if kind == "folder" else zip_path if kind == "zip" else csv_path
            if kind == "zip" and zip_upload is not None:
                if zip_upload.size > 1_000_000_000:
                    raise ValueError("ZIP upload exceeds the 1 GB local import limit")
                source = BytesIO(zip_upload.getvalue())
            if kind == "csv" and csv_upload is not None:
                stage = catalog_root(root) / "datasets" / ".uploaded_labels.csv"
                if stage.exists():
                    raise ValueError("Another CSV import is in progress; retry shortly")
                stage.write_bytes(csv_upload.getvalue())
                staged, source = stage, stage
            if not source:
                raise ValueError("Provide a folder, ZIP, or CSV source")
            bar = st.progress(0, text="Checking images and split separation")
            def progress(done, total):
                bar.progress(done / total, text=f"Validated images {done}/{total}")
            try:
                path = import_dataset(name.strip(), kind, source, root, image_root,
                    image_column, label_column, split_column or None, group_column or None,
                    split_mode, seed, train_fraction, val_fraction, progress)
            finally:
                bar.empty()
            st.success(f"Dataset imported: {path.parent.name}")
            st.rerun()
        except Exception as exc:
            st.error(f"Dataset import failed: {exc}")
        finally:
            if staged is not None:
                staged.unlink(missing_ok=True)


def render():
    load_local_factories()
    root = catalog_root()
    st.title("MedCNN Inspector")
    st.caption("One workspace for trained 2D CNNs, image datasets, and controlled interpretation experiments.")
    st.warning("Research tool only. Internal interventions can be out of distribution and do not establish clinical meaning.")
    models, datasets = list_models(root), list_datasets(root)
    dashboard, add_model, import_data, manage = st.tabs(("Workspace", "Add Model",
                                                          "Import Dataset", "Manage"))
    with add_model:
        _model_form(root)
    with import_data:
        _dataset_form(root)
    with manage:
        st.subheader("Manage local registrations")
        st.caption("Forgetting a registration leaves checkpoint and image files on disk.")
        for entry in [*models, *datasets]:
            if entry.get("managed") and st.button(f"Forget {entry['name']}", key=f"forget_{entry['id']}"):
                try:
                    forget_registration(entry["path"], root)
                    st.success(f"Forgot {entry['name']}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    with dashboard:
        left, right = st.columns(2)
        with left:
            _cards("Registered models", models, "Models")
        with right:
            _cards("Registered datasets", datasets, "Datasets")
        ready_models = [entry for entry in models if entry["ready"]]
        ready_datasets = [entry for entry in datasets if entry["ready"]]
        if ready_models and ready_datasets:
            st.subheader("Pair a model and dataset")
            left, right = st.columns(2)
            with left:
                selected_model = st.selectbox("Model", ready_models,
                    format_func=lambda entry: entry["name"])
            selected_spec = read_spec(selected_model["path"])
            matching = [i for i, entry in enumerate(ready_datasets)
                        if set(entry["classes"]) <= set(selected_spec.class_names)]
            with right:
                selected_dataset = st.selectbox("Dataset", ready_datasets,
                    index=matching[0] if matching else 0,
                    format_func=lambda entry: entry["name"])
            pair_key = (selected_model["id"], selected_dataset["id"],
                        selected_model["path"].stat().st_mtime_ns,
                        selected_dataset["path"].stat().st_mtime_ns
                        if selected_dataset["path"] else 0)
            if st.session_state.get("pair_key") != pair_key:
                try:
                    with st.spinner("Checking model outputs, image preprocessing, classes, and splits…"):
                        if not set(selected_dataset["classes"]) <= set(selected_spec.class_names):
                            raise ValueError("Dataset labels are absent from the selected model's class names")
                        model = load_model(selected_spec)
                        handle = DatasetHandle(selected_dataset)
                        report = validate_pair(selected_spec, handle, model)
                    st.session_state.pair_key = pair_key
                    st.session_state.pair_report = report
                    st.session_state.pair_handle = handle
                except Exception as exc:
                    st.session_state.pop("pair_key", None)
                    st.error(f"Model–dataset pairing failed: {exc}")
            if st.session_state.get("pair_key") == pair_key:
                report = st.session_state.pair_report
                st.success(f"Ready: {report['model']} + {report['dataset']}")
                st.caption(" · ".join(f"{split} {count}" for split, count in report["counts"].items()))
                for warning in report["warnings"]:
                    st.warning(warning)
                with st.expander("Compatibility details"):
                    st.write("**Technical compatibility:** passed model input, output, and label checks")
                    st.write("**Scientific compatibility:** user declarations and dataset metadata; not independently verified")
                    st.json(report["scientific_checklist"])
                    st.json(report["checked"])
                st.session_state.paired_dataset_id = selected_dataset["id"]
                render_analysis(selected_model["path"], st.session_state.pair_handle)
        else:
            st.info("Add a ready model and dataset to begin analysis.")
