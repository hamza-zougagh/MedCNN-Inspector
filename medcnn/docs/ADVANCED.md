# Advanced setup and integration

## Trusted CNN factories

Save a tensor state dict with `torch.save(model.state_dict(), "my_model.pt")`. Built-in factory keys are `resnet18`, `densenet121`, `efficientnet_b0`, and `vgg16`. Their final classification layer must match the class count in **Add Model**. For a trusted custom `nn.Module`, edit `medcnn/local_models.py` locally, import its definition, and register `register_factory("my_cnn", lambda classes: MyCNN(num_classes=classes))`; restart Streamlit. The UI never imports uploaded Python. Alternatively, create a local JSON spec from [the template](custom_model_spec.example.json).

The model loader requires a tensor output shaped `N,C`, or one output for binary classification, and named modules producing 4D feature maps. It uses strict state-dict matching and restricted PyTorch loading. Image preprocessing is resize, tensor conversion, and per-channel normalization, applied identically to baseline and intervened inference. CPU is the default; CUDA is optional.

## Dataset layouts and persistence

Folder imports accept `class/image.png` or `train/class/image.png`, `val/class/image.png`, and `test/class/image.png`. ZIP imports accept the same layout, optionally with one CSV manifest containing relative image paths. CSV imports need image-path and label columns and can include split and patient/group columns. Generated splits are stratified by label when no groups are provided and group-disjoint when every row has a group ID. The importer rejects duplicate paths, missing splits, unreadable or oversized images, cross-split image-content duplicates, and cross-split group reuse. Import writes to a temporary directory and publishes a registration only after validation.

ZIP images are copied to `workspace_registry/datasets/` with relative references. Folder and CSV imports retain external source paths, so moving those sources requires reimporting. New imports record decoded-content hashes; later source-image changes are flagged when used. Older saved imports without hashes remain readable and experiments still check their selected images across splits. Model configs, managed checkpoint copies, and completed experiment artifacts stay under `workspace_registry/`. This directory is ignored by Git. The **Manage** tab forgets registrations without deleting source images or checkpoints.

Completed experiment saves reuse an existing run only when all exported bytes match. If a rerun with the same configuration produces different results, it receives a numbered directory and leaves the earlier run intact.

`torch.load(..., weights_only=True)` restricts the unpickler and the UI never executes uploaded Python, but this is **not a sandbox** for arbitrary malicious checkpoints. Import only checkpoints from a trusted source. The upload limit is 1.5 GB; image uploads and imports also have size limits. A trusted custom factory is local Python code and executes with the user's permissions.

## Optional pneumonia examples

The repository can launch without preparing the examples. To generate a missing full-data checkpoint from official PneumoniaMNIST splits, run from the project root:

```powershell
.\.venv\Scripts\python.exe -m medcnn.prepare --architecture resnet18 --output medcnn_runs/resnet18_full
.\.venv\Scripts\python.exe -m medcnn.prepare --architecture densenet121 --output medcnn_runs/densenet121_full --batch-size 8
```

Preparation downloads official data and Torchvision weights as needed, freezes the backbone, and fits only the medical head. Matching local checkpoints are reused. Prepare standard layer statistics from an existing checkpoint with `python -m medcnn.prepare --existing --architecture resnet18 --output medcnn_runs/resnet18_full --all-layers`. The verified presentation ResNet checkpoint can be copied through `python -m medcnn.import_legacy --source results/resnet18 --target medcnn_runs/resnet18_full` in the parent workspace; this does not alter its source.

The earlier pneumonia-specific CLI remains available:

```powershell
.\.venv\Scripts\python.exe -m medcnn.experiment --run medcnn_runs/resnet18_full --output medcnn_runs/study1 --sample-size 64 --budgets 1 4 8 --repeats 3 --seed 42
```

The general **Dataset Experiments** screen works with imported compatible models and datasets. It saves per-image CSV, summary JSON, comparison PNG, and a configuration JSON containing checkpoint hash, selected image IDs, split roles, layer, seed, controls, and preprocessing. The previous pneumonia CLI and its saved results remain intact.
