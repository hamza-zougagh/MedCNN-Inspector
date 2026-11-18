# MedCNN Inspector validation — 2026-11-17

## Full-data model runs

Both heads use the complete official PneumoniaMNIST 64 splits: **4,708 training, 524 validation, 624 test images**. Their ImageNet backbones were frozen. The ResNet-18 checkpoint was imported from the verified presentation run; its source files and results were not changed. DenseNet-121's medical head was fitted with the self-contained MedCNN pipeline. Head regularization and classification threshold were selected on validation, followed by held-out test evaluation.

| Held-out measure | ResNet-18 | DenseNet-121 |
|---|---:|---:|
| Accuracy | 0.8910 | 0.8958 |
| Balanced accuracy | 0.8615 | 0.8714 |
| ROC-AUC | 0.9710 | 0.9767 |
| Sensitivity | 0.9795 | 0.9692 |
| Specificity | 0.7436 | 0.7735 |

These are single-checkpoint results on one dataset. The DenseNet result is a software-validation result, not a claim of clinical performance or superiority to ResNet.

## Executed checks

| Check | Result |
|---|---|
| Automated tests | `.venv\Scripts\python.exe -m pytest tests medcnn/tests -q -p no:unraisableexception`: **54 passed**, exit code 0. |
| Independent package tests | Toy CNN ranking, real forward-hook interventions, paired experiment rows, no leaked hooks or parameter changes. |
| Package build | `pip wheel . --no-deps --no-build-isolation` succeeded; wheel contains `medcnn/` and no `src/` files. |
| Exported standalone project | `MedCNN-Inspector/` contains its own package, app, lock file, license, docs, tests, model specs, and local full-data checkpoints. It contains no `src/` directory; **34 standalone tests passed** after export refresh. |
| Clean standalone checkout | A separate copy containing package, app, specs, docs, lock file, and license—but no checkpoints or data cache—ran **26 tests passed, 8 expected skips** for unavailable trained examples, and built an installable wheel. The wheel was installed in an audit environment and imported from outside the checkout using the existing dependency environment. |
| All supported layers | Training-only rankings and means prepared for ResNet-18 `layer1`–`layer4` and DenseNet-121 `features.denseblock1`–`features.denseblock4` with one CPU pass per model. |
| Real per-layer validation | Each layer passed zero and training-mean interventions on four held-out images: **16 conditions** across the two models. Baseline logits were exactly restored, hooks were removed, parameters were unchanged, and outputs were finite. |
| Full-model paired experiments | Each architecture completed 20 conditions on the same stratified 32-image test subset: **640 real prediction rows** per model. CSV, PNG, and JSON outputs were saved locally. |
| Automatic layer preparation | Selecting an unprepared DenseNet smoke-run layer in Streamlit computed and cached its training statistics without an app exception. |
| Streamlit workflow | AppTest traversed both full-data models through the unified pairing flow and all five original generic modules plus Dataset Experiments. It completed real feature extraction, patching, interaction intervention, and a prior controlled pneumonia experiment without exceptions. |
| Model and dataset import UI | AppTest registered a trusted custom 2D CNN state dict and imported a 20-image folder dataset through the forms, paired them, opened all analysis modules, and ran/saved ablation, patching, and concept dataset experiments without exceptions. |
| Dataset formats and leakage checks | Automated checks imported image folders, ZIP archives, and CSV labels; generated seeded stratified or group-disjoint splits; rejected unsafe ZIP and CSV traversal, cross-split decoded-content duplicates, and invalid assigned splits. Imports publish only after validation; later source-image changes are detected. ZIP images reopen from persistent relative references. |
| Live dashboard | Local Streamlit server rendered the workspace and general experiment screen; `medcnn_workspace.png` and `medcnn_dataset_experiments.png` capture the current interface. Earlier screenshots remain as evidence of prior analysis views. |
| General model engine | Tested registered custom multiclass CNN and restricted state-dict loading. Pickled whole-model checkpoint loading was rejected. |
| Requested CNN families | Trained ResNet-18 and DenseNet-121 completed real inference across all three generalized intervention types. EfficientNet-B0 and VGG-16 completed 4D discovery, real forward-hook patching, exact same-image restoration, and finite outputs; EfficientNet also completed a paired interaction check. EfficientNet and VGG used **untrained** weights, so no performance claim is made for them. |
| Six general workspaces | Streamlit AppTest rendered the five original analysis modules and generalized Dataset Experiments; actual extraction, patching, interaction, and all three dataset experiment actions completed. |
| Real generic analysis | Both saved full-data checkpoints completed 4D discovery, feature ranking/clustering, donor patching with exact same-image control, class-target sensitivity scans, a train/validation-separated label-direction probe, concept-direction intervention, and paired channel-mean interaction checks. |
| General experiment engine | Ablation, patching, and class-label direction experiments each completed on a custom multiclass CNN with random test weights and on both saved trained pneumonia CNNs, using real inference. Two identical seeded runs on the custom CNN yielded identical rows, reports, and configurations. No performance claim is made for the custom test model. |
| Reliability and controls | Bootstrap tests confirmed deterministic intervals and constant-effect exact intervals. Repeated-image summaries resampled unique images; one-trial variability is reported as unavailable. Patching controls use distinct donors matched on label and channels. Per-image baseline controls matched baseline exactly; hook and parameter restoration checks passed. Cross-split duplicate image content was rejected. |

Saved run artifacts live in `medcnn_runs/resnet18_full/` and `medcnn_runs/densenet121_full/`. Their `model.pt`, `config.json`, `test_metrics.json`, and `layers/` statistics remain available locally while excluded from Git. The per-layer reports are `layer_validation.json`. Controlled-experiment outputs are in `medcnn_runs/resnet18_full_experiment/` and `medcnn_runs/densenet121_full_experiment/`. Configuration files record official selected test IDs, checkpoint hashes, seeds, layer names, budgets, repeats, methods, and validation-selected thresholds.

The generic checks discovered 66 4D module outputs in the saved ResNet-18 and 431 in the saved DenseNet-121. Each workbench report (`workbench_validation.json`) records observed probabilities, top-feature examples, cluster counts, concept probe measurements, patch controls, interaction counts, and restoration checks. The concept demonstration used four training images per dataset label and two validation images per label; its ROC-AUC of 0.75 in both runs is a **tiny software check**, not evidence of a validated clinical concept. The held-out test images used for patching and intervention were separate from concept fitting and evaluation.

## Limits

The tests establish real inference and reproducible software behavior. They do not establish clinical validity, lesion localization, patient-level independence beyond the dataset's official splits, or generalization to external hospitals. Channel association uses labels; intervention sensitivity measures this checkpoint's response to changed internal values. Neither establishes a clinical feature or a causal disease mechanism. A zero or mean replacement can place internal activations outside the model's training distribution.

Generic support is limited to trusted local factories and 2D classification models with tensor outputs and 4D intermediate maps. EfficientNet-B0 and VGG-16 were tested with untrained weights for hooks and output restoration; trained state dicts for them were not supplied. Concept labels and reference training images must be supplied by the researcher. Random-direction and same-image controls test implementation behavior, but internal activation edits can still be out of distribution.

Model–dataset pairing verifies label names, image decoding and preprocessing shape, and actual prediction output on one image from each split. It cannot infer the training-time normalization, class order, or data provenance from a `state_dict`; users must enter and verify these. Imported folder and CSV datasets reference external image paths, so moving those source folders requires reimporting. ZIP imports copy images into the local registry and keep relative references. A supplied patient/group column is checked for cross-split overlap; folders without such IDs cannot receive a patient-level leakage audit.

The two supplied pneumonia model specifications now record their known dataset provenance and preprocessing/class-order declarations from the saved run configuration and project pipeline. The compatibility screen identifies these as declarations, not independent verification of patient identity or clinical transportability.

The image-bootstrap intervals condition on the fixed checkpoint and selected dataset. Repeated trials may reuse the same test images; the implementation averages those appearances before resampling images. The intervals do not account for retraining, hospital shift, or patient clustering when patient IDs are unavailable. The class-label concept experiment fits a label-associated direction and is not evidence of an independently defined clinical concept. The "repeated test-image sensitivity" interface label is based on a practical threshold and is not a multiple-testing correction or clinical validity claim. Channel ranking and concept-direction stability across training resamples are not measured. The custom CNN tests use random weights to validate software behavior only.
