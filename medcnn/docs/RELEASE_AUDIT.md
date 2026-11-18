# MedCNN Inspector release audit — 2026-11-18

## Ubuntu release gate

**Passed on 2026-11-18.** The private repository's [Ubuntu Actions run 44312116154](https://github.com/hamza-zougagh/MedCNN-Inspector/actions/runs/44312116154) completed successfully on `ubuntu-24.04` with Python 3.14.10. Its downloaded job log confirms a fresh install of pinned CPU PyTorch 2.15.0, Torchvision 0.29.2, the complete `requirements.lock.txt`, and the standalone `medcnn-inspector` wheel. `pip check` passed. Importing `medcnn` from a temporary directory resolved to the installed site-packages copy, not the checkout. The clean checkout had no checkpoints, datasets, caches, or presentation `src/` directory. Standalone tests: **26 passed, 8 expected checkpoint-dependent skips**. Streamlit launched and its health endpoint passed. The workflow does not download medical examples or exercise trained-model inference in CI; those checks remain documented in `VALIDATION.md` from the local research workspace.

The same log warned that `actions/checkout@v4` and `actions/setup-python@v5` target deprecated Node 20. The workflow now uses their Node 24 versions (`checkout@v5`, `setup-python@v6`). It also produced a nonfatal setuptools warning that `medcnn.docs` is absent from the explicit package list; the wheel nevertheless built, installed, and imported. Documentation is available in the repository. This packaging warning remains a cleanup item, not a failed check.

Before the Ubuntu run, the standalone Windows suite passed **34 tests**; a separate copy without checkpoints or datasets passed **26 tests with 8 expected skips**, and a live Streamlit health check returned HTTP 200. The ZIP and CSV import paths now reject backslashes consistently across operating systems.

## Passed checks

| Area | Evidence |
|---|---|
| Regression suite | 54 tests passed in the development workspace; 34 package tests passed in the standalone export. |
| Windows clean-copy preflight | Separate standalone copy without model checkpoints or data cache: 26 tests passed and 8 checkpoint-dependent tests skipped as intended. Wheel built, installed into an audit environment, and imported outside the checkout using existing installed dependencies. No `src/` dependency. The fresh Ubuntu install is documented above. |
| Real inference | Saved ResNet-18 and DenseNet-121 checkpoints completed dataset ablation, matched-donor patching, and class-label direction experiments. The custom multiclass test CNN completed all three with deterministic repeated outputs. |
| Other architectures | EfficientNet-B0 and VGG-16 produced 4D maps and completed real hook patching with exact same-image restoration; EfficientNet also completed an interaction experiment. Both used untrained test weights. |
| Scientific controls | Train/validation/test roles, selected-content overlap checks, same-image/zero-shift controls, matched channel count and donor label, paired image effects, image-level bootstrap, and hook/parameter restoration were tested. |
| Import and persistence | Restricted state-dict loading rejects whole-model pickle; model factory names are local and trusted. ZIP and CSV traversal checks, file-count/size limits, decoded image checks, atomic import, saved image hashes, registry reloading, and CSV/PNG/JSON exports passed tests. |
| Interface | AppTest imported a custom model and dataset, paired them, navigated analysis modules, ran all three dataset experiment types, and found locally saved exports. Training-mean references for paired datasets come from their training split. |
| Repository | Local checkpoints, datasets, experiment registry, caches, and environments are excluded by standalone `.gitignore`; docs screenshots remain trackable. Project MIT license and upstream MedMNIST/Torchvision attribution are present. The project was pushed to a **private** GitHub repository for CI; no public release was created. |

Upstream terms were checked against the [MedMNIST license and dataset statement](https://github.com/MedMNIST/MedMNIST/blob/main/README.md#license-and-citation), [MedMNIST code license](https://github.com/MedMNIST/MedMNIST/blob/main/LICENSE), and [Torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE). The loader's limits follow [PyTorch's `weights_only` documentation](https://docs.pytorch.org/docs/2.15/notes/serialization.html#weights-only-security).

## Fixes made in this audit

- Patching requires distinct primary and random donors with the same donor label, plus at least four test images per trial. Two-image trials no longer present a duplicate donor as an independent control.
- Imports validate decoded image content across splits before publishing, stage files atomically, and detect changed external images on later reads. Existing registered data and results were not deleted or migrated.
- One-trial variability is reported as unavailable. The evidence label now says **repeated test-image sensitivity**; training-reference ranking and concept-direction stability are not implied.
- Pairing reports class counts per split; unsupported layers, incomplete model specs, nonfinite model outputs, and oversized images receive clearer errors. Imported dataset rows are indexed by split for faster access.
- Generic training-mean replacement uses the paired dataset's training split. Saved pneumonia means are only reused with the built-in pneumonia dataset and matching checkpoint provenance.
- Completed experiment exports now reuse byte-identical saved results and preserve earlier artifacts when a rerun with the same configuration produces different bytes.
- ZIP and CSV image paths containing backslashes are rejected on every platform, avoiding Windows/Linux differences in path interpretation.

## Unresolved limitations and release gates

- Clinical validity, patient identity, external-site transportability, and naturalness of internal edits cannot be established by these software tests. Bootstrap intervals condition on one checkpoint and sampled images; they do not cover retraining or patient clustering without IDs. Scanning many layers is not multiple-testing corrected.
- EfficientNet-B0 and VGG-16 were not tested with trained medical checkpoints. Their compatibility tests establish execution and hook restoration only.
- Clean Ubuntu CI does not test trained medical checkpoints, downloaded PneumoniaMNIST data, a browser session, other Linux distributions, or GPU execution. These remain outside the first beta release gate.
- `weights_only=True` narrows deserialization risk but is not a sandbox or denial-of-service defense. Continue to import only trusted checkpoint files and trusted local factories.

For detailed methods and test provenance, see [METHODS.md](METHODS.md) and [VALIDATION.md](VALIDATION.md).
