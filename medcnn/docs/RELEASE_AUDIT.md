# MedCNN Inspector release audit — 2026-11-18

## Ubuntu release gate

**Pending.** This Windows machine has no Ubuntu runtime: WSL is not installed and Docker/Podman are unavailable. No fresh Ubuntu dependency installation, Linux test run, or Linux Streamlit launch is claimed here. The standalone repository now contains `.github/workflows/ubuntu-release-gate.yml` for commits and pull requests. It checks out only tracked project files, confirms that checkpoints, datasets, caches, and the presentation `src/` directory are absent, installs Python 3.14 and the pinned CPU PyTorch and lock-file dependencies, installs the package, imports it outside the checkout, runs standalone tests, and checks the live Streamlit health endpoint. The public beta release gate remains open until this workflow succeeds on Ubuntu.

The pinned PyTorch and Torchvision CPU versions were found on their package index from Windows. This confirms index listing only; it does not verify Ubuntu wheels or installation. The ZIP and CSV import paths now reject backslashes consistently across operating systems. After this change, the standalone Windows suite passed **34 tests**; a separate copy without checkpoints or datasets passed **26 tests with 8 expected skips**, and a live Streamlit health check returned HTTP 200. These are Windows preflight results, not Linux validation.

## Passed checks

| Area | Evidence |
|---|---|
| Regression suite | 54 tests passed in the development workspace; 34 package tests passed in the standalone export. |
| Clean checkout | Separate standalone copy without model checkpoints or data cache: 26 tests passed and 8 checkpoint-dependent tests skipped as intended. Wheel built, installed into an audit environment, and imported outside the checkout using existing installed dependencies. No `src/` dependency. |
| Real inference | Saved ResNet-18 and DenseNet-121 checkpoints completed dataset ablation, matched-donor patching, and class-label direction experiments. The custom multiclass test CNN completed all three with deterministic repeated outputs. |
| Other architectures | EfficientNet-B0 and VGG-16 produced 4D maps and completed real hook patching with exact same-image restoration; EfficientNet also completed an interaction experiment. Both used untrained test weights. |
| Scientific controls | Train/validation/test roles, selected-content overlap checks, same-image/zero-shift controls, matched channel count and donor label, paired image effects, image-level bootstrap, and hook/parameter restoration were tested. |
| Import and persistence | Restricted state-dict loading rejects whole-model pickle; model factory names are local and trusted. ZIP and CSV traversal checks, file-count/size limits, decoded image checks, atomic import, saved image hashes, registry reloading, and CSV/PNG/JSON exports passed tests. |
| Interface | AppTest imported a custom model and dataset, paired them, navigated analysis modules, ran all three dataset experiment types, and found locally saved exports. Training-mean references for paired datasets come from their training split. |
| Repository | Local checkpoints, datasets, experiment registry, caches, and environments are excluded by standalone `.gitignore`; docs screenshots remain trackable. Project MIT license and upstream MedMNIST/Torchvision attribution are present. No publish or push occurred. |

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
- The clean checkout used already installed pinned dependencies for tests and wheel import. A fresh dependency download/install on Linux was not executed. **The first public beta remains blocked until the Ubuntu workflow passes.**
- `weights_only=True` narrows deserialization risk but is not a sandbox or denial-of-service defense. Continue to import only trusted checkpoint files and trusted local factories.

For detailed methods and test provenance, see [METHODS.md](METHODS.md) and [VALIDATION.md](VALIDATION.md).
