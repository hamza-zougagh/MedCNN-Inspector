# Methods and interpretation

## What each view measures

Feature Discovery ranks images by actual intermediate activations and clusters pooled features. These are associations. Activation Patching transfers donor activations to a recipient, using a same-image patch as a restoration control. Concept Explorer fits a regularized linear direction on user-labeled examples and evaluates it on disjoint examples. Feature Interaction Lab measures individual and paired channel effects and their nonadditivity. None identifies a complete circuit or a clinical feature.

## General dataset experiments

Each trial samples held-out test recipients with a fixed seed. Every intervention and its controls use the same recipients and target-class probability. The raw CSV records image ID, trial, baseline, intervention, random control, baseline control, effects, and donor/channel IDs where applicable.

- **Channel ablation:** A balanced training-split reference sample ranks channels by standardized target-versus-other label association and computes training means. Selected channels are set to zero or their training mean. A seeded random channel set of the same size is the matched control. Ranking is observational; ablation measures checkpoint sensitivity.
- **Activation patching:** Each recipient receives a patch from an opposite-label test donor. A distinct randomly chosen donor with the **same donor label** is the matched control; a same-image patch checks exact baseline restoration. Trials need at least two test images in both the target and another class. All patches use the same selected channels. Donor labels are a comparison design, not a causal disease claim.
- **Class-label concept direction:** A regularized logistic probe fits pooled activations from target versus other-class **training** images and is evaluated on separate **validation** images. A positive direction shift on test images is compared with a same-norm seeded random direction and a zero shift. This direction reflects dataset labels, not a clinical concept. User-defined concept examples remain available in Concept Explorer.

The experiment checks selected image contents for duplicates across reference training, probe validation, and intervention test sets. Group-disjoint imported splits rely on supplied group IDs; their correctness as patient identifiers cannot be established by software. The official PneumoniaMNIST splits have no independent patient-level audit in this tool.

## Uncertainty and stability

The reported effect is the mean paired change in target-class probability. The control comparison is the intervention effect minus the matched random effect on the same images. For each unique recipient, effects are averaged across appearances in repeated trials; bootstrap resampling then draws **images**, not rows. The 95% percentile interval describes sampling variability for this fixed model and dataset, conditional on the selected protocol. It does not include retraining variability, hospital shift, or patient clustering when patient IDs are absent.

The report also gives between-image and between-trial standard deviations, mean absolute effect, and fractions of images/trials whose effect sign agrees with the overall mean. Trial variability and consistency are **not measurable from one trial**, and sign consistency is undefined when the overall mean is zero. The label **repeated test-image sensitivity** requires at least three trials, 20 distinct test images, and intervals excluding zero for both the effect and matched-control difference. Otherwise the label is **exploratory**. Channel ranks and concept directions are fitted once per run; their stability across training resamples is not measured. This threshold is an interface aid, not a formal discovery claim or multiple-testing correction. Scanning many layers or channels can inflate apparent findings.

Internal zeroing, mean replacement, patching, and direction shifts can create activations outside the training distribution. A stable prediction change is evidence only about this checkpoint under that edit. It does not establish a clinical mechanism, lesion location, biological cause, or diagnostic utility. Do not use the software for patient care.
