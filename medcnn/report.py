def write_report(directory, config, metadata, baseline, probe, summary):
    from pathlib import Path

    rows = []
    for strategy, group in summary[summary.strategy != "baseline"].groupby("strategy"):
        for budget, trials in group.groupby("budget"):
            rows.append(
                f"| {strategy} | {budget} | {len(trials)} | {trials.mean_absolute_probability_change.mean():.5f} | "
                f"{trials.mean_signed_probability_change.mean():+.5f} | {trials.delta_roc_auc.mean():+.4f} |"
            )
    comparisons = []
    for budget in config["budgets"]:
        group = summary[summary.budget == budget]
        top = group[group.strategy == "top"].mean_absolute_probability_change.iloc[0]
        random = group[group.strategy == "random"].mean_absolute_probability_change
        comparisons.append(
            f"Budget {budget}: top mean absolute change {top:.5f}; random mean {random.mean():.5f}, "
            f"SD {random.std(ddof=1):.5f}. Top {'exceeds' if top > random.mean() else 'does not exceed'} the random mean."
        )
    text = f"""# Investigating Internal Feature Representations and Their Causal Influence on Pneumonia Classification in CNNs

## Introduction
Which internal CNN representations are associated with pneumonia, and how do predictions change when those channels are suppressed? The hypothesis is that training-ranked pneumonia-associated channels have greater intervention effects than matched random channels. This is an empirical hypothesis.

## Methods
Official PneumoniaMNIST train/validation/test splits, 64 pixel source images, metadata-verified labels: {metadata["official_labels"]}.
Preprocessing: {metadata["preprocessing"]}.
Architecture: {config["architecture"]}, frozen ImageNet IMAGENET1K_V1 backbone. A standardized logistic classifier was fit on pooled training representations, selected among C=0.01,0.1,1 using validation ROC-AUC, and folded into a single PyTorch linear head. The threshold maximized validation Youden J. No test selection was performed.
Seed {config["seed"]}; sample limits {config["limits"]}. This {"subset smoke experiment" if any(config["limits"].values()) else "full split experiment"} uses one seed and is exploratory.
Intermediate hook outputs were spatially averaged. Training-only channel ranks use signed standardized class mean differences with a variance floor. A regularized linear probe uses training scaling and validation selection independently of the original head.
Interventions clone hook outputs and zero channels before downstream computation, with the original weights and head fixed. Budgets {config["budgets"]}; {config["repeats"]} random repetitions. All conditions use the same {config["ablation_size"]} stratified held-out test images (or the full test split if smaller). Bottom signed ranks can be strongly normal associated; they are not necessarily unimportant.

## Results
Baseline test metrics: {baseline}

Probe held-out metrics: {probe}

| Strategy | Budget | Trials | Mean absolute Δp | Mean signed Δp | ΔROC-AUC |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

{chr(10).join(comparisons)}

![ROC and PR curves](curves.png)
![Channel associations and activation distributions](channel_analysis.png)
![Representative training activations](activation_examples.png)
![Controlled ablation comparison](ablation_comparison.png)

## Discussion
These descriptive comparisons assess the stated hypothesis separately at each budget. They do not establish population-level significance. Random-control SD describes selection variability, not uncertainty across patients. Association is measured against dataset labels; intervention effects measure dependence of this particular model on a modified representation. This does not identify clinical concepts.
{"Baseline discrimination or sensitivity/specificity is limited; interpretability claims should be treated cautiously." if baseline["roc_auc"] < 0.75 or min(baseline["sensitivity"], baseline["specificity"]) < 0.6 else "The classifier generalizes on this held-out split; external clinical validity remains untested."}
Zeroing activations may create out-of-distribution internal states. Correlated channels, redundancy, downstream normalization, acquisition artifacts, and distribution differences can explain effects. Single-channel rankings need not predict joint ablation effects. No patient-level leakage audit beyond the official splits or external validation was performed. Probability shifts can be positive or negative.

## Conclusion
The experiment produced the measured comparisons above. Findings apply to this checkpoint, sampling configuration, and zero-ablation operator. A useful next study compares zero ablation with training-mean replacement and evaluates uncertainty across repeated stratified samples, with the design fixed on validation data.
"""
    Path(directory, "report.md").write_text(text, encoding="utf-8")
