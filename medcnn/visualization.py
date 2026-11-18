from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def save_curves(labels, probabilities, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="Sensitivity", title="ROC")
    axes[1].plot(recall, precision)
    axes[1].axhline(np.mean(labels), linestyle="--", color="gray")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_analysis(ranking, features, labels, summary, directory):
    directory = Path(directory)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    selected = ranking.head(12)
    axes[0].bar(selected.channel.astype(str), selected.association)
    axes[0].set(
        xlabel="Channel",
        ylabel="Standardized mean difference",
        title="Training class association",
    )
    channel = int(selected.iloc[0].channel)
    for label, name in ((0, "normal"), (1, "pneumonia")):
        axes[1].hist(
            features[labels == label, channel],
            bins=20,
            density=True,
            alpha=0.5,
            label=name,
        )
    axes[1].legend()
    axes[1].set(
        title=f"Training activation distribution, channel {channel}",
        xlabel="Mean activation",
    )
    fig.tight_layout()
    fig.savefig(directory / "channel_analysis.png", dpi=150)
    plt.close(fig)
    if summary is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for strategy, group in summary[summary.strategy != "baseline"].groupby(
            "strategy"
        ):
            for ax, field in zip(
                axes,
                ("mean_absolute_probability_change", "mean_signed_probability_change"),
            ):
                stats = group.groupby("budget")[field].agg(["mean", "std"]).fillna(0)
                ax.errorbar(
                    stats.index,
                    stats["mean"],
                    yerr=stats["std"],
                    marker="o",
                    capsize=3,
                    label=strategy,
                )
                ax.set(xlabel="Channels ablated", ylabel=field.replace("_", " "))
                ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "ablation_comparison.png", dpi=150)
        plt.close(fig)


def save_examples(model, dataset, ranking, path):
    import torch

    from .features import capture

    examples = [int(np.flatnonzero(dataset.labels == label)[0]) for label in (0, 1)]
    channels = ranking.head(3).channel.to_list()
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for row, index in enumerate(examples):
        image = dataset.original(index)
        with torch.inference_mode(), capture(model.target_layer) as cap:
            model(dataset[index][0].unsqueeze(0))
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"Training label: {dataset.labels[index]}")
        for col, channel in enumerate(channels, 1):
            activation = torch.nn.functional.interpolate(
                cap["activation"][:, channel : channel + 1],
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            axes[row, col].imshow(image, cmap="gray")
            axes[row, col].imshow(activation.numpy(), cmap="magma", alpha=0.6)
            axes[row, col].set_title(f"Channel {channel}")
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
