"""Plotting helpers."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from Online_Machine_learning_Lab.src.config import ROLLING_WINDOW


def rolling(values, n):
    window = min(ROLLING_WINDOW, max(20, n // 10))
    return pd.Series(values).rolling(window, min_periods=1).mean(), window


def plot_rolling_losses(curves, datasets, ylabel, title, path):
    """curves[(dataset, model)] = per-instance losses. One subplot per dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, ds in zip(axes.flat, datasets):
        for (d, model), losses in curves.items():
            if d == ds:
                series, window = rolling(losses, len(losses))
                ax.plot(series, label=model, lw=1)
        ax.set_title(f"{ds} (rolling window={window})")
        ax.set_xlabel("instance")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_drifts(dataset, curves, detections, true_drifts, ylabel, path):
    """Top: rolling loss for baseline / reset-on-drift variants. Bottom: detected drifts."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    for name, losses in curves.items():
        series, window = rolling(losses, len(losses))
        ax1.plot(series, label=name, lw=1)
    ax1.set_ylabel(f"{ylabel} (rolling {window})")
    ax1.set_title(f"{dataset}: performance over time and detected drifts")
    ax1.legend(fontsize=8)

    names = list(detections)
    for k, name in enumerate(names):
        ax2.scatter(detections[name], [k] * len(detections[name]), marker="|", s=200)
    ax2.set_yticks(range(len(names)), names)
    ax2.set_xlabel("instance")
    for ax in (ax1, ax2):
        for p in true_drifts:
            ax.axvline(p, color="red", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
