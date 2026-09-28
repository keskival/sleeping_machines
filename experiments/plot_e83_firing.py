"""Create standalone SHD firing visualizations; these figures are not in REPORT.pdf."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(__file__)
LOG_DIR = os.path.join(ROOT, "queue", "logs")
RESULT_DIR = os.path.join(ROOT, "results", "e83")


def read_epoch_logs():
    runs = {}
    for path in glob.glob(os.path.join(LOG_DIR, "e83_d4_*seed2.log")):
        label = os.path.basename(path)[len("e83_d4_"):-len("_seed2.log")]
        rows = []
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "epoch" in row and "spikes_per_utt" in row:
                    rows.append(row)
        if rows:
            runs[label] = rows
    return runs


def plot_activity(runs):
    if not runs:
        raise SystemExit("No E83 depth-4 epoch logs with spike counts found.")
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(runs))))
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.0), constrained_layout=True)

    for (label, rows), color in zip(sorted(runs.items()), colors):
        epochs = np.asarray([r["epoch"] for r in rows])
        counts = np.asarray([r["spikes_per_utt"] for r in rows], dtype=float)
        # Layer 4 is the last spiking layer; the classifier readout is non-spiking.
        axes[0].plot(epochs, counts[:, -1], marker="o", linewidth=2.4,
                     color=color, label=label)
        for layer in range(counts.shape[1] - 1):
            axes[1].plot(epochs, counts[:, layer], marker=".", linewidth=1.1,
                         alpha=0.38, color=color,
                         label=f"{label}, L{layer + 1}")
        axes[1].plot(epochs, counts[:, -1], marker="o", linewidth=2.3,
                     color=color, label=f"{label}, L{counts.shape[1]}")
        max_acc = [r.get("shd_max_over_time_acc") for r in rows]
        race_cov = [r.get("race_coverage") for r in rows]
        race_acc = [r.get("race_acc_when_emitted") for r in rows]
        axes[2].plot(epochs, max_acc, marker="s", linewidth=2,
                     color=color, label=f"{label}, max-readout accuracy")
        axes[2].plot(epochs, race_cov, marker="^", linestyle="--", alpha=0.7,
                     color=color, label=f"{label}, race coverage")
        axes[2].plot(epochs, race_acc, marker="x", linestyle=":", alpha=0.8,
                     color=color, label=f"{label}, emitted accuracy")

    axes[0].set_title("Final event layer activity during training")
    axes[0].set_ylabel("Layer 4 spikes / utterance")
    axes[0].set_xlabel("Training epoch (one pass over the training subset)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(title="Objective", ncol=2, fontsize=8)
    axes[1].set_title("Firing activity through the event hierarchy")
    axes[1].set_ylabel("Spikes / utterance")
    axes[1].set_xlabel("Training epoch")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=3)
    axes[2].set_title("Classification and output behavior")
    axes[2].set_ylabel("Validation fraction")
    axes[2].set_xlabel("Training epoch")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].axhline(0.05, color="black", linewidth=0.8, linestyle="--",
                    alpha=0.6, label="20-class chance")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=7, ncol=2)
    fig.suptitle("E83 depth-4 SHD validation firing counts", fontsize=14)
    path = os.path.join(RESULT_DIR, "e83_final_layer_activity.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_raster():
    candidates = []
    for path in glob.glob(os.path.join(RESULT_DIR, "*depth4*seed2.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("final_layer_firing"):
            candidates.append((os.path.getmtime(path), path, data))
    if not candidates:
        return None
    _, source, data = max(candidates)
    epochs = data["final_layer_firing"]
    nrows, ncols = len(epochs), max(len(ep["samples"]) for ep in epochs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows),
                             squeeze=False, sharex=True, sharey=True,
                             constrained_layout=True)
    xmax = 0.0
    for ri, epoch in enumerate(epochs):
        for ci in range(ncols):
            ax = axes[ri, ci]
            if ci >= len(epoch["samples"]):
                ax.axis("off")
                continue
            sample = epoch["samples"][ci]
            spikes = np.asarray(sample["spikes"], dtype=float).reshape(-1, 2)
            if len(spikes):
                ax.scatter(spikes[:, 0], spikes[:, 1], s=7, alpha=0.75,
                           color=plt.cm.viridis(0.72))
                xmax = max(xmax, float(spikes[:, 0].max()))
            ax.set_title(f"epoch {epoch['epoch']} · item {sample['eval_index']}\n"
                         f"true {sample['true_class']} / predicted {sample['predicted_class']}",
                         fontsize=8)
            ax.grid(alpha=0.2)
            if ci == 0:
                ax.set_ylabel("Layer 4 unit")
            if ri == nrows - 1:
                ax.set_xlabel("Time from first input spike (ms)")
    for ax in axes.flat:
        ax.set_xlim(0, max(1.0, xmax * 1.02))
        ax.set_ylim(-0.5, data["final_layer_firing"][0]["unit_count"] - 0.5)
    scheme = data.get("args", {}).get("objective", "unknown")
    fig.suptitle(f"Layer 4 firing raster over training · objective={scheme}\n"
                 "Fixed first four held-out utterances; final classifier readout is non-spiking",
                 fontsize=12)
    path = os.path.join(RESULT_DIR, "e83_final_layer_firing_raster.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)
    for output in (plot_activity(read_epoch_logs()), plot_raster()):
        if output:
            print(output)
