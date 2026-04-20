"""
Plot training curves for DINO from its JSON log file.

Usage:
    python plot_training_curves.py                        # auto-find latest log
    python plot_training_curves.py --dino log_DINO/train_DINO_20260419_082942_merged.json
    python plot_training_curves.py --out training_curves.png
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def load_log(path):
    with open(path) as f:
        return json.load(f)


def find_latest_log(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def extract_metrics(log):
    epochs = log["epochs"]
    result = {
        "epoch": [e["epoch"] for e in epochs],
        "train_loss": [e["train_loss"] for e in epochs],
        "val_map": [e["val_map"] for e in epochs],
    }
    for key in ("train_cls_loss", "train_bbox_loss", "train_giou_loss", "train_dn_loss"):
        if key in epochs[0]:
            result[key] = [e[key] for e in epochs]
    return result


def smooth(values, window=5):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def plot_curves(dino_metrics, out_path):
    has_dn = "train_dn_loss" in dino_metrics

    n_rows = 2  # total loss + val mAP
    n_rows += 1  # cls + bbox + giou
    if has_dn:
        n_rows += 1

    fig = plt.figure(figsize=(10, 4 * n_rows))
    fig.suptitle("DINO Training Curves", fontsize=16, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(n_rows, 1, hspace=0.45)

    color = "#F44336"
    lw = 1.5

    def _plot(ax, x, y, label, smooth_win=5, style="-"):
        y_arr = np.array(y, dtype=float)
        ax.plot(x, y_arr, style, color=color, alpha=0.25, linewidth=0.8)
        ax.plot(x, smooth(y_arr, smooth_win), style, color=color,
                linewidth=lw, label=label)

    row = 0

    ax = fig.add_subplot(gs[row])
    _plot(ax, dino_metrics["epoch"], dino_metrics["train_loss"], "DINO")
    ax.set_title("Total Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    row += 1

    ax = fig.add_subplot(gs[row])
    component_styles = {
        "train_cls_loss": ("--", "Cls"),
        "train_bbox_loss": ("-.", "BBox"),
        "train_giou_loss": (":", "GIoU"),
    }
    for key, (style, label_suffix) in component_styles.items():
        if key in dino_metrics:
            _plot(ax, dino_metrics["epoch"], dino_metrics[key],
                  f"DINO {label_suffix}", style=style)
    ax.set_title("Loss Components (Cls / BBox / GIoU)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    row += 1

    if has_dn:
        ax = fig.add_subplot(gs[row])
        _plot(ax, dino_metrics["epoch"], dino_metrics["train_dn_loss"], "DINO DN Loss")
        ax.set_title("DINO Contrastive DeNoising Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        row += 1

    ax = fig.add_subplot(gs[row])
    _plot(ax, dino_metrics["epoch"], dino_metrics["val_map"], "DINO", smooth_win=3)
    best_ep = dino_metrics["epoch"][np.argmax(dino_metrics["val_map"])]
    best_val = max(dino_metrics["val_map"])
    ax.axvline(best_ep, color=color, linestyle=":", alpha=0.6)
    ax.annotate(f"DINO best\n{best_val:.3f}",
                xy=(best_ep, best_val), xytext=(best_ep + 1, best_val - 0.02),
                fontsize=8, color=color)
    ax.set_title("Validation mAP (COCO, IoU 0.5:0.95)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino", default=None,
                        help="Path to DINO JSON log (default: latest in log_DINO/)")
    parser.add_argument("--out", default="training_curves.png",
                        help="Output figure path")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))

    dino_path = args.dino or find_latest_log(os.path.join(base, "log_DINO", "train_DINO_*.json"))

    if not dino_path or not os.path.isfile(dino_path):
        print("No DINO log found.")
        return

    print(f"Loading DINO log: {dino_path}")
    dino_metrics = extract_metrics(load_log(dino_path))

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(base, out_path)

    plot_curves(dino_metrics, out_path)


if __name__ == "__main__":
    main()
