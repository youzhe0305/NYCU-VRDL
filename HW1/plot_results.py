import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def shorten(text: str, max_len: int = 52) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot experiment progress from results.tsv",
    )
    parser.add_argument("--input", default="results.tsv", help="Path to results.tsv")
    parser.add_argument(
        "--output", default="results_progress.png", help="Output image path (.png)"
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        default=True,
        help="Annotate kept points with descriptions",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    df = pd.read_csv(in_path, sep="\t")
    df = df.copy()
    df["exp_idx"] = range(len(df))

    # Use validation accuracy directly (higher is better).
    df["metric"] = df["val_acc"].astype(float)
    df["is_keep"] = df["status"].astype(str).str.strip().eq("keep")
    df["running_best"] = df["metric"].cummax()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    discard_mask = ~df["is_keep"]
    keep_mask = df["is_keep"]

    ax.scatter(
        df.loc[discard_mask, "exp_idx"],
        df.loc[discard_mask, "metric"],
        s=18,
        color="#bfc3c7",
        alpha=0.8,
        label="Discarded",
        edgecolors="none",
        zorder=2,
    )
    ax.scatter(
        df.loc[keep_mask, "exp_idx"],
        df.loc[keep_mask, "metric"],
        s=34,
        color="#2ca25f",
        alpha=0.95,
        label="Kept",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    ax.step(
        df["exp_idx"],
        df["running_best"],
        where="post",
        color="#2ca25f",
        linewidth=1.8,
        alpha=0.95,
        label="Running best",
        zorder=1,
    )

    if args.annotate:
        best_so_far = float("-inf")
        for _, row in df.loc[keep_mask].iterrows():
            improved = row["metric"] >= best_so_far - 1e-12
            if improved:
                best_so_far = max(best_so_far, row["metric"])
                text = shorten(row["description"])
                ax.annotate(
                    text,
                    (row["exp_idx"], row["metric"]),
                    xytext=(6, -10),
                    textcoords="offset points",
                    fontsize=8,
                    color="#2f7f56",
                    rotation=-24,
                )

    kept_count = int(df["is_keep"].sum())
    total_count = len(df)
    title = (
        f"Autoresearch Progress: {total_count} Experiments, "
        f"{kept_count} Kept Improvements"
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Validation Accuracy (higher is better)")
    ax.set_ylim(0.8, 1.0)
    ax.legend(loc="upper right", frameon=True)
    ax.margins(x=0.02)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
