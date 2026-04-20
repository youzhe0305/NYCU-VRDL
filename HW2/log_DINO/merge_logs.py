"""Merge two training log JSON files: original + resumed.

Usage:
    python merge_logs.py <original_log.json> <resumed_log.json> [-o <output.json>]
    python merge_logs.py train_DINO_20260416_151823.json train_DINO_20260419_082942.json \
        -o train_DINO_20260419_082942_merged.json

Output format is identical to a normal training log, with one addition:
    "resume_info" is inserted right after "config" to record which checkpoint
    was used and at which epoch training resumed.

Merge rules:
  - Keep original log's config, start_time, device as the base.
  - Merge epochs: original + resumed, resumed epochs override on conflict.
  - Top-level summary fields (end_time, best_val_map, etc.) taken from resumed
    log (the most recent run), recalculating best_val_map across all epochs.
"""

import argparse
import json
import os


def load(path):
    with open(path) as f:
        return json.load(f)


def merge(orig, resumed):
    ckpt_path = resumed["config"].get("resume") or "<unknown checkpoint>"

    resumed_epochs = resumed.get("epochs", [])
    if not resumed_epochs:
        raise ValueError("Resumed log has no epoch entries.")

    first_resumed_epoch = resumed_epochs[0]["epoch"]

    # Build epoch map: original first, then override with resumed
    epoch_map = {e["epoch"]: e for e in orig.get("epochs", [])}
    for e in resumed_epochs:
        epoch_map[e["epoch"]] = e

    all_epochs = [epoch_map[k] for k in sorted(epoch_map)]

    # Recalculate best_val_map across all merged epochs
    best_val_map = max(
        (e.get("val_map", 0.0) for e in all_epochs), default=0.0
    )
    # Re-tag is_best based on merged best
    for e in all_epochs:
        e["is_best"] = (e.get("val_map", 0.0) == best_val_map)

    merged = {
        "config": orig["config"],
        "resume_info": {
            "checkpoint": ckpt_path,
            "resumed_at_epoch": first_resumed_epoch,
        },
        "start_time": orig.get("start_time"),
        "device": orig.get("device"),
        "epochs": all_epochs,
    }

    # Append summary fields from resumed log (final run state)
    for key in ("end_time", "training_seconds", "total_seconds", "peak_vram_mb",
                "num_params_M", "best_checkpoint"):
        if key in resumed:
            merged[key] = resumed[key]
        elif key in orig:
            merged[key] = orig[key]

    merged["best_val_map"] = round(best_val_map, 6)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge original + resumed training logs")
    parser.add_argument("original", help="Original training log JSON")
    parser.add_argument("resumed", help="Resumed training log JSON")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path (default: merged_<orig_basename>)",
    )
    args = parser.parse_args()

    orig = load(args.original)
    resumed = load(args.resumed)

    if not resumed["config"].get("resume"):
        print("[warn] resumed log config.resume is null — checkpoint recorded as <unknown checkpoint>")

    merged = merge(orig, resumed)

    if args.output:
        out_path = args.output
    else:
        base = os.path.basename(args.original)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(os.path.dirname(args.original), f"merged_{name}{ext}")

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    n_orig = sum(1 for e in merged["epochs"] if e["epoch"] < merged["resume_info"]["resumed_at_epoch"])
    n_res = len(merged["epochs"]) - n_orig
    print(f"Merged {n_orig} original + {n_res} resumed epochs -> {out_path}")
    print(f"  resumed_at_epoch  : {merged['resume_info']['resumed_at_epoch']}")
    print(f"  checkpoint        : {merged['resume_info']['checkpoint']}")
    print(f"  best_val_map      : {merged['best_val_map']}")
    print(f"  total epochs      : {len(merged['epochs'])}")


if __name__ == "__main__":
    main()
