"""Weighted Boxes Fusion (WBF) ensemble for multiple pred.json files.

Usage:
    python wbf_ensemble.py pred_1.json pred_2.json pred_3.json -o fused.json
    python wbf_ensemble.py pred_a.json pred_b.json --weights 2 1 -o fused.json
    python wbf_ensemble.py pred_a.json pred_b.json --iou_thr 0.55 --score_thr 0.01 -o fused.json

Input format  (COCO detection results):
    [{"image_id": int, "bbox": [x, y, w, h], "score": float, "category_id": int}, ...]

Output format: same as input.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Core WBF implementation (no external dependency)
# ---------------------------------------------------------------------------

def _xywh_to_xyxy(boxes, img_w, img_h):
    """Convert [x,y,w,h] absolute → [x1,y1,x2,y2] normalised to [0,1]."""
    x1 = boxes[:, 0] / img_w
    y1 = boxes[:, 1] / img_h
    x2 = (boxes[:, 0] + boxes[:, 2]) / img_w
    y2 = (boxes[:, 1] + boxes[:, 3]) / img_h
    return np.stack([x1, y1, x2, y2], axis=1).clip(0, 1)


def _xyxy_to_xywh(boxes, img_w, img_h):
    """Convert [x1,y1,x2,y2] normalised → [x,y,w,h] absolute."""
    x = boxes[:, 0] * img_w
    y = boxes[:, 1] * img_h
    w = (boxes[:, 2] - boxes[:, 0]) * img_w
    h = (boxes[:, 3] - boxes[:, 1]) * img_h
    return np.stack([x, y, w, h], axis=1)


def _iou(box, boxes):
    """IoU between one box [x1,y1,x2,y2] and an array of boxes."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def wbf_single_image(
    boxes_list,   # list[np.ndarray] shape (N_i, 4) normalised xyxy, one per model
    scores_list,  # list[np.ndarray] shape (N_i,)
    labels_list,  # list[np.ndarray] shape (N_i,) int
    weights,      # list[float], one per model
    iou_thr=0.55,
    score_thr=0.0,
):
    """
    WBF for a single image across multiple models.
    Returns fused (boxes, scores, labels) all np.ndarray.
    """
    n_models = len(boxes_list)
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    # Collect all boxes with their source model weight
    all_boxes = []   # [x1,y1,x2,y2, score, label, model_weight]
    for i, (bxs, scs, lbs) in enumerate(zip(boxes_list, scores_list, labels_list)):
        for b, s, l in zip(bxs, scs, lbs):
            all_boxes.append((b, s, int(l), weights[i]))

    if not all_boxes:
        empty = np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        return empty

    # Sort by score descending
    all_boxes.sort(key=lambda x: -x[1])

    clusters = []   # each cluster: list of (box, score, label, w)

    for box, score, label, w in all_boxes:
        matched = False
        for cluster in clusters:
            # Compare against the current fused box of the cluster
            fused_box = cluster["fused_box"]
            if _iou(box, fused_box[np.newaxis])[0] >= iou_thr:
                cluster["members"].append((box, score, label, w))
                matched = True
                # Update fused box immediately (weighted average so far)
                _update_cluster(cluster)
                break
        if not matched:
            clusters.append({
                "members": [(box, score, label, w)],
                "fused_box": box.copy(),
                "fused_score": score,
                "fused_label": label,
            })

    # Final pass: compute fused results
    fused_boxes, fused_scores, fused_labels = [], [], []
    for cluster in clusters:
        members = cluster["members"]
        boxes = np.array([m[0] for m in members])
        scores = np.array([m[1] for m in members])
        labels_arr = np.array([m[2] for m in members])
        ws = np.array([m[3] for m in members])

        weighted_scores = scores * ws
        total_w = weighted_scores.sum()

        # Fused box: weighted average by (score * model_weight)
        fused_box = (boxes * weighted_scores[:, None]).sum(axis=0) / total_w

        # Fused score: averaged over n_models (penalises low-confidence ensembles)
        fused_score = total_w / n_models

        # Fused label: majority vote weighted by score*model_weight
        label_weights = defaultdict(float)
        for lbl, sw in zip(labels_arr, weighted_scores):
            label_weights[lbl] += sw
        fused_label = max(label_weights, key=label_weights.get)

        if fused_score >= score_thr:
            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_labels.append(fused_label)

    if not fused_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    return (
        np.array(fused_boxes),
        np.array(fused_scores),
        np.array(fused_labels, dtype=int),
    )


def _update_cluster(cluster):
    members = cluster["members"]
    boxes = np.array([m[0] for m in members])
    scores = np.array([m[1] for m in members])
    ws = np.array([m[3] for m in members])
    weighted_scores = scores * ws
    total_w = weighted_scores.sum()
    cluster["fused_box"] = (boxes * weighted_scores[:, None]).sum(axis=0) / total_w


# ---------------------------------------------------------------------------
# Image size lookup (needed to normalise/denormalise boxes)
# ---------------------------------------------------------------------------

def _build_image_size_map(pred_files):
    """
    We don't have a separate image-info file, so infer image sizes from
    the validation annotation file if available, otherwise use a dummy
    large canvas (1e4 x 1e4) so normalisation is effectively a no-op
    (WBF works in [0,1] space; as long as all models use the same scale
    the fusion is correct).

    Strategy: look for nycu-hw2-data/valid.json or test.json next to this
    script's parent directory.
    """
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for ann_path in [
        os.path.join(base, "nycu-hw2-data", "valid.json"),
        os.path.join(base, "nycu-hw2-data", "train.json"),
    ]:
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                data = json.load(f)
            return {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    # Fallback: no annotation file found; use large dummy size
    print("[WARN] Could not find valid.json/train.json; using dummy image size 10000x10000")
    return defaultdict(lambda: (10000, 10000))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ensemble(pred_files, weights, iou_thr, score_thr, out_path):
    print(f"Ensembling {len(pred_files)} files:")
    for f, w in zip(pred_files, weights):
        print(f"  {f}  (weight={w})")

    # Load all predictions
    all_preds = []
    for path in pred_files:
        with open(path) as f:
            all_preds.append(json.load(f))

    # Gather all image ids
    all_image_ids = set()
    for preds in all_preds:
        for p in preds:
            all_image_ids.add(p["image_id"])
    all_image_ids = sorted(all_image_ids)
    print(f"Total unique image_ids: {len(all_image_ids)}")

    # Index predictions by image_id per model
    indexed = []
    for preds in all_preds:
        d = defaultdict(list)
        for p in preds:
            d[p["image_id"]].append(p)
        indexed.append(d)

    # Image size map for normalisation
    size_map = _build_image_size_map(pred_files)

    results = []
    for img_id in all_image_ids:
        img_w, img_h = size_map.get(img_id, (10000, 10000))

        boxes_list, scores_list, labels_list = [], [], []
        for model_preds in indexed:
            preds_for_img = model_preds.get(img_id, [])
            if preds_for_img:
                bxs = np.array([p["bbox"] for p in preds_for_img], dtype=float)
                scs = np.array([p["score"] for p in preds_for_img], dtype=float)
                lbs = np.array([p["category_id"] for p in preds_for_img], dtype=int)
                bxs_norm = _xywh_to_xyxy(bxs, img_w, img_h)
            else:
                bxs_norm = np.zeros((0, 4))
                scs = np.zeros(0)
                lbs = np.zeros(0, dtype=int)
            boxes_list.append(bxs_norm)
            scores_list.append(scs)
            labels_list.append(lbs)

        fused_boxes, fused_scores, fused_labels = wbf_single_image(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, score_thr=score_thr,
        )

        if len(fused_boxes) == 0:
            continue

        fused_xywh = _xyxy_to_xywh(fused_boxes, img_w, img_h)
        for box, score, label in zip(fused_xywh, fused_scores, fused_labels):
            results.append({
                "image_id": img_id,
                "bbox": box.tolist(),
                "score": float(score),
                "category_id": int(label),
            })

    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} fused predictions -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="WBF ensemble for COCO-format pred.json files")
    parser.add_argument("pred_files", nargs="+", help="Prediction JSON files to ensemble")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Model weights (default: equal). Must match number of pred files.")
    parser.add_argument("--iou_thr", type=float, default=0.7,
                        help="IoU threshold for WBF clustering (default: 0.55)")
    parser.add_argument("--score_thr", type=float, default=0.01,
                        help="Minimum fused score to keep (default: 0.01)")
    parser.add_argument("-o", "--output", default="fused.json",
                        help="Output file path (default: fused.json)")
    args = parser.parse_args()

    weights = args.weights if args.weights else [1.0] * len(args.pred_files)
    if len(weights) != len(args.pred_files):
        parser.error("--weights count must match number of pred files")

    ensemble(args.pred_files, weights, args.iou_thr, args.score_thr, args.output)


if __name__ == "__main__":
    main()
