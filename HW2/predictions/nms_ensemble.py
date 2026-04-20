"""NMS-based ensemble for multiple pred.json files.

Merge all predictions, scale scores by model weight, then apply
class-aware NMS (or Soft-NMS) per image.

Usage:
    python nms_ensemble.py pred_1.json pred_2.json pred_3.json -o fused.json
    python nms_ensemble.py pred_a.json pred_b.json --weights 2 1 -o fused.json
    python nms_ensemble.py pred_a.json pred_b.json --iou_thr 0.5 --soft --score_thr 0.01 -o fused.json

Input format  (COCO detection results):
    [{"image_id": int, "bbox": [x, y, w, h], "score": float, "category_id": int}, ...]

Output format: same as input.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou_matrix(boxes):
    """Compute pairwise IoU for boxes [N, 4] in [x1,y1,x2,y2] format."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    union = areas[:, None] + areas[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# NMS / Soft-NMS
# ---------------------------------------------------------------------------

def _nms(boxes, scores, iou_thr):
    """Standard greedy NMS. Returns kept indices."""
    order = np.argsort(-scores)
    kept = []
    while len(order) > 0:
        i = order[0]
        kept.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        iy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        ix2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        iy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_rest - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou < iou_thr]
    return kept


def _soft_nms(boxes, scores, iou_thr, sigma=0.5, score_thr=0.001):
    """Soft-NMS with Gaussian decay. Returns (kept_indices, updated_scores)."""
    scores = scores.copy()
    order = np.argsort(-scores).tolist()
    kept = []
    while order:
        # Pick highest score
        best_pos = int(np.argmax(scores[order]))
        i = order[best_pos]
        order.pop(best_pos)
        kept.append(i)
        # Decay scores of remaining boxes
        for j in order[:]:
            ix1 = max(boxes[i, 0], boxes[j, 0])
            iy1 = max(boxes[i, 1], boxes[j, 1])
            ix2 = min(boxes[i, 2], boxes[j, 2])
            iy2 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0.0
            scores[j] *= np.exp(-(iou ** 2) / sigma)
            if scores[j] < score_thr:
                order.remove(j)
    return kept, scores


# ---------------------------------------------------------------------------
# Per-image NMS ensemble
# ---------------------------------------------------------------------------

def nms_single_image(boxes_list, scores_list, labels_list, weights,
                     iou_thr=0.5, score_thr=0.01, soft=False, sigma=0.5):
    """
    Merge all model predictions, scale by weight, apply class-aware NMS.
    All boxes in [x1,y1,x2,y2] normalised [0,1].
    """
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    all_boxes, all_scores, all_labels = [], [], []
    for bxs, scs, lbs, w in zip(boxes_list, scores_list, labels_list, weights):
        if len(bxs) == 0:
            continue
        all_boxes.append(bxs)
        all_scores.append(scs * w)   # scale score by model weight
        all_labels.append(lbs)

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Class-aware NMS: run separately per category
    final_boxes, final_scores, final_labels = [], [], []
    for cls in np.unique(all_labels):
        mask = all_labels == cls
        bxs = all_boxes[mask]
        scs = all_scores[mask]

        if soft:
            kept, scs = _soft_nms(bxs, scs, iou_thr=iou_thr, sigma=sigma,
                                  score_thr=score_thr)
        else:
            kept = _nms(bxs, scs, iou_thr=iou_thr)
            scs = scs[kept]

        bxs = bxs[kept]
        valid = scs >= score_thr
        final_boxes.append(bxs[valid])
        final_scores.append(scs[valid])
        final_labels.append(np.full(valid.sum(), cls, dtype=int))

    if not final_boxes or all(len(b) == 0 for b in final_boxes):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    return (
        np.concatenate(final_boxes, axis=0),
        np.concatenate(final_scores, axis=0),
        np.concatenate(final_labels, axis=0),
    )


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _xywh_to_xyxy(boxes, img_w, img_h):
    x1 = boxes[:, 0] / img_w
    y1 = boxes[:, 1] / img_h
    x2 = (boxes[:, 0] + boxes[:, 2]) / img_w
    y2 = (boxes[:, 1] + boxes[:, 3]) / img_h
    return np.stack([x1, y1, x2, y2], axis=1).clip(0, 1)


def _xyxy_to_xywh(boxes, img_w, img_h):
    x = boxes[:, 0] * img_w
    y = boxes[:, 1] * img_h
    w = (boxes[:, 2] - boxes[:, 0]) * img_w
    h = (boxes[:, 3] - boxes[:, 1]) * img_h
    return np.stack([x, y, w, h], axis=1)


def _build_image_size_map(pred_files):
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
    print("[WARN] Could not find valid.json/train.json; using dummy image size 10000x10000")
    return defaultdict(lambda: (10000, 10000))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ensemble(pred_files, weights, iou_thr, score_thr, soft, sigma, out_path):
    mode = "Soft-NMS" if soft else "NMS"
    print(f"[{mode}] Ensembling {len(pred_files)} files:")
    for f, w in zip(pred_files, weights):
        print(f"  {f}  (weight={w})")

    all_preds = []
    for path in pred_files:
        with open(path) as f:
            all_preds.append(json.load(f))

    all_image_ids = sorted({p["image_id"] for preds in all_preds for p in preds})
    print(f"Total unique image_ids: {len(all_image_ids)}")

    indexed = []
    for preds in all_preds:
        d = defaultdict(list)
        for p in preds:
            d[p["image_id"]].append(p)
        indexed.append(d)

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

        fused_boxes, fused_scores, fused_labels = nms_single_image(
            boxes_list, scores_list, labels_list,
            weights=weights, iou_thr=iou_thr, score_thr=score_thr,
            soft=soft, sigma=sigma,
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
    print(f"Saved {len(results)} predictions -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="NMS ensemble for COCO-format pred.json files")
    parser.add_argument("pred_files", nargs="+", help="Prediction JSON files to ensemble")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Model weights (default: equal)")
    parser.add_argument("--iou_thr", type=float, default=0.5,
                        help="IoU threshold for NMS (default: 0.5)")
    parser.add_argument("--score_thr", type=float, default=0.01,
                        help="Minimum score to keep after NMS (default: 0.01)")
    parser.add_argument("--soft", action="store_true",
                        help="Use Soft-NMS instead of standard NMS")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Gaussian sigma for Soft-NMS decay (default: 0.5)")
    parser.add_argument("-o", "--output", default="nms_fused.json",
                        help="Output file path (default: nms_fused.json)")
    args = parser.parse_args()

    weights = args.weights if args.weights else [1.0] * len(args.pred_files)
    if len(weights) != len(args.pred_files):
        parser.error("--weights count must match number of pred files")

    ensemble(args.pred_files, weights, args.iou_thr, args.score_thr,
             args.soft, args.sigma, args.output)


if __name__ == "__main__":
    main()
