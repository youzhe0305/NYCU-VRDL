"""Inference script for DINO digit detection (HW2).

Generates pred.json in COCO detection format for submission.
Based on inference.py with modifications for DINO's sigmoid-based scoring.

Optional Test-Time Augmentation (TTA):
  --tta runs inference 3 times and merges the per-image predictions with
  Weighted Boxes Fusion (WBF, Solovyev 2021).
    --tta_mode color : original + 2 ColorJitter variants (default)
    --tta_mode scale : identity transform at 2/3, 1, 4/3 of max/min side
"""

import argparse
import json
import os
import zipfile

import numpy as np
import torch
import yaml
from torchvision import transforms as tv_transforms
from torch.utils.data import DataLoader

from dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TestDataset,
    collate_fn,
    get_val_transforms,
    build_dataloaders,
)
from model_DINO import build_model
from train_DINO import box_cxcywh_to_xyxy, evaluate_map, compute_ap, _box_iou_single


# ---------------------------------------------------------------------------
# TTA transforms
# ---------------------------------------------------------------------------

def get_tta_transforms():
    """Three transform pipelines for TTA: identity + two ColorJitter variants.

    ColorJitter is given tuple ranges of equal lo/hi so each variant is
    deterministic (picklable across DataLoader workers).
    """
    normalize = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    identity = normalize

    cj1 = tv_transforms.Compose([
        tv_transforms.ColorJitter(
            brightness=(0.8, 0.8),
            contrast=(1.2, 1.2),
            saturation=(1.1, 1.1),
            hue=(-0.05, -0.05),
        ),
        normalize,
    ])

    cj2 = tv_transforms.Compose([
        tv_transforms.ColorJitter(
            brightness=(1.2, 1.2),
            contrast=(0.85, 0.85),
            saturation=(0.9, 0.9),
            hue=(0.05, 0.05),
        ),
        normalize,
    ])

    return [("orig", identity), ("cj1", cj1), ("cj2", cj2)]


# ---------------------------------------------------------------------------
# Weighted Boxes Fusion (Solovyev et al., 2021)
# ---------------------------------------------------------------------------

def _box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    weights=None, iou_thr=0.55, skip_box_thr=0.0,
):
    """Fuse predictions from multiple models/TTA variants.

    Inputs
      boxes_list:  list (per model) of (N, 4) np.ndarray, xyxy normalised to [0,1]
      scores_list: list (per model) of (N,)  np.ndarray
      labels_list: list (per model) of (N,)  np.ndarray of int class ids

    Returns tuple of (fused_boxes, fused_scores, fused_labels).
    """
    num_models = len(boxes_list)
    if weights is None:
        weights = [1.0] * num_models
    weights = np.asarray(weights, dtype=np.float32)
    weight_sum = float(weights.sum()) if weights.sum() > 0 else 1.0

    entries = []
    for m in range(num_models):
        bxs = np.asarray(boxes_list[m], dtype=np.float32).reshape(-1, 4)
        scs = np.asarray(scores_list[m], dtype=np.float32).reshape(-1)
        lbs = np.asarray(labels_list[m]).reshape(-1)
        for i in range(len(bxs)):
            if scs[i] < skip_box_thr:
                continue
            entries.append((float(scs[i]), int(lbs[i]), m, bxs[i]))

    entries.sort(key=lambda e: -e[0])

    clusters = []  # each: {'label', 'items': [(score, m, box)], 'fused_box'}
    for score, label, m, box in entries:
        best_iou, best_idx = iou_thr, -1
        for idx, c in enumerate(clusters):
            if c['label'] != label:
                continue
            iou = _box_iou(c['fused_box'], box)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx >= 0:
            c = clusters[best_idx]
            c['items'].append((score, m, box))
            ws = np.array(
                [s * weights[mm] for s, mm, _ in c['items']], dtype=np.float32,
            )
            bxs = np.stack([bb for _, _, bb in c['items']])
            c['fused_box'] = (bxs * ws[:, None]).sum(axis=0) / max(ws.sum(), 1e-8)
        else:
            clusters.append(
                {'label': label, 'items': [(score, m, box)], 'fused_box': box.copy()}
            )

    if not clusters:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )

    out_b, out_s, out_l = [], [], []
    for c in clusters:
        accum = sum(s * weights[mm] for s, mm, _ in c['items'])
        # sum(score * weight) / total_weight — absent models contribute 0
        final_score = accum / weight_sum
        # Clip to [0, 1] just in case of numerical drift
        fb = np.clip(c['fused_box'], 0.0, 1.0)
        out_b.append(fb)
        out_s.append(final_score)
        out_l.append(c['label'])

    return (
        np.stack(out_b).astype(np.float32),
        np.array(out_s, dtype=np.float32),
        np.array(out_l, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _collect_raw_predictions(model, test_loader, device, keep_top_k=None):
    """Run the model and return per-image raw predictions in normalised xyxy.

    Returns dict: image_id -> {'boxes', 'scores', 'labels', 'orig_h', 'orig_w'}.
    No score thresholding here — all queries are returned so WBF has full
    access to low-confidence candidates.  `keep_top_k` optionally keeps only
    the top-K queries per image (by score) to bound memory.
    """
    model.eval()
    preds = {}
    with torch.no_grad():
        for images, masks, targets_list in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images, masks)
            pred_logits = outputs["pred_logits"]
            pred_boxes = outputs["pred_boxes"]

            for b, tgt in enumerate(targets_list):
                image_id = int(tgt["image_id"])
                orig_h, orig_w = tgt["orig_size"].tolist()

                scores_per_class = pred_logits[b].sigmoid()  # [N, C]
                scores, classes = scores_per_class.max(dim=-1)
                boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])  # [N,4] in [0,1]

                if keep_top_k is not None and keep_top_k < len(scores):
                    topk = torch.topk(scores, keep_top_k)
                    idx = topk.indices
                    scores = scores[idx]
                    classes = classes[idx]
                    boxes_xyxy = boxes_xyxy[idx]

                preds[image_id] = {
                    "boxes": boxes_xyxy.cpu().numpy().astype(np.float32),
                    "scores": scores.cpu().numpy().astype(np.float32),
                    "labels": classes.cpu().numpy().astype(np.int64),
                    "orig_h": float(orig_h),
                    "orig_w": float(orig_w),
                }
    return preds


def _format_coco(image_id, boxes_xyxy, scores, labels, orig_h, orig_w,
                 score_threshold):
    """Convert normalised xyxy predictions to COCO submission dicts."""
    out = []
    for box, s, l in zip(boxes_xyxy, scores, labels):
        if s < score_threshold:
            continue
        x1, y1, x2, y2 = box
        x_min = float(x1 * orig_w)
        y_min = float(y1 * orig_h)
        bw = float((x2 - x1) * orig_w)
        bh = float((y2 - y1) * orig_h)
        out.append({
            "image_id": image_id,
            "bbox": [x_min, y_min, bw, bh],
            "score": float(s),
            "category_id": int(l) + 1,  # 1-indexed
        })
    return out


def run_inference(model, test_loader, device, score_threshold=0.5):
    """Standard single-pass inference producing COCO-format predictions."""
    raw = _collect_raw_predictions(model, test_loader, device)
    results = []
    for image_id, r in raw.items():
        results.extend(_format_coco(
            image_id, r["boxes"], r["scores"], r["labels"],
            r["orig_h"], r["orig_w"], score_threshold,
        ))
    return results


def _build_tta_variants(args):
    """Produce (name, transform, max_side, min_side) tuples for the chosen mode.

    mode="color": identity + 2 ColorJitter variants at the original resolution.
    mode="scale": identity transform at 3 resolutions (2/3, 1, 4/3 of max/min).
    """
    mode = args.tta_mode

    if mode == "color":
        return [
            (name, tfm, args.max_side, args.min_side)
            for name, tfm in get_tta_transforms()
        ]

    if mode == "scale":
        normalize = tv_transforms.Compose([
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        ratios = [("s67", 2.0 / 3.0), ("orig", 1.0), ("s133", 4.0 / 3.0)]
        return [
            (
                name,
                normalize,
                max(1, int(round(args.max_side * r))),
                max(1, int(round(args.min_side * r))),
            )
            for name, r in ratios
        ]

    raise ValueError(f"Unknown tta_mode: {mode!r} (expected 'color' or 'scale')")


def run_inference_tta(
    model, args, device,
    iou_thr=0.55, wbf_skip_thr=0.001, keep_top_k=None,
):
    """TTA inference: 3 variants merged with WBF."""
    variants = _build_tta_variants(args)

    per_variant_preds = []  # list of {image_id -> pred dict}
    for name, tfm, max_side, min_side in variants:
        ds = TestDataset(
            img_dir=os.path.join(args.data_dir, "test"),
            transforms=tfm,
            max_side=max_side,
            min_side=min_side,
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        )
        print(f"[TTA:{args.tta_mode}] Running variant: {name} "
              f"(max_side={max_side}, min_side={min_side}, {len(ds)} images)")
        per_variant_preds.append(
            _collect_raw_predictions(model, loader, device, keep_top_k=keep_top_k)
        )

    # Fuse per image
    all_ids = set()
    for p in per_variant_preds:
        all_ids.update(p.keys())

    results = []
    for image_id in sorted(all_ids):
        boxes_list, scores_list, labels_list = [], [], []
        orig_h = orig_w = None
        for p in per_variant_preds:
            if image_id in p:
                r = p[image_id]
                boxes_list.append(r["boxes"])
                scores_list.append(r["scores"])
                labels_list.append(r["labels"])
                orig_h, orig_w = r["orig_h"], r["orig_w"]
            else:
                boxes_list.append(np.zeros((0, 4), dtype=np.float32))
                scores_list.append(np.zeros(0, dtype=np.float32))
                labels_list.append(np.zeros(0, dtype=np.int64))

        fused_b, fused_s, fused_l = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=iou_thr, skip_box_thr=wbf_skip_thr,
        )
        results.extend(_format_coco(
            image_id, fused_b, fused_s, fused_l,
            orig_h, orig_w, args.score_threshold,
        ))
    return results


# ---------------------------------------------------------------------------
# TTA val mAP
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_map_tta(model, args, device, iou_thr=0.55, wbf_skip_thr=0.001):
    """Compute val mAP using the same TTA+WBF pipeline as test inference.

    For scale TTA: builds separate val loaders per scale variant.
    For color TTA: val transforms don't include ColorJitter, so all variants
    are identical — result equals single-pass val mAP.
    """
    variants = _build_tta_variants(args)
    per_variant_preds = []
    all_gts = {}

    for i, (name, _tfm, max_side, min_side) in enumerate(variants):
        _, val_loader, _ = build_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_side=max_side,
            min_side=min_side,
        )
        print(f"[TTA val:{args.tta_mode}] variant={name} "
              f"(max_side={max_side}, min_side={min_side}, {len(val_loader.dataset)} images)")

        model.eval()
        preds = {}
        for images, masks, targets_list in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images, masks)

            for b, tgt in enumerate(targets_list):
                img_id = int(tgt["image_id"])
                scores_pc = outputs["pred_logits"][b].sigmoid()
                scores, classes = scores_pc.max(dim=-1)
                boxes_xyxy = box_cxcywh_to_xyxy(outputs["pred_boxes"][b])
                preds[img_id] = {
                    "boxes": boxes_xyxy.cpu().numpy().astype(np.float32),
                    "scores": scores.cpu().numpy().astype(np.float32),
                    "labels": classes.cpu().numpy().astype(np.int64),
                }
                if i == 0:
                    gt_xyxy = box_cxcywh_to_xyxy(tgt["boxes"])
                    all_gts[img_id] = [
                        (int(lbl.item()), gt_xyxy[j].tolist())
                        for j, lbl in enumerate(tgt["labels"])
                    ]
        per_variant_preds.append(preds)

    # WBF fusion per image
    all_preds = []
    for img_id in sorted(all_gts.keys()):
        boxes_list, scores_list, labels_list = [], [], []
        for p in per_variant_preds:
            r = p.get(img_id)
            if r:
                boxes_list.append(r["boxes"])
                scores_list.append(r["scores"])
                labels_list.append(r["labels"])
            else:
                boxes_list.append(np.zeros((0, 4), np.float32))
                scores_list.append(np.zeros(0, np.float32))
                labels_list.append(np.zeros(0, np.int64))

        fused_b, fused_s, fused_l = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=iou_thr, skip_box_thr=wbf_skip_thr,
        )
        for box, s, l in zip(fused_b, fused_s, fused_l):
            if s >= args.score_threshold:
                all_preds.append((img_id, int(l) + 1, float(s), box.tolist()))

    # mAP computation (mirrors evaluate_map from train_DINO)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    all_aps = []
    for thr in iou_thresholds:
        for cls in range(1, args.num_classes + 1):
            cls_preds = sorted(
                [(s, iid, b4) for iid, c, s, b4 in all_preds if c == cls],
                key=lambda x: -x[0],
            )
            num_gt = sum(sum(1 for lbl, _ in gts if lbl == cls) for gts in all_gts.values())
            if num_gt == 0:
                continue
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            matched = {iid: set() for iid in all_gts}
            for k, (_, iid, pred_box) in enumerate(cls_preds):
                gts_for_img = [
                    (j, b4) for j, (lbl, b4) in enumerate(all_gts.get(iid, [])) if lbl == cls
                ]
                best_iou, best_j = 0.0, -1
                for j, gt_box in gts_for_img:
                    iou = _box_iou_single(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= thr and best_j not in matched[iid]:
                    tp[k] = 1
                    matched[iid].add(best_j)
                else:
                    fp[k] = 1
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / num_gt
            prec = tp_cum / (tp_cum + fp_cum + 1e-9)
            all_aps.append(compute_ap(rec, prec))

    return float(np.mean(all_aps)) if all_aps else 0.0


# ---------------------------------------------------------------------------
# Confusion matrix on validation set
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_confusion_matrix(
    model, val_loader, device, num_classes,
    score_threshold=0.3, iou_threshold=0.5,
):
    """Compute (C+1) x (C+1) confusion matrix on the validation set.

    Layout (rows = prediction, cols = ground truth):
        cm[c1, c2]   : prediction matched a GT (IoU >= iou_threshold) and
                       predicted class c1 while GT class is c2.
        cm[c, bg]    : prediction with class c that matched no GT (false positive).
        cm[bg, c]    : GT of class c that no prediction matched (missed / FN).

    Matching is greedy: predictions are sorted by score (high to low) and each
    grabs the still-unmatched GT with the highest IoU above the threshold,
    regardless of class — so off-diagonal entries reflect *class* confusions
    on otherwise well-localised boxes.
    """
    model.eval()
    bg = num_classes
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for images, masks, targets_list in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images, masks)
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        for b, tgt in enumerate(targets_list):
            scores_pc = pred_logits[b].sigmoid()
            scores, classes = scores_pc.max(dim=-1)
            boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])

            keep = scores >= score_threshold
            scores = scores[keep].cpu().numpy()
            classes = classes[keep].cpu().numpy().astype(np.int64)
            boxes = boxes_xyxy[keep].cpu().numpy()

            # GT labels are 1-indexed (1..num_classes); shift to 0-indexed
            # so they line up with the model's 0..num_classes-1 outputs.
            gt_labels = tgt["labels"].cpu().numpy().astype(np.int64) - 1
            gt_boxes = box_cxcywh_to_xyxy(tgt["boxes"]).cpu().numpy()

            order = np.argsort(-scores)
            classes = classes[order]
            boxes = boxes[order]

            gt_matched = np.zeros(len(gt_boxes), dtype=bool)
            for p_box, p_cls in zip(boxes, classes):
                best_iou, best_j = iou_threshold, -1
                for j, g_box in enumerate(gt_boxes):
                    if gt_matched[j]:
                        continue
                    iou = _box_iou(p_box.tolist(), g_box.tolist())
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j >= 0:
                    gt_matched[best_j] = True
                    cm[int(p_cls), int(gt_labels[best_j])] += 1
                else:
                    cm[int(p_cls), bg] += 1

            for j, matched in enumerate(gt_matched):
                if not matched:
                    cm[bg, int(gt_labels[j])] += 1

    return cm


def save_per_class_summary(cm, output_path, num_classes):
    """Derive per-class TP / FP / FN / Precision / Recall / F1 from the CM.

    Definitions (bg = background / no-match slot, index num_classes):
        TP_c  = cm[c, c]
        FP_c  = cm[c, bg] + sum over c' != c of cm[c, c']   (pred=c but wrong)
        FN_c  = cm[bg, c] + sum over c' != c of cm[c', c]   (GT=c but missed or mis-classified)
    """
    bg = num_classes
    lines = ["class,TP,FP,FN,Precision,Recall,F1"]

    total_tp = total_fp = total_fn = 0
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[c, bg]) + int(cm[c, :bg].sum() - cm[c, c])
        fn = int(cm[bg, c]) + int(cm[:bg, c].sum() - cm[c, c])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        lines.append(f"{c},{tp},{fp},{fn},{prec:.4f},{rec:.4f},{f1:.4f}")
        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)
                  if (overall_prec + overall_rec) > 0 else 0.0)
    lines.append(
        f"overall(micro),{total_tp},{total_fp},{total_fn},"
        f"{overall_prec:.4f},{overall_rec:.4f},{overall_f1:.4f}"
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved per-class summary CSV: {output_path}")

    # Also print to stdout for quick inspection
    print()
    print(f"{'class':>12} {'TP':>7} {'FP':>7} {'FN':>7} "
          f"{'Prec':>7} {'Rec':>7} {'F1':>7}")
    for line in lines[1:]:
        cols = line.split(",")
        print(f"{cols[0]:>12} {cols[1]:>7} {cols[2]:>7} {cols[3]:>7} "
              f"{cols[4]:>7} {cols[5]:>7} {cols[6]:>7}")
    print()


def save_confusion_matrix(cm, output_path, num_classes):
    """Save the confusion matrix as both a CSV and a PNG heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(i) for i in range(num_classes)] + ["bg"]

    # CSV (raw counts)
    csv_path = os.path.splitext(output_path)[0] + ".csv"
    with open(csv_path, "w") as f:
        f.write("pred\\gt," + ",".join(labels) + "\n")
        for i, row_label in enumerate(labels):
            f.write(row_label + "," + ",".join(str(int(v)) for v in cm[i]) + "\n")
    print(f"Saved confusion-matrix CSV: {csv_path}")

    # 11x11 heatmap (column-normalised by GT count)
    col_sums = cm.sum(axis=0, keepdims=True).astype(np.float64)
    cm_norm = np.divide(
        cm, col_sums,
        where=col_sums > 0,
        out=np.zeros(cm.shape, dtype=np.float64),
    )

    # Mark the bg/bg cell explicitly — TN is undefined for detection.
    cm_norm_display = cm_norm.copy()
    cm_norm_display[num_classes, num_classes] = np.nan

    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(cm_norm_display, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title(
        "Validation confusion matrix\n"
        "(cell % = fraction of each GT column; diagonal = per-class recall)"
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == num_classes and j == num_classes:
                ax.text(
                    j, i, "N/A\n(TN)",
                    ha="center", va="center",
                    color="black", fontsize=7,
                )
                continue
            count = int(cm[i, j])
            if count == 0:
                continue
            prop = cm_norm[i, j]
            color = "white" if prop > 0.5 else "black"
            ax.text(
                j, i, f"{count}\n{prop * 100:.1f}%",
                ha="center", va="center",
                color=color, fontsize=7,
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion-matrix PNG: {output_path}")

    # Per-class TP/FP/FN summary
    summary_path = os.path.splitext(output_path)[0] + "_summary.csv"
    save_per_class_summary(cm, summary_path, num_classes)

    # 2x2 binary confusion matrix (aggregated over all classes)
    binary_path = os.path.splitext(output_path)[0] + "_2x2.png"
    save_binary_confusion_matrix(cm, binary_path, num_classes)


def save_binary_confusion_matrix(cm, output_path, num_classes):
    """Render a 2x2 TP/FP/FN heatmap aggregated over all classes.

    Note: for detection tasks TN (correctly predicting no object at a location)
    is not well-defined — there is no fixed set of negatives to count. We mark
    that cell as N/A.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg = num_classes
    # A prediction is "correct" only if it landed on the diagonal (same class,
    # matched GT). Everything else in the pred-rows (incl. class mix-ups and
    # pred→bg) is a false positive.
    tp = int(np.trace(cm[:bg, :bg]))
    fp = int(cm[:bg, :].sum()) - tp
    fn = int(cm[bg, :bg].sum()) + int(cm[:bg, :bg].sum() - tp)
    # cm[bg, :bg] = pure misses; cm[:bg, :bg] off-diagonal counts as both FP
    # (wrong class predicted) and FN (true class missed), which is the
    # standard micro-averaged treatment.

    # 2x2 grid: rows = prediction (Positive, Negative); cols = GT (Positive, Negative)
    grid = np.array(
        [
            [tp, fp],
            [fn, np.nan],   # TN is undefined for detection
        ],
        dtype=np.float64,
    )

    # Normalise the three defined cells by their joint total so the heatmap is
    # readable even when TP dominates.
    total = tp + fp + fn
    grid_norm = np.where(np.isnan(grid), np.nan, grid / max(total, 1))

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#dddddd")     # grey for N/A
    im = ax.imshow(grid_norm, cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive (object)", "Negative (bg)"])
    ax.set_yticklabels(["Positive (detected)", "Negative (missed)"])
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.set_title("2x2 binary confusion matrix (aggregated)")

    cell_labels = [
        [f"TP\n{tp}", f"FP\n{fp}"],
        [f"FN\n{fn}", "TN\nN/A"],
    ]
    for i in range(2):
        for j in range(2):
            val = grid_norm[i, j]
            if np.isnan(val):
                color = "black"
            else:
                color = "white" if val > 0.5 else "black"
            ax.text(
                j, i, cell_labels[i][j],
                ha="center", va="center",
                color=color, fontsize=13, fontweight="bold",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved 2x2 binary confusion matrix PNG: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        d_model=args.d_model,
        n_heads=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ffn=args.dim_feedforward,
        dropout=0.0,
        n_levels=args.n_levels,
        n_points=args.n_points,
        pretrained_backbone=False,
        dn_number=0,  # no DN at inference
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    if args.run_val:
        _, val_loader, _ = build_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_side=args.max_side,
            min_side=args.min_side,
        )
        val_map = evaluate_map(model, val_loader, device, score_thr=args.score_threshold)
        print(f"Val mAP (single-pass, score_thr={args.score_threshold}): {val_map:.4f}")

    if args.confusion_matrix:
        _, val_loader, _ = build_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_side=args.max_side,
            min_side=args.min_side,
        )
        print(f"[CM] Computing confusion matrix on validation set "
              f"(score_thr={args.cm_score_thr}, iou_thr={args.cm_iou_thr})")
        cm = compute_confusion_matrix(
            model, val_loader, device, args.num_classes,
            score_threshold=args.cm_score_thr,
            iou_threshold=args.cm_iou_thr,
        )
        os.makedirs(os.path.dirname(args.cm_output) or ".", exist_ok=True)
        save_confusion_matrix(cm, args.cm_output, args.num_classes)

    if args.tta:
        print(f"[TTA] enabled — mode={args.tta_mode}, "
              f"iou_thr={args.tta_iou_thr}, skip_thr={args.tta_skip_thr}")
        results = run_inference_tta(
            model, args, device,
            iou_thr=args.tta_iou_thr,
            wbf_skip_thr=args.tta_skip_thr,
            keep_top_k=args.tta_keep_top_k,
        )
        if args.run_val:
            tta_val_map = evaluate_map_tta(
                model, args, device,
                iou_thr=args.tta_iou_thr,
                wbf_skip_thr=args.tta_skip_thr,
            )
            print(f"Val mAP (TTA {args.tta_mode}): {tta_val_map:.4f}")
    else:
        test_dataset = TestDataset(
            img_dir=os.path.join(args.data_dir, "test"),
            transforms=get_val_transforms(),
            max_side=args.max_side,
            min_side=args.min_side,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        print(f"Test samples: {len(test_dataset)}")
        results = run_inference(model, test_loader, device, args.score_threshold)

    print(f"Total predictions: {len(results)}")

    os.makedirs(args.output_dir, exist_ok=True)
    pred_json_path = os.path.join(args.output_dir, "pred.json")
    with open(pred_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved: {pred_json_path}")

    zip_path = os.path.join(args.output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_json_path, "pred.json")
    print(f"Saved: {zip_path}")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINO inference for digit detection")
    parser.add_argument("--config", default="configs/default_DINO.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default="predictions")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_side", type=int, default=None)
    parser.add_argument("--min_side", type=int, default=None)
    parser.add_argument("--score_threshold", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num_encoder_layers", type=int, default=None)
    parser.add_argument("--num_decoder_layers", type=int, default=None)
    parser.add_argument("--dim_feedforward", type=int, default=None)
    parser.add_argument("--n_levels", type=int, default=None)
    parser.add_argument("--n_points", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run_val", action="store_true")

    # TTA
    parser.add_argument("--tta", action="store_true",
                        help="Enable TTA (3 variants fused via WBF)")
    parser.add_argument("--tta_mode", choices=["color", "scale"], default="color",
                        help="'color': orig + 2 ColorJitter variants at the original "
                             "resolution; 'scale': identity transform at 2/3, 1, 4/3 "
                             "of max_side/min_side")
    parser.add_argument("--tta_iou_thr", type=float, default=0.65,
                        help="WBF IoU threshold for cluster matching")
    parser.add_argument("--tta_skip_thr", type=float, default=0.001,
                        help="Drop candidates below this score before WBF")
    parser.add_argument("--tta_keep_top_k", type=int, default=None,
                        help="Keep only top-K queries per image per variant before WBF "
                             "(default: keep all)")

    # Confusion matrix
    parser.add_argument("--confusion_matrix", action="store_true",
                        help="Compute & save (C+1)x(C+1) confusion matrix on the val set")
    parser.add_argument("--cm_iou_thr", type=float, default=0.5,
                        help="IoU threshold for matching predictions to GT in CM")
    parser.add_argument("--cm_score_thr", type=float, default=0.01,
                        help="Drop predictions below this score before CM matching")
    parser.add_argument("--cm_output", type=str,
                        default="predictions/confusion_matrix.png",
                        help="Output PNG path; CSV uses the same stem")

    args = parser.parse_args()
    cfg = load_config(args.config)

    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is None and key in cfg:
            setattr(args, key, cfg[key])

    defaults = dict(
        data_dir="nycu-hw2-data",
        batch_size=4,
        num_workers=4,
        max_side=640,
        min_side=480,
        score_threshold=0.5,
        num_classes=10,
        num_queries=300,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        n_levels=4,
        n_points=4,
        dropout=0.0,
        device="cuda",
    )
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    main(args)
