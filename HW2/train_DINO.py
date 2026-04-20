"""Training script for DINO digit detection (HW2).

Based on train.py with modifications for DINO:
  - Focal loss (sigmoid-based) instead of CE for classification
  - DN loss computation for contrastive denoising
  - Encoder auxiliary loss
  - Targets passed to model forward for DN
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

from dataset import build_dataloaders
from model_DINO import build_model


# ---------------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------------

def box_cxcywh_to_xyxy(boxes):
    """[cx, cy, w, h] -> [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
    )


def giou_pairwise(b1, b2):
    """Element-wise GIoU for two [N, 4] sets in xyxy format."""
    ix1 = torch.max(b1[:, 0], b2[:, 0])
    iy1 = torch.max(b1[:, 1], b2[:, 1])
    ix2 = torch.min(b1[:, 2], b2[:, 2])
    iy2 = torch.min(b1[:, 3], b2[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    a1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    union = a1 + a2 - inter
    iou = inter / (union + 1e-6)

    ex1 = torch.min(b1[:, 0], b2[:, 0])
    ey1 = torch.min(b1[:, 1], b2[:, 1])
    ex2 = torch.max(b1[:, 2], b2[:, 2])
    ey2 = torch.max(b1[:, 3], b2[:, 3])
    enc = (ex2 - ex1).clamp(0) * (ey2 - ey1).clamp(0)

    return iou - (enc - union) / (enc + 1e-6)


def giou_cost_matrix(pred_boxes, gt_boxes):
    """Compute an [N, M] GIoU cost matrix (pred vs. GT) in cxcywh format."""
    N = pred_boxes.shape[0]
    M = gt_boxes.shape[0]
    pred_xy = box_cxcywh_to_xyxy(pred_boxes)
    gt_xy = box_cxcywh_to_xyxy(gt_boxes)

    p = pred_xy.unsqueeze(1).expand(N, M, 4).reshape(N * M, 4)
    g = gt_xy.unsqueeze(0).expand(N, M, 4).reshape(N * M, 4)
    giou = giou_pairwise(p, g).view(N, M)
    return -giou


# ---------------------------------------------------------------------------
# Focal loss (sigmoid-based, DINO-style)
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="none"):
    """Sigmoid focal loss.

    Args:
        inputs: [N, C] raw logits
        targets: [N, C] one-hot or soft targets
    """
    p = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


# ---------------------------------------------------------------------------
# Hungarian matching (focal-loss based cost)
# ---------------------------------------------------------------------------

@torch.no_grad()
def hungarian_match(logits, boxes, target, num_classes):
    """
    Match N predictions to M ground-truth objects (DINO-style with focal cost).

    Args:
        logits: [N, num_classes] raw class logits (no softmax, sigmoid-based)
        boxes:  [N, 4] predicted boxes (cxcywh, normalised)
        target: dict with 'labels' [M] (1-indexed) and 'boxes' [M, 4]
        num_classes: number of foreground classes

    Returns:
        (pred_idx, gt_idx) LongTensors
    """
    M = target["labels"].shape[0]
    if M == 0:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty

    # Focal-loss based classification cost
    out_prob = logits.sigmoid()  # [N, C]
    gt_cls = target["labels"] - 1  # 0-indexed [M]

    alpha = 0.25
    gamma = 2.0
    neg_cost = (1 - alpha) * (out_prob ** gamma) * (
        -(1 - out_prob + 1e-8).log()
    )
    pos_cost = alpha * ((1 - out_prob) ** gamma) * (
        -(out_prob + 1e-8).log()
    )
    cls_cost = pos_cost[:, gt_cls] - neg_cost[:, gt_cls]  # [N, M]

    # L1 bbox cost
    gt_boxes = target["boxes"].to(boxes.device)
    l1_cost = torch.cdist(boxes, gt_boxes, p=1)

    # GIoU cost
    g_cost = giou_cost_matrix(boxes, gt_boxes)

    cost = 2.0 * cls_cost + 5.0 * l1_cost + 2.0 * g_cost
    if torch.isnan(cost).any():
        raise ValueError("Cost matrix contains NaN")
    pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())
    return (
        torch.tensor(pred_idx, dtype=torch.long),
        torch.tensor(gt_idx, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# DINO Loss
# ---------------------------------------------------------------------------

class DINOLoss(torch.nn.Module):
    """DINO loss: focal loss + L1 + GIoU, with DN and encoder auxiliary losses."""

    def __init__(
        self,
        num_classes,
        lambda_cls=1.0,
        lambda_bbox=5.0,
        lambda_giou=2.0,
        focal_alpha=0.25,
        focal_gamma=2.0,
        use_aux_loss=True,
        use_enc_loss=True,
        use_dn_loss=True,
        lambda_aux=1.0,
        lambda_enc=1.0,
        lambda_dn=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cls = lambda_cls
        self.lambda_bbox = lambda_bbox
        self.lambda_giou = lambda_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_aux_loss = use_aux_loss
        self.use_enc_loss = use_enc_loss
        self.use_dn_loss = use_dn_loss
        self.lambda_aux = lambda_aux
        self.lambda_enc = lambda_enc
        self.lambda_dn = lambda_dn

    def _compute_single_loss(self, pred_logits, pred_boxes, targets_list):
        """Compute loss for one set of predictions (focal cls + L1 + GIoU)."""
        B, N, C = pred_logits.shape

        cls_losses, bbox_losses, giou_losses = [], [], []
        total_matched = 0

        for b in range(B):
            logits_b = pred_logits[b]  # [N, C]
            boxes_b = pred_boxes[b]    # [N, 4]
            tgt = targets_list[b]
            M = tgt["labels"].shape[0]

            pred_idx, gt_idx = hungarian_match(
                logits_b, boxes_b, tgt, self.num_classes
            )

            # Build one-hot targets for focal loss
            tgt_cls = torch.zeros(
                N, self.num_classes, dtype=torch.float32, device=logits_b.device
            )
            if M > 0 and len(pred_idx) > 0:
                gt_labels_0 = tgt["labels"][gt_idx].to(logits_b.device) - 1
                tgt_cls[pred_idx, gt_labels_0] = 1.0

            cls_loss = sigmoid_focal_loss(
                logits_b, tgt_cls,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
            ).sum() / max(M, 1)
            cls_losses.append(cls_loss)

            if M > 0 and len(pred_idx) > 0:
                matched_pred = boxes_b[pred_idx]
                matched_gt = tgt["boxes"][gt_idx].to(boxes_b.device)
                bbox_losses.append(
                    F.l1_loss(matched_pred, matched_gt, reduction="sum")
                )
                giou = giou_pairwise(
                    box_cxcywh_to_xyxy(matched_pred),
                    box_cxcywh_to_xyxy(matched_gt),
                )
                giou_losses.append((1 - giou).sum())
                total_matched += len(pred_idx)

        num_boxes = max(total_matched, 1)
        cls_loss_avg = torch.stack(cls_losses).mean()
        bbox_loss_avg = (
            torch.stack(bbox_losses).sum() / num_boxes
            if bbox_losses
            else pred_logits.new_tensor(0.0)
        )
        giou_loss_avg = (
            torch.stack(giou_losses).sum() / num_boxes
            if giou_losses
            else pred_logits.new_tensor(0.0)
        )

        total = (
            self.lambda_cls * cls_loss_avg
            + self.lambda_bbox * bbox_loss_avg
            + self.lambda_giou * giou_loss_avg
        )
        return total, cls_loss_avg, bbox_loss_avg, giou_loss_avg

    def _compute_dn_loss(self, dn_class, dn_coord, targets_list, dn_meta):
        """Compute contrastive denoising loss (vectorized).

        Positive DN queries should reconstruct GT.
        Negative DN queries should predict background (no object).
        """
        if dn_meta is None or dn_class is None:
            return dn_class.new_tensor(0.0) if dn_class is not None else torch.tensor(0.0)

        num_layers = dn_class.shape[0]
        single_pad = dn_meta["single_pad"]
        num_groups = dn_meta["num_dn_group"]
        max_gt = dn_meta["max_gt"]
        B = dn_class.shape[1]
        device = dn_class.device

        # Pre-compute positive and negative indices for all groups
        # Layout per group: [pos_0, neg_0, pos_1, neg_1, ..., pos_{max_gt-1}, neg_{max_gt-1}]
        pos_indices_per_group = torch.arange(0, single_pad, 2, device=device)  # [max_gt]
        neg_indices_per_group = torch.arange(1, single_pad, 2, device=device)  # [max_gt]
        group_offsets = torch.arange(num_groups, device=device) * single_pad    # [num_groups]

        # All positive indices: [num_groups * max_gt]
        all_pos_idx = (group_offsets[:, None] + pos_indices_per_group[None, :]).reshape(-1)
        all_neg_idx = (group_offsets[:, None] + neg_indices_per_group[None, :]).reshape(-1)

        # Build GT targets for each batch: repeated for all groups
        # gt_labels_repeated[b]: [num_groups * max_gt] (with padding for images with fewer GTs)
        gt_labels_all = []
        gt_boxes_all = []
        valid_masks = []
        for b in range(B):
            gt_lbl = targets_list[b]["labels"].to(device)
            gt_box = targets_list[b]["boxes"].to(device)
            M = len(gt_lbl)
            # Pad to max_gt
            lbl_padded = torch.zeros(max_gt, dtype=torch.long, device=device)
            box_padded = torch.zeros(max_gt, 4, device=device)
            mask = torch.zeros(max_gt, dtype=torch.bool, device=device)
            if M > 0:
                lbl_padded[:M] = gt_lbl - 1  # 0-indexed
                box_padded[:M] = gt_box
                mask[:M] = True
            # Repeat for all groups
            gt_labels_all.append(lbl_padded.repeat(num_groups))   # [num_groups * max_gt]
            gt_boxes_all.append(box_padded.repeat(num_groups, 1))  # [num_groups * max_gt, 4]
            valid_masks.append(mask.repeat(num_groups))            # [num_groups * max_gt]

        gt_labels_all = torch.stack(gt_labels_all)  # [B, num_groups * max_gt]
        gt_boxes_all = torch.stack(gt_boxes_all)     # [B, num_groups * max_gt, 4]
        valid_masks = torch.stack(valid_masks)        # [B, num_groups * max_gt]

        total_pos = valid_masks.sum().item()
        num_pos = max(total_pos, 1)

        total_dn_loss = dn_class.new_tensor(0.0)

        for lid in range(num_layers):
            cls_pred = dn_class[lid]   # [B, pad_size, C]
            box_pred = dn_coord[lid]   # [B, pad_size, 4]

            # Gather positive and negative predictions
            pos_cls = cls_pred[:, all_pos_idx, :]   # [B, num_groups*max_gt, C]
            pos_box = box_pred[:, all_pos_idx, :]   # [B, num_groups*max_gt, 4]
            neg_cls = cls_pred[:, all_neg_idx, :]   # [B, num_groups*max_gt, C]

            # --- Positive classification loss ---
            pos_cls_target = torch.zeros_like(pos_cls)  # [B, num_groups*max_gt, C]
            # Scatter GT labels into one-hot
            pos_cls_target.scatter_(2, gt_labels_all.unsqueeze(-1), 1.0)
            # Mask out invalid (padded) positions
            pos_focal = sigmoid_focal_loss(
                pos_cls, pos_cls_target,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
            )  # [B, num_groups*max_gt, C]
            pos_focal = (pos_focal * valid_masks.unsqueeze(-1)).sum()

            # --- Negative classification loss (all zeros target = background) ---
            neg_cls_target = torch.zeros_like(neg_cls)
            neg_focal = sigmoid_focal_loss(
                neg_cls, neg_cls_target,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
            )
            neg_focal = (neg_focal * valid_masks.unsqueeze(-1)).sum()

            dn_cls_loss = (pos_focal + neg_focal) / num_pos

            # --- Positive box L1 loss ---
            l1 = F.l1_loss(pos_box, gt_boxes_all, reduction="none").sum(-1)  # [B, N]
            dn_box_loss = (l1 * valid_masks).sum() / num_pos

            # --- Positive box GIoU loss ---
            # Flatten valid entries only for GIoU (avoid computing on padding)
            flat_valid = valid_masks.reshape(-1)  # [B * N]
            if flat_valid.any():
                pred_xyxy = box_cxcywh_to_xyxy(pos_box.reshape(-1, 4)[flat_valid])
                gt_xyxy = box_cxcywh_to_xyxy(gt_boxes_all.reshape(-1, 4)[flat_valid])
                giou = giou_pairwise(pred_xyxy, gt_xyxy)
                dn_giou_loss = (1 - giou).sum() / num_pos
            else:
                dn_giou_loss = cls_pred.new_tensor(0.0)

            layer_loss = (
                self.lambda_cls * dn_cls_loss
                + self.lambda_bbox * dn_box_loss
                + self.lambda_giou * dn_giou_loss
            )
            total_dn_loss = total_dn_loss + layer_loss

        return total_dn_loss / num_layers

    def forward(self, outputs, targets_list):
        # Main loss from final decoder layer
        main_loss, cls_loss, bbox_loss, giou_loss = self._compute_single_loss(
            outputs["pred_logits"], outputs["pred_boxes"], targets_list
        )

        # Auxiliary losses from intermediate decoder layers
        aux_loss = outputs["pred_logits"].new_tensor(0.0)
        if self.use_aux_loss and "aux_outputs" in outputs:
            for aux_out in outputs["aux_outputs"]:
                a_loss, _, _, _ = self._compute_single_loss(
                    aux_out["pred_logits"], aux_out["pred_boxes"], targets_list
                )
                aux_loss = aux_loss + a_loss

        # DN loss
        dn_loss = outputs["pred_logits"].new_tensor(0.0)
        if (
            self.use_dn_loss
            and "dn_class" in outputs
            and outputs.get("dn_meta") is not None
        ):
            dn_loss = self._compute_dn_loss(
                outputs["dn_class"], outputs["dn_coord"],
                targets_list, outputs["dn_meta"],
            )

        # Encoder auxiliary loss
        enc_loss = outputs["pred_logits"].new_tensor(0.0)
        if self.use_enc_loss and "enc_outputs" in outputs:
            enc_out = outputs["enc_outputs"]
            e_loss, _, _, _ = self._compute_single_loss(
                enc_out["pred_logits"], enc_out["pred_boxes"], targets_list
            )
            enc_loss = e_loss

        total = (
            main_loss
            + self.lambda_aux * aux_loss
            + self.lambda_dn * dn_loss
            + self.lambda_enc * enc_loss
        )

        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "bbox_loss": bbox_loss.detach(),
            "giou_loss": giou_loss.detach(),
            "dn_loss": dn_loss.detach() if isinstance(dn_loss, torch.Tensor) else dn_loss,
        }


# ---------------------------------------------------------------------------
# COCO mAP evaluation
# ---------------------------------------------------------------------------

def compute_ap(recalls, precisions):
    """Compute area under PR curve using 101-point interpolation."""
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])


@torch.no_grad()
def evaluate_map(model, val_loader, device, iou_thresholds=None, score_thr=0.01):
    """Compute per-class AP and mAP over the validation set."""
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    model.eval()
    all_preds = []
    all_gts = {}

    for images, masks, targets_list in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images, masks)
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        for b, tgt in enumerate(targets_list):
            img_id = int(tgt["image_id"])
            logits = pred_logits[b]
            boxes = pred_boxes[b]

            # Sigmoid-based scoring (DINO uses sigmoid, not softmax)
            scores_per_class = logits.sigmoid()  # [N, C]
            scores, classes = scores_per_class.max(dim=-1)  # [N]

            boxes_xyxy = box_cxcywh_to_xyxy(boxes)

            for n in range(logits.shape[0]):
                s = scores[n].item()
                if s < score_thr:
                    continue
                c = int(classes[n].item()) + 1  # 1-indexed
                b4 = boxes_xyxy[n].tolist()
                all_preds.append((img_id, c, s, b4))

            gt_boxes_xyxy = box_cxcywh_to_xyxy(tgt["boxes"])
            all_gts[img_id] = [
                (int(lbl.item()), gt_boxes_xyxy[i].tolist())
                for i, lbl in enumerate(tgt["labels"])
            ]

    num_classes = 10
    all_aps = []

    for iou_thr in iou_thresholds:
        for cls in range(1, num_classes + 1):
            cls_preds = sorted(
                [(s, img_id, b4) for img_id, c, s, b4 in all_preds if c == cls],
                key=lambda x: -x[0],
            )
            num_gt = sum(
                sum(1 for lbl, _ in gts if lbl == cls)
                for gts in all_gts.values()
            )
            if num_gt == 0:
                continue

            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            matched = {img_id: set() for img_id in all_gts}

            for k, (_, img_id, pred_box) in enumerate(cls_preds):
                gts_for_img = [
                    (i, b4)
                    for i, (lbl, b4) in enumerate(all_gts.get(img_id, []))
                    if lbl == cls
                ]
                best_iou, best_j = 0.0, -1
                for j, gt_box in gts_for_img:
                    iou = _box_iou_single(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if best_iou >= iou_thr and best_j not in matched[img_id]:
                    tp[k] = 1
                    matched[img_id].add(best_j)
                else:
                    fp[k] = 1

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / num_gt
            prec = tp_cum / (tp_cum + fp_cum + 1e-9)
            all_aps.append(compute_ap(rec, prec))

    return float(np.mean(all_aps)) if all_aps else 0.0


def _box_iou_single(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / (union + 1e-6)


# ---------------------------------------------------------------------------
# Detailed evaluation (mAP50, mAP75, digit accuracy by size category)
# ---------------------------------------------------------------------------

def _compute_ap_subset(all_preds, all_gts, img_ids_set, iou_thr, num_classes=10):
    """Compute mAP for a subset of images at a single IoU threshold."""
    gts_sub = {img_id: gts for img_id, gts in all_gts.items() if img_id in img_ids_set}
    all_aps = []
    for cls in range(1, num_classes + 1):
        cls_preds = sorted(
            [(s, img_id, b4) for img_id, c, s, b4 in all_preds if c == cls and img_id in img_ids_set],
            key=lambda x: -x[0],
        )
        num_gt = sum(sum(1 for lbl, _ in gts if lbl == cls) for gts in gts_sub.values())
        if num_gt == 0:
            continue
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        matched = {img_id: set() for img_id in gts_sub}
        for k, (_, img_id, pred_box) in enumerate(cls_preds):
            gts_for_img = [
                (i, b4) for i, (lbl, b4) in enumerate(gts_sub.get(img_id, [])) if lbl == cls
            ]
            best_iou, best_j = 0.0, -1
            for j, gt_box in gts_for_img:
                iou = _box_iou_single(pred_box, gt_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j not in matched[img_id]:
                tp[k] = 1
                matched[img_id].add(best_j)
            else:
                fp[k] = 1
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / num_gt
        prec = tp_cum / (tp_cum + fp_cum + 1e-9)
        all_aps.append(compute_ap(rec, prec))
    return float(np.mean(all_aps)) if all_aps else 0.0


def _compute_digit_acc_subset(all_preds, all_gts, img_ids_set, iou_thr=0.5):
    """Digit recognition accuracy: GT boxes matched at iou_thr with correct class."""
    preds_by_img = {}
    for img_id, c, s, b4 in all_preds:
        if img_id in img_ids_set:
            preds_by_img.setdefault(img_id, []).append((c, s, b4))

    total_gt = 0
    correct = 0
    for img_id, gts in all_gts.items():
        if img_id not in img_ids_set:
            continue
        preds = sorted(preds_by_img.get(img_id, []), key=lambda x: -x[1])
        matched_preds = set()
        for gt_lbl, gt_box in gts:
            total_gt += 1
            best_iou, best_pi = 0.0, -1
            for pi, (pc, ps, pb) in enumerate(preds):
                if pi in matched_preds:
                    continue
                iou = _box_iou_single(pb, gt_box)
                if iou > best_iou:
                    best_iou, best_pi = iou, pi
            if best_iou >= iou_thr and best_pi >= 0:
                matched_preds.add(best_pi)
                if preds[best_pi][0] == gt_lbl:
                    correct += 1
    return correct / total_gt if total_gt > 0 else 0.0


@torch.no_grad()
def evaluate_detailed(model, loader, device, size_info, score_thr=0.01, split_name="Val"):
    """Evaluate mAP50, mAP75, digit accuracy broken down by image/digit size categories."""
    model.eval()
    all_preds = []
    all_gts = {}

    for images, masks, targets_list in loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images, masks)
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        for b, tgt in enumerate(targets_list):
            img_id = int(tgt["image_id"])
            logits = pred_logits[b]
            boxes = pred_boxes[b]
            scores_per_class = logits.sigmoid()
            scores, classes = scores_per_class.max(dim=-1)
            boxes_xyxy = box_cxcywh_to_xyxy(boxes)
            for n in range(logits.shape[0]):
                s = scores[n].item()
                if s < score_thr:
                    continue
                c = int(classes[n].item()) + 1
                all_preds.append((img_id, c, s, boxes_xyxy[n].tolist()))
            gt_boxes_xyxy = box_cxcywh_to_xyxy(tgt["boxes"])
            all_gts[img_id] = [
                (int(lbl.item()), gt_boxes_xyxy[i].tolist())
                for i, lbl in enumerate(tgt["labels"])
            ]

    # Build subset id sets
    all_ids = set(all_gts.keys())
    subsets = {"Overall": all_ids}
    for cat_key, cat_name in [("image_size_cat", "img"), ("digit_size_cat", "dig")]:
        for val in ("small", "medium", "large"):
            ids = {
                int(img_id) for img_id, info in size_info.items()
                if info[cat_key] == val and int(img_id) in all_ids
            }
            subsets[f"{cat_name}={val}"] = ids

    iou_thresholds = np.arange(0.50, 1.00, 0.05)  # 0.50, 0.55, ..., 0.95

    # Compute metrics per subset
    rows = []
    for name, ids in subsets.items():
        if not ids:
            continue
        map50 = _compute_ap_subset(all_preds, all_gts, ids, iou_thr=0.50)
        map75 = _compute_ap_subset(all_preds, all_gts, ids, iou_thr=0.75)
        map95 = _compute_ap_subset(all_preds, all_gts, ids, iou_thr=0.95)
        map5095 = float(np.mean([
            _compute_ap_subset(all_preds, all_gts, ids, iou_thr=t)
            for t in iou_thresholds
        ]))
        dig_acc = _compute_digit_acc_subset(all_preds, all_gts, ids, iou_thr=0.50)
        rows.append((name, len(ids), map50, map75, map95, map5095, dig_acc))

    # --- mAP table ---
    w = 76
    print(f"\n{'='*w}")
    print(f"  {split_name} mAP Breakdown")
    print(f"{'='*w}")
    print(f"  {'Category':<28} {'N':>5}  {'mAP50':>6}  {'mAP75':>6}  {'mAP95':>6}  {'mAP50:95':>8}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")
    for name, n, m50, m75, m95, m5095, _ in rows:
        print(f"  {name:<28} {n:>5}  {m50:>6.4f}  {m75:>6.4f}  {m95:>6.4f}  {m5095:>8.4f}")

    # --- Digit accuracy table ---
    print(f"\n  {'Category':<28} {'N':>5}  {'DigitAcc@IoU0.5':>15}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*15}")
    for name, n, _, _, _, _, acc in rows:
        print(f"  {name:<28} {n:>5}  {acc:>15.4f}")
    print(f"{'='*w}\n")

    result = {
        name: {"mAP50": m50, "mAP75": m75, "mAP95": m95, "mAP50:95": m5095, "dig_acc": acc}
        for name, _, m50, m75, m95, m5095, acc in rows
    }
    return result


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, epoch,
    grad_accum=1, max_norm=0.1,
):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_giou_loss = 0.0
    total_dn_loss = 0.0
    n_batches = len(loader)
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Train E{epoch}", unit="batch", dynamic_ncols=True)
    for i, (images, masks, targets_list) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            # DINO needs targets for DN during training
            outputs = model(images, masks, targets=targets_list)
            if (torch.isnan(outputs["pred_logits"]).any() or
                    torch.isnan(outputs["pred_boxes"]).any()):
                print(f"\n[warn] NaN in model output at batch {i}, skipping")
                optimizer.zero_grad()
                continue
            loss_dict = criterion(outputs, targets_list)
            loss = loss_dict["loss"] / grad_accum

        scaler.scale(loss).backward()

        if (i + 1) % grad_accum == 0 or (i + 1) == n_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss_dict["loss"].item()
        total_cls_loss += loss_dict["cls_loss"].item()
        total_bbox_loss += loss_dict["bbox_loss"].item()
        total_giou_loss += loss_dict["giou_loss"].item()
        dn_val = loss_dict["dn_loss"].item() if isinstance(loss_dict["dn_loss"], torch.Tensor) else 0.0
        total_dn_loss += dn_val
        avg_loss = total_loss / (i + 1)
        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            cls=f"{loss_dict['cls_loss'].item():.4f}",
            bbox=f"{loss_dict['bbox_loss'].item():.4f}",
            giou=f"{loss_dict['giou_loss'].item():.4f}",
            dn=f"{dn_val:.4f}",
        )

    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls_loss / n_batches,
        "bbox_loss": total_bbox_loss / n_batches,
        "giou_loss": total_giou_loss / n_batches,
        "dn_loss": total_dn_loss / n_batches,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TIME_BUDGET_SECONDS = float("inf")


def main(args):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_side=args.max_side,
        min_side=args.min_side,
        train_scales=args.train_scales,
        aug_cfg=args.aug_cfg,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    # Load size category info for detailed evaluation
    valid_size_path = os.path.join(args.data_dir, "valid_size.json")
    with open(valid_size_path) as f:
        valid_size_info = json.load(f)
    print(f"Loaded size info: {len(valid_size_info)} val images")

    model = build_model(
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        d_model=args.d_model,
        n_heads=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ffn=args.dim_feedforward,
        dropout=args.dropout,
        n_levels=args.n_levels,
        n_points=args.n_points,
        pretrained_backbone=True,
        dn_number=args.dn_number,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_box_noise_scale=args.dn_box_noise_scale,
        use_aff=args.use_aff,
        aff_kernel_size=args.aff_kernel_size,
    )
    model = model.to(device)

    criterion = DINOLoss(
        num_classes=args.num_classes,
        lambda_cls=args.lambda_cls,
        lambda_bbox=args.lambda_bbox,
        lambda_giou=args.lambda_giou,
        use_aux_loss=args.use_aux_loss,
        use_enc_loss=args.use_enc_loss,
        use_dn_loss=args.use_dn_loss,
        lambda_aux=args.lambda_aux,
        lambda_enc=args.lambda_enc,
        lambda_dn=args.lambda_dn,
    )

    # Separate LRs: lower for pretrained backbone
    backbone_params = list(model.backbone.parameters())
    other_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("backbone.")
    ]
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    else:
        scheduler = MultiStepLR(
            optimizer,
            milestones=[int(args.epochs * 0.9)],
            gamma=0.1,
        )

    scaler = GradScaler(enabled=device.type == "cuda")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    start_epoch = 1

    def _load_model_weights(ckpt_path, label="checkpoint"):
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_sd = ckpt["model_state_dict"]
        model_sd = model.state_dict()
        compatible = {
            k: v for k, v in ckpt_sd.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        skipped = [k for k in ckpt_sd if k not in compatible]
        result = model.load_state_dict(compatible, strict=False)
        print(f"Loaded weights from {ckpt_path} [{label}] "
              f"(epoch={ckpt.get('epoch')}, mAP={ckpt.get('val_map', 0):.4f})")
        print(f"  Loaded: {len(compatible)}/{len(ckpt_sd)} keys")
        if skipped:
            print(f"  Skipped (shape mismatch, {len(skipped)} keys): {skipped}")
        if result.missing_keys:
            print(f"  Missing (random init, {len(result.missing_keys)} keys): {result.missing_keys}")
        return ckpt

    if args.load_weight:
        _load_model_weights(args.load_weight, label="weights-only init")

    if args.resume:
        ckpt = _load_model_weights(args.resume, label="full resume")
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print("  Optimizer state restored.")
            except Exception as e:
                print(f"  Could not restore optimizer state: {e}")
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
            for _ in range(ckpt["epoch"]):
                scheduler.step()
            print(f"  Scheduler fast-forwarded to epoch {ckpt['epoch']}. Resuming from epoch {start_epoch}.")

    best_map = 0.0
    best_ckpt_path = None
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"train_DINO_{run_ts}.json")
    training_log = {
        "config": vars(args),
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "epochs": [],
    }

    wall_start = time.time()
    training_seconds = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch,
            grad_accum=args.grad_accum,
            max_norm=args.max_norm,
        )
        train_loss = train_metrics["loss"]
        t_val = time.time()

        val_detail = evaluate_detailed(
            model, val_loader, device, valid_size_info,
            score_thr=args.val_score_thr, split_name="Val",
        )

        overall = val_detail.get("Overall", {})
        val_map50 = overall.get("mAP50", 0.0)
        val_map75 = overall.get("mAP75", 0.0)
        val_map5095 = overall.get("mAP50:95", 0.0)
        val_dig_acc = overall.get("dig_acc", 0.0)
        val_map = val_map5095  # use mAP50:95 for best-model tracking
        val_time = time.time() - t_val

        scheduler.step()

        epoch_time = time.time() - t0
        training_seconds += epoch_time
        lr_trans = optimizer.param_groups[1]["lr"]
        lr_bb = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"LR: {lr_trans:.2e} (trans) / {lr_bb:.2e} (backbone) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val mAP50: {val_map50:.4f} | mAP75: {val_map75:.4f} | "
            f"mAP50:95: {val_map5095:.4f} | DigAcc: {val_dig_acc:.4f} | "
            f"Time: {epoch_time:.1f}s (val: {val_time:.1f}s)"
        )

        is_best = val_map > best_map
        if is_best:
            best_map = val_map
            best_ckpt_path = os.path.join(
                args.save_dir, f"best_model_DINO_{run_ts}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_map": val_map,
                },
                best_ckpt_path,
            )
            print(f"  -> Best model saved (mAP={val_map:.4f})")

        training_log["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_cls_loss": round(train_metrics["cls_loss"], 6),
                "train_bbox_loss": round(train_metrics["bbox_loss"], 6),
                "train_giou_loss": round(train_metrics["giou_loss"], 6),
                "train_dn_loss": round(train_metrics["dn_loss"], 6),
                "val_map50": round(val_map50, 6),
                "val_map75": round(val_map75, 6),
                "val_map5095": round(val_map5095, 6),
                "val_dig_acc": round(val_dig_acc, 6),
                "val_detail": {
                    k: {mk: round(mv, 6) for mk, mv in v.items()}
                    for k, v in val_detail.items()
                },
                "val_map": round(val_map, 6),
                "elapsed_sec": round(epoch_time, 2),
                "val_time_sec": round(val_time, 2),
                "is_best": is_best,
            }
        )
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        if training_seconds >= TIME_BUDGET_SECONDS:
            print(f"Time budget reached after epoch {epoch}. Stopping.")
            break

    total_sec = time.time() - wall_start
    peak_vram = 0.0
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024
    num_params_m = sum(p.numel() for p in model.parameters()) / 1e6

    training_log["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    training_log["best_val_map"] = round(best_map, 6)
    training_log["training_seconds"] = round(training_seconds, 1)
    training_log["total_seconds"] = round(total_sec, 1)
    training_log["peak_vram_mb"] = round(peak_vram, 1)
    training_log["num_params_M"] = round(num_params_m, 1)
    training_log["best_checkpoint"] = best_ckpt_path
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nTraining complete. Best Val mAP: {best_map:.4f}")
    print(f"Training log: {log_path}")
    if best_ckpt_path:
        print(f"Best checkpoint: {best_ckpt_path}")
    print(f"peak_vram_mb:  {peak_vram:.1f}")
    print(f"num_params_M:  {num_params_m:.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DINO for digit detection")
    parser.add_argument("--config", default="configs/default_DINO.yaml")
    # Data
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--log_dir", default=None)
    # Training
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_side", type=int, default=None)
    parser.add_argument("--min_side", type=int, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None)
    # Model
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num_encoder_layers", type=int, default=None)
    parser.add_argument("--num_decoder_layers", type=int, default=None)
    parser.add_argument("--dim_feedforward", type=int, default=None)
    parser.add_argument("--n_levels", type=int, default=None)
    parser.add_argument("--n_points", type=int, default=None)
    # AFF
    parser.add_argument("--use_aff", type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=None, help="Enable Adaptive Feature Fusion")
    parser.add_argument("--aff_kernel_size", type=int, default=None)
    # DN
    parser.add_argument("--dn_number", type=int, default=None)
    parser.add_argument("--dn_label_noise_ratio", type=float, default=None)
    parser.add_argument("--dn_box_noise_scale", type=float, default=None)
    # Loss
    parser.add_argument("--lambda_cls", type=float, default=None)
    parser.add_argument("--lambda_bbox", type=float, default=None)
    parser.add_argument("--lambda_giou", type=float, default=None)
    parser.add_argument("--use_aux_loss", type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=None, help="Enable decoder intermediate-layer auxiliary loss")
    parser.add_argument("--use_enc_loss", type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=None, help="Enable encoder auxiliary loss")
    parser.add_argument("--use_dn_loss", type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=None, help="Enable contrastive denoising (DN) loss")
    parser.add_argument("--lambda_aux", type=float, default=None)
    parser.add_argument("--lambda_enc", type=float, default=None)
    parser.add_argument("--lambda_dn", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--max_norm", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val_score_thr", type=float, default=None)
    parser.add_argument("--load_weight", type=str, default=None,
                        help="Init model weights from checkpoint (epoch/optimizer state ignored)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Continue training: restores model, optimizer, scheduler, and epoch")

    args = parser.parse_args()
    cfg = load_config(args.config)

    # Extract augmentation config directly from YAML (not through argparse)
    aug_cfg = {k: v for k, v in cfg.items() if k.startswith("aug_")}
    args.aug_cfg = aug_cfg if aug_cfg else None

    for key, val in vars(args).items():
        if key in ("config", "aug_cfg"):
            continue
        if val is None and key in cfg:
            setattr(args, key, cfg[key])

    defaults = dict(
        data_dir="nycu-hw2-data",
        save_dir="checkpoints",
        log_dir="log",
        epochs=50,
        batch_size=4,
        lr=1e-4,
        backbone_lr=1e-5,
        weight_decay=1e-4,
        dropout=0.1,
        num_workers=4,
        max_side=640,
        min_side=480,
        train_scales=None,
        lr_scheduler="cosine",
        num_classes=10,
        num_queries=300,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        n_levels=4,
        n_points=4,
        use_aff=False,
        aff_kernel_size=3,
        dn_number=100,
        dn_label_noise_ratio=0.5,
        dn_box_noise_scale=0.4,
        lambda_cls=1.0,
        lambda_bbox=5.0,
        lambda_giou=2.0,
        use_aux_loss=True,
        use_enc_loss=True,
        use_dn_loss=True,
        lambda_aux=1.0,
        lambda_enc=1.0,
        lambda_dn=1.0,
        grad_accum=1,
        max_norm=0.1,
        device="cuda",
        val_score_thr=0.01,
    )
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    main(args)
