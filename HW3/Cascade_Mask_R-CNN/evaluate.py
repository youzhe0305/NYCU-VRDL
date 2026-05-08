import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
import torchvision
import yaml
import tifffile
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm


# ─── TTA helpers ───


def _encode_submission_mask(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return {
        "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
        "counts": rle["counts"],
    }


def _append_submission_result(results, image_id, label, score, binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
    if binary_mask.sum() < 1:
        return
    results.append({
        "image_id": int(image_id),
        "category_id": int(label) + 1,
        "segmentation": _encode_submission_mask(binary_mask),
        "score": round(float(score), 6),
    })


def _hw3_path(path):
    if path is None or os.path.isabs(path):
        return path
    hw3_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(hw3_root, path)


def _load_test_image(path):
    image = tifffile.imread(path)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] >= 4:
        image = image[:, :, :3]
    if image.ndim == 3 and image.shape[2] == 3:
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image, dtype=np.uint8)


def _build_detector(args, checkpoint, score_threshold):
    import dataset  # noqa: F401 - registers LoadTifImageFromFile
    import model as model_module  # noqa: F401 - registers backbone
    from dataset import get_test_pipeline
    from mmdet.apis import init_detector
    from mmengine.config import Config
    from model import _build_model_cfg

    cfg_dict = vars(args).copy()
    cfg_dict["pretrained"] = False
    cfg_dict["score_threshold"] = score_threshold
    model_cfg = _build_model_cfg(cfg_dict)
    full_cfg = dict(
        model=model_cfg,
        default_scope="mmdet",
        test_dataloader=dict(
            dataset=dict(
                type="CocoDataset",
                pipeline=get_test_pipeline(args.img_scale),
            ),
        ),
        test_cfg=dict(type="TestLoop"),
        test_evaluator=dict(type="CocoMetric", metric=["segm"]),
    )
    detector = init_detector(Config(full_cfg), checkpoint, device=args.device)
    detector.eval()
    return detector


def _flip_image(image, direction):
    if direction == "horizontal":
        return cv2.flip(image, 1)
    if direction == "vertical":
        return cv2.flip(image, 0)
    return image


def _flip_numpy_predictions(boxes, masks, direction, height, width):
    boxes = boxes.copy()
    if direction == "horizontal":
        x1 = boxes[:, 0].copy()
        x2 = boxes[:, 2].copy()
        boxes[:, 0] = width - x2
        boxes[:, 2] = width - x1
        masks = np.flip(masks, axis=2).copy() if len(masks) else masks
    elif direction == "vertical":
        y1 = boxes[:, 1].copy()
        y2 = boxes[:, 3].copy()
        boxes[:, 1] = height - y2
        boxes[:, 3] = height - y1
        masks = np.flip(masks, axis=1).copy() if len(masks) else masks
    return boxes, masks


def _resize_masks(masks, height, width):
    masks = np.asarray(masks)
    if len(masks) == 0:
        return np.zeros((0, height, width), dtype=np.uint8)
    if masks.shape[1] == height and masks.shape[2] == width:
        return masks.astype(np.uint8)
    resized = np.empty((len(masks), height, width), dtype=np.uint8)
    for i, mask in enumerate(masks):
        resized[i] = cv2.resize(
            mask.astype(np.uint8), (width, height),
            interpolation=cv2.INTER_NEAREST)
    return resized


def _predict_one_model(detector, image, args, model_idx=0):
    from mmdet.apis import inference_detector

    height, width = image.shape[:2]
    views = [None, "horizontal", "vertical"] if args.tta else [None]
    all_boxes, all_labels, all_scores, all_masks = [], [], [], []

    for direction in views:
        view = _flip_image(image, direction) if direction else image
        result = inference_detector(detector, view)
        pred = result.pred_instances
        boxes = pred.bboxes.cpu().numpy().astype(np.float32)
        labels = pred.labels.cpu().numpy().astype(np.int64)
        scores = pred.scores.cpu().numpy().astype(np.float32)
        masks = pred.masks.cpu().numpy().astype(np.uint8) \
            if hasattr(pred, "masks") and pred.masks is not None \
            else np.zeros((0, height, width), dtype=np.uint8)

        if direction:
            boxes, masks = _flip_numpy_predictions(
                boxes, masks, direction, height, width)
        all_boxes.append(boxes)
        all_labels.append(labels)
        all_scores.append(scores)
        all_masks.append(_resize_masks(masks, height, width))

    boxes = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4), np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.zeros(0, np.int64)
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0, np.float32)
    masks = np.concatenate(all_masks) if all_masks else \
        np.zeros((0, height, width), dtype=np.uint8)
    model_ids = np.full(len(scores), model_idx, dtype=np.int32)

    keep = scores >= args.per_model_score_threshold
    boxes, labels, scores, masks, model_ids = (
        boxes[keep], labels[keep], scores[keep], masks[keep], model_ids[keep])

    if len(scores) > 0 and args.tta:
        keep_indices = []
        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            kept = torchvision.ops.nms(
                torch.from_numpy(boxes[cls_idx]).float(),
                torch.from_numpy(scores[cls_idx]).float(),
                args.nms_threshold,
            ).numpy()
            keep_indices.extend(cls_idx[kept].tolist())
        keep_indices = np.asarray(sorted(keep_indices), dtype=np.int64)
        boxes, labels, scores, masks, model_ids = (
            boxes[keep_indices], labels[keep_indices], scores[keep_indices],
            masks[keep_indices], model_ids[keep_indices])

    return boxes, labels, scores, masks, model_ids


def _find_checkpoint(args):
    work_dir = _hw3_path(args.work_dir)
    candidates = sorted(glob.glob(os.path.join(
        work_dir, "best_coco_segm_mAP_50*.pth")))
    if candidates:
        return candidates[-1]
    legacy_candidates = sorted(glob.glob(os.path.join(
        _hw3_path(args.save_dir), "best_model*.pth")))
    return legacy_candidates[-1] if legacy_candidates else None


def run_mmdet_inference(args):
    data_dir = _hw3_path(args.data_dir)
    test_dir = _hw3_path(args.test_dir) if args.test_dir else \
        os.path.join(data_dir, "test_release")
    mapping_path = _hw3_path(args.id_mapping) if args.id_mapping else \
        os.path.join(data_dir, "test_image_name_to_ids.json")

    with open(mapping_path) as f:
        id_mapping = json.load(f)
    name_to_info = {item["file_name"]: item for item in id_mapping}

    checkpoint = args.checkpoint or _find_checkpoint(args)
    if not checkpoint:
        raise FileNotFoundError("No checkpoint found for inference.")
    detector = _build_detector(
        args, checkpoint, args.per_model_score_threshold)

    test_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".tif"))
    results = []

    for fname in tqdm(test_files, desc="Inference"):
        if fname not in name_to_info:
            continue
        info = name_to_info[fname]
        image_id = info["id"]
        image = _load_test_image(os.path.join(test_dir, fname))

        boxes, labels, scores, masks, _ = _predict_one_model(
            detector, image, args, model_idx=0)
        for label, score, mask in zip(labels, scores, masks):
            if score < args.score_threshold:
                continue
            _append_submission_result(results, image_id, label, score, mask)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Cascade Mask R-CNN inference")
    parser.add_argument("--config", default="configs/cascade_mask_rcnn.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path. If omitted, the best checkpoint in work_dir is used.")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--test_dir", default=None)
    parser.add_argument("--id_mapping", default=None)
    parser.add_argument("--work_dir", default=None)
    parser.add_argument("--output", default="test-results-cascade.json")
    parser.add_argument("--mode", default="test",
                        choices=["test"], help=argparse.SUPPRESS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument("--per_model_score_threshold", type=float, default=None)
    parser.add_argument("--mask_threshold", type=float, default=None)
    parser.add_argument("--nms_threshold", type=float, default=None)
    parser.add_argument("--max_det", type=int, default=None)
    parser.add_argument("--img_scale", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--coco_verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path) and not os.path.isabs(config_path):
        config_path = _hw3_path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg.items():
        if not hasattr(args, key):
            setattr(args, key, val)
    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is None and key in cfg:
            setattr(args, key, cfg[key])

    defaults = {
        "data_dir": "data",
        "test_dir": None,
        "id_mapping": None,
        "work_dir": "work_dirs/cascade_mask_rcnn",
        "device": "cuda:0",
        "seed": 42,
        "tta": True,
        "score_threshold": 0.05,
        "per_model_score_threshold": 0.03,
        "mask_threshold": 0.5,
        "nms_threshold": 0.5,
        "max_det": 300,
        "img_scale": 1024,
        "num_classes": 4,
        "num_workers": 4,
        "coco_verbose": False,
        "pretrained": False,
        "detector_type": "CascadeRCNN",
        "backbone_type": "ConvNeXtV2Backbone",
        "backbone_name": "convnextv2_base",
        "backbone_out_channels": None,
        "backbone_img_size": None,
        "backbone_depth": 50,
        "resnet_frozen_stages": -1,
        "backbone_norm_type": "LayerNorm",
        "drop_path_rate": 0.4,
        "neck_type": "FPN",
        "fpn_out_channels": 256,
        "fpn_num_outs": 5,
        "nasfpn_stack_times": 1,
        "neck_no_norm_on_lateral": None,
        "neck_add_extra_convs": None,
        "neck_relu_before_extra_convs": None,
        "neck_upsample_cfg": None,
        "neck_norm_cfg": None,
        "neck_act_cfg": None,
        "rpn_strides": [4, 8, 16, 32, 64],
        "rpn_head_type": "RPNHead",
        "rpn_loss_cls_type": "CrossEntropyLoss",
        "rpn_focal_gamma": 2.0,
        "rpn_focal_alpha": 0.25,
        "rpn_loss_cls_weight": 1.0,
        "rpn_loss_bbox_weight": 1.0,
        "rpn_assigner_type": "MaxIoUAssigner",
        "rpn_atss_topk": 9,
        "rpn_sampler_type": "RandomSampler",
        "rpn_pos_sampler_type": "InstanceBalancedPosSampler",
        "rpn_neg_sampler_type": "IoUBalancedNegSampler",
        "rpn_iou_balanced_floor_thr": -1,
        "rpn_iou_balanced_floor_fraction": 0,
        "rpn_iou_balanced_num_bins": 3,
        "ga_scales_per_octave": 3,
        "ga_approx_ratios": [0.5, 1.0, 2.0],
        "ga_loc_filter_thr": 0.01,
        "ga_allowed_border": -1,
        "ga_center_ratio": 0.2,
        "ga_ignore_ratio": 0.5,
        "ga_anchor_target_stds": [0.07, 0.07, 0.14, 0.14],
        "ga_bbox_target_stds": [0.07, 0.07, 0.11, 0.11],
        "roi_featmap_strides": [4, 8, 16, 32],
        "bbox_roi_extractor_type": "SingleRoIExtractor",
        "mask_roi_extractor_type": "SingleRoIExtractor",
        "roi_layer_type": "RoIAlign",
        "roi_aggregation": "sum",
        "roi_sampling_ratio": 0,
        "freeze_stages": 0,
        "mask_head_type": "FCNMaskHead",
        "bbox_head_type": "Shared2FCBBoxHead",
        "bbox_loss_cls_type": "CrossEntropyLoss",
        "bbox_focal_gamma": 2.0,
        "bbox_focal_alpha": 0.25,
        "bbox_loss_cls_weight": 1.0,
        "bbox_loss_type": "SmoothL1Loss",
        "bbox_loss_weight": 1.0,
        "bbox_smooth_l1_beta": 1.0,
        "bbox_balanced_l1_alpha": 0.5,
        "bbox_balanced_l1_gamma": 1.5,
        "bbox_balanced_l1_beta": 1.0,
        "bbox_reg_decoded_bbox": False,
        "bbox_num_shared_convs": 2,
        "bbox_num_shared_fcs": 1,
        "bbox_num_convs": 4,
        "bbox_num_fcs": 2,
        "bbox_conv_out_channels": 256,
        "mask_class_agnostic": False,
        "mask_conv_to_res": True,
        "mask_dice_loss_weight": 1.0,
        "mask_ce_loss_weight": 1.0,
        "rcnn_sampler_type": "RandomSampler",
        "rcnn_pos_sampler_type": "InstanceBalancedPosSampler",
        "rcnn_neg_sampler_type": "IoUBalancedNegSampler",
        "rcnn_iou_balanced_floor_thr": -1,
        "rcnn_iou_balanced_floor_fraction": 0,
        "rcnn_iou_balanced_num_bins": 3,
        "save_dir": "checkpoints/cascade_mask_rcnn",
    }
    for key, default in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, cfg.get(key, default))

    run_mmdet_inference(args)
