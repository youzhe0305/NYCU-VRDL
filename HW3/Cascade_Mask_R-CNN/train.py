import argparse
import contextlib
import io
import json
import os
import random
import warnings
from datetime import datetime

import cv2
import numpy as np
import torch
import yaml
from mmdet.evaluation import CocoMetric
from mmdet.registry import METRICS
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.amp import autocast
from tqdm.auto import tqdm

from model import detector_blueprint
from runtime_defaults import TRAIN_FALLBACKS, populate_runtime_args

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"mmdet\.models\.roi_heads\.mask_heads\.fcn_mask_head",
)
warnings.filterwarnings(
    "ignore",
    message=r"To copy construct from a tensor.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*sourceTensor\.clone\(\)\.detach\(\).*",
    category=UserWarning,
)


@METRICS.register_module(force=True)
class QuietCocoMetric(CocoMetric):
    def compute_metrics(self, results):
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return super().compute_metrics(results)


@HOOKS.register_module(force=True)
class CompactTrainLogHook(Hook):
    def __init__(self):
        self.best_mAP50 = float("-inf")

    def before_train_epoch(self, runner):
        self.loss_sum = 0.0
        self.loss_count = 0

    def before_val_epoch(self, runner):
        if "best_score" in runner.message_hub.runtime_info:
            self.best_mAP50 = float(runner.message_hub.get_info("best_score"))

    def after_train_iter(self, runner, batch_idx,
                         data_batch=None, outputs=None):
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
            if hasattr(loss, "detach"):
                loss = loss.detach().float().item()
            self.loss_sum += float(loss)
            self.loss_count += 1

    def after_train_epoch(self, runner):
        avg_loss = self.loss_sum / max(self.loss_count, 1)
        param_groups = runner.optim_wrapper.optimizer.param_groups
        head_lr = param_groups[-1]["lr"] if param_groups else 0.0
        backbone_lr = param_groups[0]["lr"] if param_groups else head_lr
        epoch = runner.epoch + 1
        max_epochs = runner.max_epochs
        print(
            f"Epoch {epoch}/{max_epochs} | "
            f"loss={avg_loss:.4f} | lr={head_lr:.3e} | "
            f"backbone_lr={backbone_lr:.3e}",
            flush=True,
        )

    def after_val_epoch(self, runner, metrics=None):
        metrics = metrics or {}
        epoch = max(int(runner.epoch), 1)
        segm_ap50 = metrics.get("coco/segm_mAP_50")
        if segm_ap50 is not None:
            segm_ap50 = float(segm_ap50)
            print(f"Epoch {epoch} validation | mAP50={segm_ap50:.4f}",
                  flush=True)
            if segm_ap50 > self.best_mAP50:
                self.best_mAP50 = segm_ap50
                print(f"-> Best model saved (mAP={segm_ap50:.4f})",
                      flush=True)


@HOOKS.register_module(force=True)
class TrainingArtifactsHook(Hook):
    def __init__(self, num_classes=4, score_thr=0.05, mask_iou_thr=0.5,
                 num_visualizations=6, seed=42, draw_score_thr=0.3,
                 interval=1, out_dir_name="analysis"):
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.score_thr = score_thr
        self.mask_iou_thr = mask_iou_thr
        self.num_visualizations = num_visualizations
        self.seed = seed
        self.draw_score_thr = draw_score_thr
        self.interval = max(int(interval), 1)
        self.out_dir_name = out_dir_name
        self.train_history = []
        self.val_history = []
        self.loss_sum = 0.0
        self.loss_count = 0
        self._vis_indices = None
        self._val_seen = 0
        self._vis_samples = []
        self._confusion = None

    def before_train_epoch(self, runner):
        self.loss_sum = 0.0
        self.loss_count = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        if not isinstance(outputs, dict) or "loss" not in outputs:
            return
        loss = outputs["loss"]
        if hasattr(loss, "detach"):
            loss = loss.detach().float().item()
        self.loss_sum += float(loss)
        self.loss_count += 1

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        avg_loss = self.loss_sum / max(self.loss_count, 1)
        self.train_history.append({"epoch": epoch, "loss": avg_loss})

    def before_val_epoch(self, runner):
        self._val_seen = 0
        self._vis_samples = []
        self._confusion = np.zeros(
            (self.num_classes + 1, self.num_classes + 1),
            dtype=np.int64,
        )
        if self._vis_indices is None:
            dataset_len = len(runner.val_dataloader.dataset)
            k = min(self.num_visualizations, dataset_len)
            rng = random.Random(self.seed)
            self._vis_indices = set(rng.sample(range(dataset_len), k))

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        data_batch = data_batch or {}
        data_samples = data_batch.get("data_samples", [])
        inputs = data_batch.get("inputs", [])
        outputs = outputs or []
        if isinstance(inputs, torch.Tensor):
            inputs = list(inputs)

        for inp, data_sample, output in zip(inputs, data_samples, outputs):
            self._update_confusion(data_sample, output)
            if self._val_seen in self._vis_indices:
                self._vis_samples.append(
                    (self._image_from_input(inp), data_sample, output))
            self._val_seen += 1

    def after_val_epoch(self, runner, metrics=None):
        metrics = metrics or {}
        epoch = max(int(runner.epoch), 1)
        segm_ap50 = metrics.get("coco/segm_mAP_50")
        if segm_ap50 is not None:
            self.val_history.append({
                "epoch": epoch,
                "coco/segm_mAP_50": float(segm_ap50),
            })

        if epoch % self.interval != 0:
            return

        out_dir = os.path.join(runner.work_dir, self.out_dir_name)
        os.makedirs(out_dir, exist_ok=True)
        self._write_history(out_dir)
        self._plot_curves(out_dir)
        self._write_confusion(out_dir)
        self._plot_confusion(out_dir)
        self._save_visualizations(out_dir, epoch)

    def _write_history(self, out_dir):
        train_by_epoch = {d["epoch"]: d["loss"] for d in self.train_history}
        val_by_epoch = {
            d["epoch"]: d["coco/segm_mAP_50"] for d in self.val_history
        }
        epochs = sorted(set(train_by_epoch) | set(val_by_epoch))
        path = os.path.join(out_dir, "training_history.tsv")
        with open(path, "w") as f:
            f.write("epoch\ttrain_loss\tval_segm_mAP50\n")
            for epoch in epochs:
                f.write(
                    f"{epoch}\t{train_by_epoch.get(epoch, '')}\t"
                    f"{val_by_epoch.get(epoch, '')}\n")

    def _plot_curves(self, out_dir):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            with open(os.path.join(out_dir, "plot_error.txt"), "a") as f:
                f.write(f"training_curves.png: {exc}\n")
            return

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
        if self.train_history:
            axes[0].plot([d["epoch"] for d in self.train_history],
                         [d["loss"] for d in self.train_history],
                         marker="o")
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        if self.val_history:
            axes[1].plot([d["epoch"] for d in self.val_history],
                         [d["coco/segm_mAP_50"] for d in self.val_history],
                         marker="o", color="tab:orange")
        axes[1].set_title("Validation Segm mAP@50")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mAP@50")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training_curves.png"))
        plt.close(fig)

    def _write_confusion(self, out_dir):
        labels = [f"class{i + 1}" for i in range(self.num_classes)] + ["bg"]
        path = os.path.join(out_dir, "confusion_matrix.tsv")
        with open(path, "w") as f:
            f.write("gt\\pred\t" + "\t".join(labels) + "\n")
            for label, row in zip(labels, self._confusion):
                f.write(label + "\t" + "\t".join(map(str, row.tolist())) +
                        "\n")

    def _plot_confusion(self, out_dir):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            with open(os.path.join(out_dir, "plot_error.txt"), "a") as f:
                f.write(f"confusion_matrix.png: {exc}\n")
            return

        labels = [f"class{i + 1}" for i in range(self.num_classes)] + ["bg"]
        cm = self._confusion.astype(np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, np.maximum(row_sums, 1.0))

        fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45,
                      ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("Validation Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        color="black" if cm_norm[i, j] < 0.5 else "white",
                        fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "confusion_matrix.png"))
        plt.close(fig)

    def _save_visualizations(self, out_dir, epoch):
        vis_dir = os.path.join(out_dir, "val_visualizations",
                               f"epoch_{epoch:03d}")
        os.makedirs(vis_dir, exist_ok=True)
        panels = []
        for i, (image, data_sample, output) in enumerate(self._vis_samples):
            img_id = data_sample.metainfo.get("img_id", i)
            gt_img = self._draw_instances(image.copy(), data_sample,
                                          is_prediction=False)
            pred_img = self._draw_instances(image.copy(), output,
                                            is_prediction=True)
            canvas = np.concatenate([gt_img, pred_img], axis=1)
            cv2.putText(canvas, f"GT | Pred  img_id={img_id}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)
            fname = f"sample_{i:02d}_img_{img_id}.png"
            out_path = os.path.join(vis_dir, fname)
            cv2.imwrite(out_path, canvas)
            panels.append(canvas)
        if panels:
            self._save_contact_sheet(panels, os.path.join(
                vis_dir, "contact_sheet.png"))

    def _save_contact_sheet(self, panels, out_path):
        max_h = max(p.shape[0] for p in panels)
        max_w = max(p.shape[1] for p in panels)
        padded = []
        for panel in panels:
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            canvas[:panel.shape[0], :panel.shape[1]] = panel
            padded.append(canvas)
        rows = []
        for i in range(0, len(padded), 2):
            row = padded[i:i + 2]
            if len(row) == 1:
                row.append(np.zeros_like(row[0]))
            rows.append(np.concatenate(row, axis=1))
        cv2.imwrite(out_path, np.concatenate(rows, axis=0))

    def _update_confusion(self, data_sample, output):
        gt_labels = self._labels(data_sample.gt_instances)
        gt_masks = self._masks(data_sample.gt_instances)
        pred_instances = output.pred_instances
        pred_labels = self._labels(pred_instances)
        pred_scores = self._scores(pred_instances)
        pred_masks = self._masks(pred_instances)

        keep = pred_scores >= self.score_thr
        pred_labels = pred_labels[keep]
        pred_masks = pred_masks[keep]

        matched_gt = set()
        matched_pred = set()
        pairs = []
        for gi, gt_mask in enumerate(gt_masks):
            for pi, pred_mask in enumerate(pred_masks):
                pairs.append((self._mask_iou(gt_mask, pred_mask), gi, pi))
        for iou, gi, pi in sorted(pairs, reverse=True):
            if iou < self.mask_iou_thr:
                break
            if gi in matched_gt or pi in matched_pred:
                continue
            self._confusion[int(gt_labels[gi]), int(pred_labels[pi])] += 1
            matched_gt.add(gi)
            matched_pred.add(pi)

        for gi, label in enumerate(gt_labels):
            if gi not in matched_gt:
                self._confusion[int(label), self.bg_idx] += 1
        for pi, label in enumerate(pred_labels):
            if pi not in matched_pred:
                self._confusion[self.bg_idx, int(label)] += 1

    def _draw_instances(self, image, data_sample, is_prediction):
        img_shape = data_sample.metainfo.get(
            "img_shape", image.shape[:2],
        )
        image = image[:img_shape[0], :img_shape[1]].copy()
        instances = (data_sample.pred_instances if is_prediction
                     else data_sample.gt_instances)
        labels = self._labels(instances)
        masks = self._masks(instances)
        scores = (self._scores(instances) if is_prediction
                  else np.ones(len(labels), dtype=np.float32))
        order = np.argsort(scores)[::-1]
        for idx in order:
            if is_prediction and scores[idx] < self.draw_score_thr:
                continue
            label = int(labels[idx])
            color = self._color(label)
            mask = self._resize_mask_to_image(masks[idx], image.shape[:2])
            image[mask] = (0.55 * image[mask] + 0.45 *
                           np.array(color, dtype=np.float32)).astype(np.uint8)
            x1, y1, x2, y2 = self._bbox_from_mask(mask)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                text = f"c{label + 1}"
                if is_prediction:
                    text += f" {scores[idx]:.2f}"
                cv2.putText(image, text, (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                            cv2.LINE_AA)
        title = "Pred" if is_prediction else "GT"
        cv2.putText(image, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return image

    @staticmethod
    def _resize_mask_to_image(mask, image_shape):
        mask = mask.astype(bool)
        h, w = image_shape
        if mask.shape[:2] == (h, w):
            return mask
        return cv2.resize(
            mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    def _image_from_input(self, inp):
        image = inp.detach().cpu().numpy() if hasattr(inp, "detach") else inp
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))

    def _labels(self, instances):
        labels = getattr(instances, "labels", None)
        if labels is None:
            return np.zeros((0,), dtype=np.int64)
        return labels.detach().cpu().numpy().astype(np.int64)

    def _scores(self, instances):
        scores = getattr(instances, "scores", None)
        if scores is None:
            labels = self._labels(instances)
            return np.ones(len(labels), dtype=np.float32)
        return scores.detach().cpu().numpy().astype(np.float32)

    def _masks(self, instances):
        masks = getattr(instances, "masks", None)
        if masks is None:
            return np.zeros((0, 1, 1), dtype=bool)
        if hasattr(masks, "masks"):
            masks = masks.masks
        elif hasattr(masks, "to_tensor"):
            masks = masks.to_tensor(dtype=torch.bool, device="cpu")
        if hasattr(masks, "detach"):
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None, ...]
        return masks.astype(bool)

    @staticmethod
    def _mask_iou(mask_a, mask_b):
        h = min(mask_a.shape[0], mask_b.shape[0])
        w = min(mask_a.shape[1], mask_b.shape[1])
        a = mask_a[:h, :w].astype(bool)
        b = mask_b[:h, :w].astype(bool)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    @staticmethod
    def _bbox_from_mask(mask):
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return 0, 0, 0, 0
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @staticmethod
    def _color(label):
        colors = [
            (56, 179, 255),
            (86, 220, 120),
            (255, 168, 64),
            (220, 96, 255),
        ]
        return colors[label % len(colors)]


def train_one_epoch(model, loader, optimizer, device, scaler, grad_clip=1.0,
                    epoch=1, total_epochs=1, use_amp=True):
    model.train()
    total_loss = 0.0
    n_batches = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs}",
        dynamic_ncols=True,
        leave=False,
    )
    for images, data_samples in progress:
        batch_inputs = torch.stack(images).to(device)
        data_samples = [ds.to(device) for ds in data_samples]

        optimizer.zero_grad()
        amp_enabled = bool(use_amp and device.type == "cuda")
        with autocast(device_type=device.type, enabled=amp_enabled):
            loss_dict = model(batch_inputs, data_samples, mode="loss")
            losses = sum(v for k, v in loss_dict.items()
                         if "loss" in k and isinstance(v, torch.Tensor))

        if not torch.isfinite(losses):
            print(f"WARNING: non-finite loss {losses.item()}, skipping batch")
            optimizer.zero_grad()
            continue

        scaler.scale(losses).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()
        n_batches += 1
        avg_loss = total_loss / n_batches
        progress.set_postfix({
            "loss": f"{losses.item():.4f}",
            "avg": f"{avg_loss:.4f}",
            "lr": f"{optimizer.param_groups[-1]['lr']:.2e}",
        })

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, val_loader, device, score_thr=0.05, coco_verbose=False):
    model.eval()

    gt_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class{i}"} for i in range(1, 5)],
    }
    predictions = []
    ann_id = 1

    for images, data_samples in val_loader:
        batch_inputs = torch.stack(images).to(device)
        data_samples = [ds.to(device) for ds in data_samples]

        results = model(batch_inputs, data_samples, mode="predict")

        for ds, res in zip(data_samples, results):
            img_id = ds.metainfo.get("img_id", 0)
            h, w = ds.metainfo["ori_shape"]
            gt_dict["images"].append({"id": img_id, "height": h, "width": w})

            gt = ds.gt_instances
            gt_masks_bm = gt.masks
            gt_labels = gt.labels.cpu().numpy()
            for i in range(len(gt_labels)):
                m = gt_masks_bm.masks[i]
                rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                gt_dict["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(gt_labels[i]) + 1,
                    "segmentation": rle,
                    "area": int(m.sum()),
                    "bbox": mask_utils.toBbox(
                        mask_utils.encode(
                            np.asfortranarray(
                                m.astype(np.uint8)))
                    ).tolist(),
                    "iscrowd": 0,
                })
                ann_id += 1

            pred = res.pred_instances
            pred_scores = pred.scores.cpu().numpy()
            pred_labels = pred.labels.cpu().numpy()
            pred_masks = (
                pred.masks.cpu().numpy()
                if hasattr(pred, "masks")
                else np.zeros((0, h, w), dtype=np.uint8)
            )

            for i in range(len(pred_scores)):
                if pred_scores[i] < score_thr:
                    continue
                binary = pred_masks[i].astype(np.uint8)
                if binary.sum() < 1:
                    continue
                rle = mask_utils.encode(np.asfortranarray(binary))
                rle["counts"] = rle["counts"].decode("utf-8")
                predictions.append({
                    "image_id": img_id,
                    "category_id": int(pred_labels[i]) + 1,
                    "segmentation": rle,
                    "score": float(pred_scores[i]),
                })

    if len(predictions) == 0 or len(gt_dict["annotations"]) == 0:
        return 0.0

    output_context = (
        contextlib.nullcontext()
        if coco_verbose
        else contextlib.redirect_stdout(io.StringIO())
    )
    with output_context:
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return float(coco_eval.stats[1])


def load_config(path):
    if not os.path.exists(path) and not os.path.isabs(path):
        hw3_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(hw3_root, path)
    with open(path) as f:
        return yaml.safe_load(f)


def _hw3_path(path):
    if path is None or os.path.isabs(path):
        return path
    hw3_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(hw3_root, path)


def build_runner_config(args):
    import dataset  # noqa: F401 - registers LoadTifImageFromFile
    from dataset import get_train_pipeline, get_val_pipeline

    data_root = _hw3_path(args.data_dir)
    ann_dir = (_hw3_path(args.ann_dir) if args.ann_dir
               else os.path.join(data_root, "annotations"))
    img_scale = args.img_scale
    multiscale = bool(args.multiscale)
    use_amp = bool(args.amp)

    model_cfg = detector_blueprint(vars(args))
    train_ann = os.path.join(ann_dir, "train.json")
    val_ann = os.path.join(ann_dir, "val.json")

    metainfo = dict(classes=("class1", "class2", "class3", "class4"))
    train_dataloader = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        sampler=dict(type="DefaultSampler", shuffle=True),
        batch_sampler=dict(type="AspectRatioBatchSampler"),
        dataset=dict(
            type="CocoDataset",
            data_root=data_root,
            ann_file=train_ann,
            data_prefix=dict(img=""),
            filter_cfg=dict(filter_empty_gt=True, min_size=args.min_size),
            pipeline=get_train_pipeline(img_scale, multiscale=multiscale),
            metainfo=metainfo,
        ),
    )
    val_dataloader = dict(
        batch_size=1,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dict(
            type="CocoDataset",
            data_root=data_root,
            ann_file=val_ann,
            data_prefix=dict(img=""),
            test_mode=True,
            pipeline=get_val_pipeline(img_scale),
            metainfo=metainfo,
        ),
    )

    backbone_lr_mult = args.backbone_lr / args.lr if args.lr > 0 else 0.1
    min_lr_ratio = args.min_lr / args.lr if args.lr > 0 else 0.01
    clip_grad = None
    if args.grad_clip and args.grad_clip > 0:
        clip_grad = dict(max_norm=args.grad_clip, norm_type=2)

    optim_wrapper = dict(
        type="AmpOptimWrapper" if use_amp else "OptimWrapper",
        optimizer=dict(
            type="AdamW",
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        ),
        clip_grad=clip_grad,
        paramwise_cfg=dict(custom_keys={
            "backbone": dict(lr_mult=backbone_lr_mult),
            "norm": dict(decay_mult=0.0),
            "bias": dict(decay_mult=0.0),
        }),
    )

    warmup_end = min(args.warmup_epochs, max(args.epochs - 1, 1))
    param_scheduler = [
        dict(
            type="LinearLR",
            start_factor=0.001,
            by_epoch=True,
            begin=0,
            end=warmup_end,
        ),
        dict(
            type="CosineAnnealingLR",
            eta_min=args.lr * min_lr_ratio,
            by_epoch=True,
            begin=warmup_end,
            end=args.epochs,
        ),
    ]

    work_dir = _hw3_path(args.work_dir)
    run_start_time = args.run_start_time
    vis_backends = [dict(type="LocalVisBackend")]
    if args.wandb_project and args.wandb_project.lower() != "none":
        vis_backends.append(dict(
            type="WandbVisBackend",
            init_kwargs=dict(
                project=args.wandb_project,
                name=f"{args.backbone_name}_ep{args.epochs}",
            ),
        ))

    return dict(
        model=model_cfg,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        val_evaluator=dict(
            type="QuietCocoMetric",
            ann_file=val_ann,
            metric="segm",
            metric_items=["mAP_50"],
            format_only=False,
        ),
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        train_cfg=dict(
            type="EpochBasedTrainLoop",
            max_epochs=args.epochs,
            val_interval=args.val_interval,
        ),
        val_cfg=dict(type="ValLoop"),
        default_hooks=dict(
            timer=dict(type="IterTimerHook"),
            logger=dict(
                type="LoggerHook",
                interval=args.log_interval,
                ignore_last=True,
                interval_exp_name=1_000_000_000,
            ),
            param_scheduler=dict(type="ParamSchedulerHook"),
            checkpoint=dict(
                type="CheckpointHook",
                interval=args.save_interval,
                save_best="coco/segm_mAP_50",
                rule="greater",
                max_keep_ckpts=3,
                filename_tmpl=f"epoch_{{}}_{run_start_time}.pth",
            ),
            sampler_seed=dict(type="DistSamplerSeedHook"),
        ),
        custom_hooks=[
            dict(type="CompactTrainLogHook"),
            dict(
                type="TrainingArtifactsHook",
                num_classes=args.num_classes,
                score_thr=args.artifact_score_threshold,
                mask_iou_thr=args.artifact_mask_iou_threshold,
                num_visualizations=args.artifact_num_visualizations,
                seed=args.artifact_seed,
                draw_score_thr=args.artifact_draw_score_threshold,
                interval=args.artifact_interval,
                out_dir_name=args.artifact_dir_name,
            ),
        ] if args.training_artifacts else [dict(type="CompactTrainLogHook")],
        visualizer=dict(
            type="DetLocalVisualizer",
            vis_backends=vis_backends,
            name="visualizer",
        ),
        log_processor=dict(type="LogProcessor", window_size=50, by_epoch=True),
        work_dir=work_dir,
        env_cfg=dict(
            cudnn_benchmark=False,
            mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
            dist_cfg=dict(backend="nccl"),
        ),
        default_scope="mmdet",
        log_level=args.log_level,
        randomness=dict(seed=args.seed),
        load_from=args.resume or args.load_weight,
        resume=bool(args.resume),
    )


def train_with_runner(args):
    from mmengine.config import Config
    from mmengine.runner import Runner

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"mmdet\.models\.roi_heads\.mask_heads\.fcn_mask_head",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"To copy construct from a tensor.*",
        category=UserWarning,
    )

    if args.device and str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {args.device}, but CUDA is not "
                "available."
            )
        device_text = str(args.device)
        device_idx = 0
        if ":" in device_text:
            device_idx = int(device_text.split(":", 1)[1])
        torch.cuda.set_device(device_idx)
        gpu_name = torch.cuda.get_device_name(device_idx)
        print(f"Using device: cuda:{device_idx} ({gpu_name})")
    elif args.device:
        print(f"Using device: {args.device}")

    cfg = Config(build_runner_config(args))
    runner = Runner.from_cfg(cfg)
    model = runner.model
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"Model: total={total / 1e6:.2f}M, "
          f"trainable={trainable / 1e6:.2f}M")
    assert trainable < 200_000_000, (
        f"Trainable params exceed 200M: {trainable:,}")
    runner.train()


def make_training_parser():
    parser = argparse.ArgumentParser(description="Train Cascade Mask R-CNN")
    parser.add_argument("--config", default="configs/cascade_mask_rcnn.yaml")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--ann_dir", default=None)
    parser.add_argument("--work_dir", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--val_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--log_level", default=None)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument("--mask_threshold", type=float, default=None)
    parser.add_argument("--multiscale", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--run_start_time", default=None)
    parser.add_argument(
        "--coco_verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use CUDA automatic mixed precision.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--load_weight", default=None)
    return parser


def hydrate_training_args(args):
    cfg = load_config(args.config)
    for key, val in vars(args).items():
        if key == "config":
            continue
        if val is None and key in cfg:
            setattr(args, key, cfg[key])

    populate_runtime_args(args, cfg, TRAIN_FALLBACKS)

    if args.run_start_time is None:
        args.run_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args


def main():
    args = hydrate_training_args(make_training_parser().parse_args())
    cfg_str = json.dumps(dict(vars(args)), indent=2, default=str)
    print(f"Config: {cfg_str}")
    train_with_runner(args)


if __name__ == "__main__":
    main()
