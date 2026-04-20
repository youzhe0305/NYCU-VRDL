"""COCO-format dataset for digit detection (HW2).

Supports configurable spatial and pixel augmentations via aug_cfg dict
(loaded from the YAML config).  When aug_cfg is None the legacy behaviour
(ColorJitter + Sharpness) is preserved.
"""

import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as tv_transforms
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN_RGB = (124, 116, 104)  # approx IMAGENET_MEAN * 255, for canvas fill


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

def _resize(img, max_side=640, min_side=480):
    """Resize image so shorter side >= min_side and longer side <= max_side."""
    w, h = img.size
    scale = min_side / min(w, h)
    if scale * max(w, h) > max_side:
        scale = max_side / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BICUBIC), scale


# ---------------------------------------------------------------------------
# Bounding-box helpers (absolute xyxy format)
# ---------------------------------------------------------------------------

def _clip_boxes(boxes, w, h):
    """Clip boxes to [0, w] x [0, h].  boxes: Nx4 numpy [x1,y1,x2,y2]."""
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
    return boxes


def _boxes_area(boxes):
    return (
        np.maximum(boxes[:, 2] - boxes[:, 0], 0)
        * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    )


# ---------------------------------------------------------------------------
# Spatial augmentations (modify image + bboxes)
# ---------------------------------------------------------------------------

def aug_translate(img, boxes, labels, max_shift=0.1, min_area_ratio=0.25):
    """Random translation.  Clip bboxes; drop if remaining area < threshold."""
    if len(boxes) == 0:
        return img, boxes, labels

    w, h = img.size
    dx = int(round(random.uniform(-max_shift, max_shift) * w))
    dy = int(round(random.uniform(-max_shift, max_shift) * h))
    if dx == 0 and dy == 0:
        return img, boxes, labels

    new_boxes = boxes.copy()
    new_boxes[:, [0, 2]] += dx
    new_boxes[:, [1, 3]] += dy
    orig_areas = _boxes_area(boxes)
    clipped = _clip_boxes(new_boxes, w, h)
    new_areas = _boxes_area(clipped)

    keep = (new_areas > min_area_ratio * orig_areas) & (new_areas > 1.0)
    if keep.sum() == 0:
        return img, boxes, labels  # abort — all boxes lost

    canvas = Image.new("RGB", (w, h), IMAGENET_MEAN_RGB)
    canvas.paste(img, (dx, dy))
    return canvas, clipped[keep], labels[keep]


def aug_random_crop(img, boxes, labels, min_scale=0.5, max_scale=1.0,
                    min_overlap=0.5, max_attempts=50):
    """Random crop.  Keep bboxes whose overlap with crop >= min_overlap."""
    if len(boxes) == 0:
        return img, boxes, labels

    w, h = img.size
    for _ in range(max_attempts):
        # Use same scale for both dimensions to preserve aspect ratio
        s = random.uniform(min_scale, max_scale)
        cw = s * w
        ch = s * h

        cx1 = random.uniform(0, max(w - cw, 0))
        cy1 = random.uniform(0, max(h - ch, 0))
        cx2 = cx1 + cw
        cy2 = cy1 + ch

        # Overlap = intersection / box_area
        inter_x1 = np.maximum(boxes[:, 0], cx1)
        inter_y1 = np.maximum(boxes[:, 1], cy1)
        inter_x2 = np.minimum(boxes[:, 2], cx2)
        inter_y2 = np.minimum(boxes[:, 3], cy2)
        inter_area = (
            np.maximum(0, inter_x2 - inter_x1) *
            np.maximum(0, inter_y2 - inter_y1)
        )
        box_area = _boxes_area(boxes)
        overlap = inter_area / (box_area + 1e-6)

        keep = overlap >= min_overlap
        if keep.sum() == 0:
            continue

        ix1, iy1 = int(round(cx1)), int(round(cy1))
        ix2, iy2 = int(round(cx2)), int(round(cy2))
        ix2 = max(ix2, ix1 + 1)
        iy2 = max(iy2, iy1 + 1)
        cropped = img.crop((ix1, iy1, ix2, iy2))

        new_boxes = boxes[keep].copy()
        new_boxes[:, [0, 2]] -= ix1
        new_boxes[:, [1, 3]] -= iy1
        cw_int, ch_int = ix2 - ix1, iy2 - iy1
        new_boxes = _clip_boxes(new_boxes, cw_int, ch_int)

        # Drop boxes that became degenerate after clip
        areas = _boxes_area(new_boxes)
        valid = areas > 1.0
        if valid.sum() == 0:
            continue

        return cropped, new_boxes[valid], labels[keep][valid]

    return img, boxes, labels  # all attempts failed


def aug_random_expand(img, boxes, labels, max_ratio=2.0):
    """Zoom-out: paste image on a larger canvas (good for small objects)."""
    if len(boxes) == 0:
        return img, boxes, labels

    w, h = img.size
    ratio = random.uniform(1.0, max_ratio)
    new_w, new_h = int(w * ratio), int(h * ratio)

    canvas = Image.new("RGB", (new_w, new_h), IMAGENET_MEAN_RGB)
    left = random.randint(0, new_w - w)
    top = random.randint(0, new_h - h)
    canvas.paste(img, (left, top))

    new_boxes = boxes.copy()
    new_boxes[:, [0, 2]] += left
    new_boxes[:, [1, 3]] += top
    return canvas, new_boxes, labels.copy()


# ---------------------------------------------------------------------------
# Pixel augmentations (image only)
# ---------------------------------------------------------------------------

def aug_gaussian_blur(img, sigma_min=0.1, sigma_max=2.0):
    sigma = random.uniform(sigma_min, sigma_max)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def aug_iso_noise(img, intensity=0.05):
    """Additive Gaussian noise simulating camera ISO noise."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.randn(*arr.shape) * (intensity * 255)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_base_transforms():
    """ToTensor + Normalize only (used when aug_cfg controls augmentations)."""
    return tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transforms():
    """Legacy training transforms (used when aug_cfg is None)."""
    return tv_transforms.Compose([
        tv_transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        ),
        tv_transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    return tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class COCODetectionDataset(Dataset):
    """Dataset for COCO-format object detection annotations.

    When *aug_cfg* is provided, all augmentations (including ColorJitter and
    Sharpness) are controlled by the config dict.  The ``transforms`` argument
    should then be ``get_base_transforms()`` (ToTensor + Normalize only).

    When *aug_cfg* is ``None`` (val / test, or legacy training), augmentations
    are handled entirely by ``transforms``.
    """

    def __init__(self, img_dir, ann_file, transforms=None, max_side=640,
                 min_side=480, train_scales=None, aug_cfg=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.max_side = max_side
        self.min_side = min_side
        self.train_scales = train_scales
        self.aug_cfg = aug_cfg

        with open(ann_file) as f:
            data = json.load(f)

        self.id_to_info = {img["id"]: img for img in data["images"]}
        self.ann_by_image = {}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            self.ann_by_image.setdefault(iid, []).append(ann)
        self.image_ids = list(self.id_to_info.keys())

        # Pre-build reusable torchvision transforms from config
        if aug_cfg is not None:
            if aug_cfg.get("aug_color_jitter", True):
                self._color_jitter = tv_transforms.ColorJitter(
                    brightness=aug_cfg.get("aug_color_jitter_brightness", 0.4),
                    contrast=aug_cfg.get("aug_color_jitter_contrast", 0.4),
                    saturation=aug_cfg.get("aug_color_jitter_saturation", 0.4),
                    hue=aug_cfg.get("aug_color_jitter_hue", 0.1),
                )
            else:
                self._color_jitter = None
        else:
            self._color_jitter = None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        info = self.id_to_info[img_id]

        img_path = os.path.join(self.img_dir, info["file_name"])
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Scale jitter
        min_side = (
            random.choice(self.train_scales)
            if self.train_scales
            else self.min_side
        )
        img, scale = _resize(img, self.max_side, min_side)

        # Build bboxes in absolute xyxy
        anns = self.ann_by_image.get(img_id, [])
        boxes_list, labels_list = [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes_list.append([
                x * scale, y * scale,
                (x + w) * scale, (y + h) * scale,
            ])
            labels_list.append(ann["category_id"])  # 1-indexed

        boxes = (
            np.array(boxes_list, dtype=np.float32).reshape(-1, 4)
            if boxes_list
            else np.zeros((0, 4), dtype=np.float32)
        )
        labels = (
            np.array(labels_list, dtype=np.int64)
            if labels_list
            else np.zeros(0, dtype=np.int64)
        )

        # ---- Augmentations (training only, controlled by aug_cfg) ----
        if self.aug_cfg is not None:
            c = self.aug_cfg

            # -- Spatial augmentations (image + bboxes) --
            if len(boxes) > 0:
                if (c.get("aug_random_expand")
                        and random.random() < c.get("aug_random_expand_p", 0.3)):
                    img, boxes, labels = aug_random_expand(
                        img, boxes, labels,
                        max_ratio=c.get("aug_random_expand_max_ratio", 2.0),
                    )

                if (c.get("aug_random_crop")
                        and random.random() < c.get("aug_random_crop_p", 0.5)):
                    img, boxes, labels = aug_random_crop(
                        img, boxes, labels,
                        min_scale=c.get("aug_random_crop_min_scale", 0.5),
                        max_scale=c.get("aug_random_crop_max_scale", 1.0),
                        min_overlap=c.get("aug_random_crop_min_overlap", 0.5),
                    )

                if (c.get("aug_translation")
                        and random.random() < c.get("aug_translation_p", 0.3)):
                    img, boxes, labels = aug_translate(
                        img, boxes, labels,
                        max_shift=c.get("aug_translation_max_shift", 0.1),
                        min_area_ratio=c.get("aug_translation_min_area_ratio", 0.25),
                    )

            # -- Re-resize to enforce max_side / min_side after spatial augs --
            img_resized, re_scale = _resize(img, self.max_side, min_side)
            if len(boxes) > 0 and abs(re_scale - 1.0) > 1e-6:
                boxes[:, :4] *= re_scale
            img = img_resized

            # -- Pixel augmentations (image only) --
            if self._color_jitter is not None:
                img = self._color_jitter(img)

            if (c.get("aug_sharpness")
                    and random.random() < c.get("aug_sharpness_p", 0.3)):
                factor = c.get("aug_sharpness_factor", 2.0)
                img = TF.adjust_sharpness(img, factor)

            if (c.get("aug_gaussian_blur")
                    and random.random() < c.get("aug_gaussian_blur_p", 0.2)):
                img = aug_gaussian_blur(
                    img,
                    sigma_min=c.get("aug_gaussian_blur_sigma_min", 0.1),
                    sigma_max=c.get("aug_gaussian_blur_sigma_max", 2.0),
                )

            if (c.get("aug_iso_noise")
                    and random.random() < c.get("aug_iso_noise_p", 0.2)):
                img = aug_iso_noise(
                    img,
                    intensity=c.get("aug_iso_noise_intensity", 0.05),
                )

        # ---- Convert xyxy → normalised cxcywh ----
        img_w, img_h = img.size
        valid_boxes, valid_labels = [], []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            bw, bh = x2 - x1, y2 - y1
            if bw > 0 and bh > 0:
                valid_boxes.append([
                    (x1 + x2) / 2 / img_w,
                    (y1 + y2) / 2 / img_h,
                    bw / img_w,
                    bh / img_h,
                ])
                valid_labels.append(labels[i])

        # ---- ToTensor + Normalize ----
        if self.transforms:
            img = self.transforms(img)

        target = {
            "image_id": img_id,
            "boxes": (
                torch.tensor(valid_boxes, dtype=torch.float32)
                if valid_boxes
                else torch.zeros((0, 4), dtype=torch.float32)
            ),
            "labels": (
                torch.tensor(valid_labels, dtype=torch.long)
                if valid_labels
                else torch.zeros(0, dtype=torch.long)
            ),
            "orig_size": torch.tensor([orig_h, orig_w]),
            "size": torch.tensor([img_h, img_w]),
        }
        return img, target


class TestDataset(Dataset):
    """Test dataset without annotations; image_id derived from filename."""

    def __init__(self, img_dir, transforms=None, max_side=640, min_side=480):
        self.img_dir = img_dir
        self.transforms = transforms
        self.max_side = max_side
        self.min_side = min_side
        self.files = sorted(
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        img, _ = _resize(img, self.max_side, self.min_side)
        new_w, new_h = img.size

        if self.transforms:
            img = self.transforms(img)

        image_id = int(os.path.splitext(fname)[0])
        return img, {
            "image_id": image_id,
            "orig_size": torch.tensor([orig_h, orig_w]),
            "size": torch.tensor([new_h, new_w]),
        }


def collate_fn(batch):
    """Pad images to the same spatial size within a batch."""
    images, targets = zip(*batch)

    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded = torch.zeros(len(images), 3, max_h, max_w)
    # mask: True = padding region (ignored)
    masks = torch.ones(len(images), max_h, max_w, dtype=torch.bool)

    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        padded[i, :, :h, :w] = img
        masks[i, :h, :w] = False

    return padded, masks, list(targets)


def build_dataloaders(
    data_dir, batch_size=4, num_workers=4, max_side=640, min_side=480,
    train_scales=None, aug_cfg=None,
):
    # When aug_cfg is given, color augmentations are applied inside __getitem__
    # so the transforms only need ToTensor + Normalize.
    if aug_cfg:
        train_transforms = get_base_transforms()
    else:
        train_transforms = get_train_transforms()  # legacy behaviour

    train_dataset = COCODetectionDataset(
        img_dir=os.path.join(data_dir, "train"),
        ann_file=os.path.join(data_dir, "train.json"),
        transforms=train_transforms,
        max_side=max_side,
        min_side=min_side,
        train_scales=train_scales,
        aug_cfg=aug_cfg,
    )
    val_dataset = COCODetectionDataset(
        img_dir=os.path.join(data_dir, "valid"),
        ann_file=os.path.join(data_dir, "valid.json"),
        transforms=get_val_transforms(),
        max_side=max_side,
        min_side=min_side,
    )
    test_dataset = TestDataset(
        img_dir=os.path.join(data_dir, "test"),
        transforms=get_val_transforms(),
        max_side=max_side,
        min_side=min_side,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
