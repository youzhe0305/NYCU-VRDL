import argparse
import json
import os

import cv2
import numpy as np
import tifffile
import yaml
from pycocotools import mask as mask_utils
from sklearn.model_selection import train_test_split


CATEGORIES = [
    {"id": 1, "name": "class1", "supercategory": "cell"},
    {"id": 2, "name": "class2", "supercategory": "cell"},
    {"id": 3, "name": "class3", "supercategory": "cell"},
    {"id": 4, "name": "class4", "supercategory": "cell"},
]


def encode_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.shape[0] >= 3:
            polygon = contour.reshape(-1).tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)
    return polygons


def collect_annotations(data_root, min_area):
    train_dir = os.path.join(data_root, "train")
    images = []
    annotations = []
    class_counts = []
    ann_id = 1
    img_id = 1

    for dirname in sorted(os.listdir(train_dir)):
        sample_dir = os.path.join(train_dir, dirname)
        img_path = os.path.join(sample_dir, "image.tif")
        if not os.path.isdir(sample_dir) or not os.path.exists(img_path):
            continue

        image = tifffile.imread(img_path)
        height, width = image.shape[:2]
        images.append({
            "id": img_id,
            "file_name": os.path.join("train", dirname, "image.tif"),
            "height": int(height),
            "width": int(width),
        })

        class_files = sorted(
            f for f in os.listdir(sample_dir)
            if f.startswith("class") and f.endswith(".tif"))
        class_counts.append(len(class_files))

        for class_file in class_files:
            class_id = int(class_file.replace("class", "").replace(".tif", ""))
            mask = tifffile.imread(os.path.join(sample_dir, class_file))
            if mask.ndim == 3:
                mask = mask[..., 0]

            for instance_id in np.unique(mask):
                if instance_id == 0:
                    continue
                binary = (mask == instance_id).astype(np.uint8)
                area = int(binary.sum())
                if area < min_area:
                    continue

                ys, xs = np.where(binary)
                if len(xs) == 0:
                    continue
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                bbox_w = x2 - x1 + 1
                bbox_h = y2 - y1 + 1
                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                segmentation = mask_to_polygons(binary)
                if not segmentation:
                    segmentation = encode_rle(binary)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": [x1, y1, bbox_w, bbox_h],
                    "iscrowd": 0,
                })
                ann_id += 1

        img_id += 1

    return images, annotations, np.asarray(class_counts)


def subset_coco(images, annotations, image_ids):
    image_ids = set(image_ids)
    return {
        "images": [img for img in images if img["id"] in image_ids],
        "annotations": [
            ann for ann in annotations if ann["image_id"] in image_ids],
        "categories": CATEGORIES,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare single COCO train/val split")
    parser.add_argument("--config", default=None,
                        help="Optional yaml config to read val_ratio from.")
    parser.add_argument("--data_root", default="../data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_area", type=int, default=10)
    return parser.parse_args()


def _default_config_path():
    hw3_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(hw3_root, "configs", "cascade_mask_rcnn.yaml")


def _load_val_ratio(args):
    if args.val_ratio is not None:
        return args.val_ratio

    config_path = args.config or _default_config_path()
    if not os.path.exists(config_path) and not os.path.isabs(config_path):
        hw3_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(hw3_root, config_path)
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return float(cfg.get("val_ratio", 0.2))
    return 0.2


def _stratified_train_val_split(image_ids, class_counts, val_ratio, seed):
    unique_classes = np.unique(class_counts)
    val_size = int(np.ceil(len(image_ids) * val_ratio))
    val_size = max(val_size, len(unique_classes))
    val_size = min(val_size, len(image_ids) - len(unique_classes))
    if val_size <= 0:
        raise ValueError(
            f"Not enough images ({len(image_ids)}) for a stratified split.")

    try:
        return train_test_split(
            image_ids,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
            stratify=class_counts,
        )
    except ValueError as err:
        print(f"Warning: stratified split failed ({err}); "
              "falling back to random split.")
        return train_test_split(
            image_ids,
            test_size=val_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )


def main():
    args = parse_args()
    val_ratio = _load_val_ratio(args)
    output_dir = args.output_dir or os.path.join(args.data_root, "annotations")
    os.makedirs(output_dir, exist_ok=True)

    images, annotations, class_counts = collect_annotations(
        args.data_root, args.min_area)
    image_ids = np.asarray([img["id"] for img in images])

    full = {"images": images, "annotations": annotations,
            "categories": CATEGORIES}
    with open(os.path.join(output_dir, "train_all.json"), "w") as f:
        json.dump(full, f)

    train_ids, val_ids = _stratified_train_val_split(
        image_ids, class_counts, val_ratio, args.seed)
    train_json = subset_coco(images, annotations, train_ids)
    val_json = subset_coco(images, annotations, val_ids)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_json, f)
    with open(os.path.join(output_dir, "val.json"), "w") as f:
        json.dump(val_json, f)
    print(f"single split: {len(train_json['images'])} train, "
          f"{len(val_json['images'])} val (val_ratio={val_ratio})")


if __name__ == "__main__":
    main()
