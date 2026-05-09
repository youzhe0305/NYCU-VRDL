import argparse
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import tifffile
import yaml
from sklearn.model_selection import train_test_split

from cell_io import cell_categories, coco_rle, hw3_relative_path


@dataclass
class CocoTables:
    images: list
    annotations: list
    class_counts: np.ndarray

    def materialize(self, ids=None):
        allowed = None if ids is None else set(int(x) for x in ids)
        images = self.images
        annotations = self.annotations
        if allowed is not None:
            images = [row for row in images if row["id"] in allowed]
            annotations = [
                row for row in annotations if row["image_id"] in allowed
            ]
        return {
            "images": images,
            "annotations": annotations,
            "categories": cell_categories(),
        }


def _instance_polygons(binary):
    contours, _ = cv2.findContours(
        binary.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return [
        contour.reshape(-1).tolist()
        for contour in contours
        if contour.shape[0] >= 3 and contour.size >= 6
    ]


def _mask_rows(mask, image_id, category_id, first_ann_id, min_area):
    mask = mask[..., 0] if mask.ndim == 3 else mask
    next_id = first_ann_id
    rows = []
    for instance_value in np.unique(mask):
        if instance_value == 0:
            continue

        binary = (mask == instance_value).astype(np.uint8)
        area = int(binary.sum())
        if area < min_area:
            continue

        ys, xs = np.where(binary)
        if xs.size == 0:
            continue

        left, top = int(xs.min()), int(ys.min())
        width = int(xs.max()) - left + 1
        height = int(ys.max()) - top + 1
        if width <= 0 or height <= 0:
            continue

        rows.append({
            "id": next_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": _instance_polygons(binary) or coco_rle(binary),
            "area": area,
            "bbox": [left, top, width, height],
            "iscrowd": 0,
        })
        next_id += 1
    return rows, next_id


def _sample_dirs(train_root):
    for name in sorted(os.listdir(train_root)):
        directory = os.path.join(train_root, name)
        image_path = os.path.join(directory, "image.tif")
        if os.path.isdir(directory) and os.path.exists(image_path):
            yield name, directory, image_path


def scan_training_set(data_root, min_area):
    images, annotations, class_counts = [], [], []
    ann_id = 1

    for image_id, (name, directory, image_path) in enumerate(
        _sample_dirs(os.path.join(data_root, "train")),
        start=1,
    ):
        height, width = tifffile.imread(image_path).shape[:2]
        images.append({
            "id": image_id,
            "file_name": os.path.join("train", name, "image.tif"),
            "height": int(height),
            "width": int(width),
        })

        class_files = sorted(
            fname for fname in os.listdir(directory)
            if fname.startswith("class") and fname.endswith(".tif")
        )
        class_counts.append(len(class_files))

        for class_file in class_files:
            label_id = int(class_file[5:-4])
            mask = tifffile.imread(os.path.join(directory, class_file))
            rows, ann_id = _mask_rows(
                mask,
                image_id=image_id,
                category_id=label_id,
                first_ann_id=ann_id,
                min_area=min_area,
            )
            annotations.extend(rows)

    return CocoTables(images, annotations, np.asarray(class_counts))


def _default_yaml():
    return hw3_relative_path("configs/cascade_mask_rcnn.yaml")


def _validation_ratio(args):
    if args.val_ratio is not None:
        return float(args.val_ratio)

    config_path = hw3_relative_path(args.config or _default_yaml())
    if os.path.exists(config_path):
        with open(config_path) as handle:
            return float((yaml.safe_load(handle) or {}).get("val_ratio", 0.2))
    return 0.2


def _split_once(image_ids, class_counts, ratio, seed):
    distinct_bins = np.unique(class_counts)
    val_count = int(np.ceil(len(image_ids) * ratio))
    val_count = max(val_count, len(distinct_bins))
    val_count = min(val_count, len(image_ids) - len(distinct_bins))
    if val_count <= 0:
        raise ValueError(
            f"Need more than {len(image_ids)} images for a stratified split."
        )

    try:
        return train_test_split(
            image_ids,
            test_size=val_count,
            random_state=seed,
            shuffle=True,
            stratify=class_counts,
        )
    except ValueError as exc:
        print(f"Warning: using random split because stratification failed: {exc}")
        return train_test_split(
            image_ids,
            test_size=val_count,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare one COCO train/val split")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_root", default="../data")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_area", type=int, default=10)
    return parser.parse_args()


def _write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle)


def main():
    args = parse_args()
    ratio = _validation_ratio(args)
    out_dir = args.output_dir or os.path.join(args.data_root, "annotations")
    os.makedirs(out_dir, exist_ok=True)

    tables = scan_training_set(args.data_root, args.min_area)
    image_ids = np.asarray([item["id"] for item in tables.images])
    train_ids, val_ids = _split_once(
        image_ids, tables.class_counts, ratio, args.seed)

    _write_json(os.path.join(out_dir, "train_all.json"), tables.materialize())
    train_json = tables.materialize(train_ids)
    val_json = tables.materialize(val_ids)
    _write_json(os.path.join(out_dir, "train.json"), train_json)
    _write_json(os.path.join(out_dir, "val.json"), val_json)

    print(
        f"single split: {len(train_json['images'])} train, "
        f"{len(val_json['images'])} val (val_ratio={ratio})"
    )


if __name__ == "__main__":
    main()
