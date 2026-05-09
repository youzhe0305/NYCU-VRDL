import os

import numpy as np
import tifffile
from pycocotools import mask as mask_utils


def hw3_relative_path(path):
    if path is None or os.path.isabs(path):
        return path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, path)


def image_size(value):
    if isinstance(value, int):
        return (value, value)
    return tuple(value)


def read_tif_as_bgr(path, *, float32=False):
    image = tifffile.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] > 3:
        image = image[..., :3]

    if image.ndim == 3 and image.shape[2] == 3:
        image = image[..., ::-1]

    image = np.ascontiguousarray(image, dtype=np.uint8)
    return image.astype(np.float32) if float32 else image


def coco_rle(binary_mask):
    packed = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    packed["counts"] = packed["counts"].decode("utf-8")
    return packed


def submission_rle(binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
    rle = coco_rle(binary_mask)
    return {
        "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
        "counts": rle["counts"],
    }


def cell_categories(n_classes=4):
    return [
        {"id": idx, "name": f"class{idx}", "supercategory": "cell"}
        for idx in range(1, n_classes + 1)
    ]
