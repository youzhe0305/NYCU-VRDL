import numpy as np
import tifffile
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadTifImageFromFile(BaseTransform):
    def __init__(self, to_float32=False, color_type="color"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def transform(self, results):
        image = tifffile.imread(results["img_path"])
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] >= 4:
            image = image[:, :, :3]
        if image.ndim == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1]
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if self.to_float32:
            image = image.astype(np.float32)
        results["img"] = image
        results["img_shape"] = image.shape[:2]
        results["ori_shape"] = image.shape[:2]
        return results


def _scale_tuple(img_scale):
    if isinstance(img_scale, int):
        return (img_scale, img_scale)
    return tuple(img_scale)


def get_train_pipeline(img_scale=1024, multiscale=True):
    img_scale = _scale_tuple(img_scale)
    pipeline = [
        dict(type="LoadTifImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    ]
    if multiscale:
        pipeline.extend([
            dict(
                type="RandomResize",
                scale=img_scale,
                ratio_range=(0.5, 2.0),
                keep_ratio=True,
            ),
            dict(
                type="RandomCrop",
                crop_size=img_scale,
                crop_type="absolute",
                recompute_bbox=True,
                allow_negative_crop=True,
            ),
        ])
    else:
        pipeline.append(dict(type="Resize", scale=img_scale, keep_ratio=True))

    pipeline.extend([
        dict(type="RandomFlip", prob=0.5, direction="horizontal"),
        dict(type="RandomFlip", prob=0.5, direction="vertical"),
        dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
        dict(type="Pad", size=img_scale, pad_val=dict(img=(114, 114, 114))),
        dict(type="PackDetInputs"),
    ])
    return pipeline


def get_val_pipeline(img_scale=1024):
    img_scale = _scale_tuple(img_scale)
    return [
        dict(type="LoadTifImageFromFile"),
        dict(type="Resize", scale=img_scale, keep_ratio=True),
        dict(type="Pad", size=img_scale, pad_val=dict(img=(114, 114, 114))),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                       "scale_factor"),
        ),
    ]


def get_test_pipeline(img_scale=1024):
    img_scale = _scale_tuple(img_scale)
    return [
        dict(type="LoadTifImageFromFile"),
        dict(type="Resize", scale=img_scale, keep_ratio=True),
        dict(type="Pad", size=img_scale, pad_val=dict(img=(114, 114, 114))),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                       "scale_factor"),
        ),
    ]
