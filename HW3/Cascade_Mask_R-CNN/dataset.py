from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

from cell_io import image_size, read_tif_as_bgr


@TRANSFORMS.register_module(name="LoadTifImageFromFile", force=True)
class TiffFrameLoader(BaseTransform):
    def __init__(self, to_float32=False, color_type="color"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def transform(self, record):
        frame = read_tif_as_bgr(record["img_path"], float32=self.to_float32)
        record.update(
            img=frame,
            img_shape=frame.shape[:2],
            ori_shape=frame.shape[:2],
        )
        return record


def _loader():
    return dict(type="LoadTifImageFromFile")


def _pad_to_canvas(side, value=114):
    return dict(type="Pad", size=side, pad_val=dict(img=(value, value, value)))


def _pack(include_gt):
    meta = ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")
    if include_gt:
        return dict(type="PackDetInputs", meta_keys=meta)
    return dict(type="PackDetInputs", meta_keys=meta)


def _resize_stage(side, multiscale):
    if not multiscale:
        return [dict(type="Resize", scale=side, keep_ratio=True)]
    return [
        dict(
            type="RandomResize",
            scale=side,
            ratio_range=(0.5, 2.0),
            keep_ratio=True,
        ),
        dict(
            type="RandomCrop",
            crop_size=side,
            crop_type="absolute",
            recompute_bbox=True,
            allow_negative_crop=True,
        ),
    ]


def training_recipe(img_scale=1024, multiscale=True):
    side = image_size(img_scale)
    flow = [
        _loader(),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
        *_resize_stage(side, multiscale),
        dict(type="RandomFlip", prob=0.5, direction="horizontal"),
        dict(type="RandomFlip", prob=0.5, direction="vertical"),
        dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1)),
        _pad_to_canvas(side),
        dict(type="PackDetInputs"),
    ]
    return flow


def eval_recipe(img_scale=1024, *, annotations):
    side = image_size(img_scale)
    flow = [
        _loader(),
        dict(type="Resize", scale=side, keep_ratio=True),
        _pad_to_canvas(side),
    ]
    if annotations:
        flow.append(dict(type="LoadAnnotations", with_bbox=True, with_mask=True))
    flow.append(_pack(include_gt=annotations))
    return flow


def get_train_pipeline(img_scale=1024, multiscale=True):
    return training_recipe(img_scale, multiscale)


def get_val_pipeline(img_scale=1024):
    return eval_recipe(img_scale, annotations=True)


def get_test_pipeline(img_scale=1024):
    return eval_recipe(img_scale, annotations=False)
