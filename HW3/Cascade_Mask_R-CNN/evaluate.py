import argparse
import glob
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

from cell_io import hw3_relative_path, read_tif_as_bgr, submission_rle
from runtime_defaults import INFERENCE_FALLBACKS, populate_runtime_args


@dataclass(frozen=True)
class InferencePaths:
    data_dir: str
    test_dir: str
    mapping_file: str
    output_file: str


class ResultLedger:
    def __init__(self):
        self.rows = []

    def add(self, image_id, label, score, binary_mask):
        _record_prediction(self.rows, image_id, label, score, binary_mask)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.rows, handle)
        print(f"Saved {len(self.rows)} predictions to {path}")


def _record_prediction(results, image_id, label, score, binary_mask):
    binary_mask = binary_mask.astype(np.uint8)
    if binary_mask.sum() < 1:
        return
    results.append({
        "image_id": int(image_id),
        "category_id": int(label) + 1,
        "segmentation": submission_rle(binary_mask),
        "score": round(float(score), 6),
    })


def _hw3_path(path):
    return hw3_relative_path(path)


def _load_test_image(path):
    return read_tif_as_bgr(path)


def _build_detector(args, checkpoint, score_threshold):
    import dataset  # noqa: F401 - registers LoadTifImageFromFile
    import model as model_module  # noqa: F401 - registers backbone
    from dataset import get_test_pipeline
    from mmdet.apis import init_detector
    from mmengine.config import Config
    from model import detector_blueprint

    cfg_dict = vars(args).copy()
    cfg_dict["pretrained"] = False
    cfg_dict["score_threshold"] = score_threshold
    model_cfg = detector_blueprint(cfg_dict)
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


def _read_mapping(path):
    with open(path) as handle:
        return {
            item["file_name"]: item
            for item in json.load(handle)
        }


def _inference_paths(args):
    data_dir = _hw3_path(args.data_dir)
    test_dir = (_hw3_path(args.test_dir) if args.test_dir
                else os.path.join(data_dir, "test_release"))
    mapping_file = (_hw3_path(args.id_mapping) if args.id_mapping
                    else os.path.join(data_dir,
                                      "test_image_name_to_ids.json"))
    return InferencePaths(data_dir, test_dir, mapping_file, args.output)


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


def _single_pass_instances(detector, image, score_floor):
    from mmdet.apis import inference_detector

    height, width = image.shape[:2]
    pred = inference_detector(detector, image).pred_instances
    labels = pred.labels.cpu().numpy().astype(np.int64)
    scores = pred.scores.cpu().numpy().astype(np.float32)
    masks = (
        pred.masks.cpu().numpy().astype(np.uint8)
        if hasattr(pred, "masks") and pred.masks is not None
        else np.zeros((0, height, width), dtype=np.uint8)
    )
    masks = _resize_masks(masks, height, width)
    keep = scores >= score_floor
    return labels[keep], scores[keep], masks[keep]


def _find_checkpoint(args):
    work_dir = _hw3_path(args.work_dir)
    candidates = sorted(glob.glob(os.path.join(
        work_dir, "best_coco_segm_mAP_50*.pth")))
    if candidates:
        return candidates[-1]
    legacy_candidates = sorted(glob.glob(os.path.join(
        _hw3_path(args.save_dir), "best_model*.pth")))
    return legacy_candidates[-1] if legacy_candidates else None


class SubmissionJob:
    def __init__(self, args):
        self.args = args
        self.paths = _inference_paths(args)
        self.name_to_info = _read_mapping(self.paths.mapping_file)
        checkpoint = args.checkpoint or _find_checkpoint(args)
        if not checkpoint:
            raise FileNotFoundError("No checkpoint found for inference.")
        self.detector = _build_detector(
            args, checkpoint, args.candidate_score_threshold)

    def tif_names(self):
        return sorted(
            name for name in os.listdir(self.paths.test_dir)
            if name.endswith(".tif")
        )

    def emit_image_rows(self, fname, ledger):
        info = self.name_to_info.get(fname)
        if info is None:
            return

        image = _load_test_image(os.path.join(self.paths.test_dir, fname))
        labels, scores, masks = _single_pass_instances(
            self.detector, image, self.args.candidate_score_threshold)
        for label, score, mask in zip(labels, scores, masks):
            if score >= self.args.score_threshold:
                ledger.add(info["id"], label, score, mask)

    def run(self):
        ledger = ResultLedger()
        for fname in tqdm(self.tif_names(), desc="Inference"):
            self.emit_image_rows(fname, ledger)
        ledger.save(self.paths.output_file)


def run_mmdet_inference(args):
    SubmissionJob(args).run()


def make_inference_parser():
    parser = argparse.ArgumentParser(
        description="Run Cascade Mask R-CNN inference")
    parser.add_argument("--config", default="configs/cascade_mask_rcnn.yaml")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint path. If omitted, the best checkpoint "
             "in work_dir is used.",
    )
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--test_dir", default=None)
    parser.add_argument("--id_mapping", default=None)
    parser.add_argument("--work_dir", default=None)
    parser.add_argument("--output", default="test-results-cascade.json")
    parser.add_argument("--mode", default="test",
                        choices=["test"], help=argparse.SUPPRESS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument(
        "--candidate_score_threshold", type=float, default=None,
    )
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
    return parser


def hydrate_inference_args(args):
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

    populate_runtime_args(args, cfg, INFERENCE_FALLBACKS)
    return args


def main():
    args = hydrate_inference_args(make_inference_parser().parse_args())
    run_mmdet_inference(args)


if __name__ == "__main__":
    main()
