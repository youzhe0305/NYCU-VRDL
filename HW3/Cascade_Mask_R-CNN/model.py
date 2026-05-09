import os

import mmdet.models  # noqa: F401 — trigger module registration
import timm
import torch
import torch.nn as nn
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.roi_heads.roi_extractors import GenericRoIExtractor
from mmdet.registry import MODELS
from mmengine.config import ConfigDict
from mmengine.registry import DefaultScope

from architecture import ModelRecipe, detector_blueprint


@MODELS.register_module(name="ConvNeXtV2Backbone", force=True)
class TimmStagePyramid(nn.Module):

    def __init__(self, model_name="convnextv2_base",
                 pretrained=True, drop_path_rate=0.4,
                 out_indices=(0, 1, 2, 3), freeze_stages=0,
                 img_size=None, norm_type="LayerNorm"):
        super().__init__()
        create_kwargs = dict(
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=drop_path_rate,
        )
        if img_size is not None:
            create_kwargs["img_size"] = img_size
        self.body = timm.create_model(
            model_name,
            **create_kwargs,
        )
        self._out_channels = self.body.feature_info.channels()
        self.norm_type = norm_type
        if norm_type == "BatchNorm2d":
            self.norms = nn.ModuleList([
                nn.BatchNorm2d(ch) for ch in self._out_channels
            ])
        elif norm_type == "Identity":
            self.norms = nn.ModuleList([
                nn.Identity() for _ in self._out_channels
            ])
        else:
            self.norms = nn.ModuleList([
                nn.LayerNorm(ch) for ch in self._out_channels
            ])
        self.out_indices = out_indices
        self.freeze_stages = freeze_stages
        self._freeze_stages()

    def _freeze_stages(self):
        if self.freeze_stages <= 0:
            return
        for name, param in self.named_parameters():
            freeze = name.startswith("body.stem")
            for s in range(self.freeze_stages):
                if (name.startswith(f"body.stages.{s}")
                        or name.startswith(f"body.stages_{s}")
                        or name.startswith(f"norms.{s}")):
                    freeze = True
            if freeze:
                param.requires_grad = False

    def forward(self, x):
        features = self.body(x)
        outs = []
        for i, feat in enumerate(features):
            if i in self.out_indices:
                channels = self._out_channels[i]
                is_nhwc = feat.shape[-1] == channels
                if self.norm_type == "BatchNorm2d":
                    if is_nhwc:
                        feat = feat.permute(0, 3, 1, 2).contiguous()
                    feat = self.norms[i](feat)
                elif is_nhwc:
                    feat = self.norms[i](feat)
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                else:
                    feat = feat.permute(0, 2, 3, 1)
                    feat = self.norms[i](feat)
                    feat = feat.permute(0, 3, 1, 2).contiguous()
                outs.append(feat)
        return tuple(outs)


@MODELS.register_module(name="AmpSafeGenericRoIExtractor", force=True)
class PrecisionAlignedRoIExtractor(GenericRoIExtractor):
    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) > 0 and rois.dtype != feats[0].dtype:
            rois = rois.to(feats[0].dtype)
        return super().forward(feats, rois, roi_scale_factor)


@MODELS.register_module(name="DiceFCNMaskHead", force=True)
class HybridDiceMaskHead(FCNMaskHead):
    def __init__(self, dice_loss_weight=1.0, ce_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.ce_loss_weight = ce_loss_weight
        self.loss_dice = MODELS.build(dict(
            type="DiceLoss",
            use_sigmoid=True,
            activate=True,
            loss_weight=dice_loss_weight,
        ))

    def loss_and_target(self, mask_preds, sampling_results,
                        batch_gt_instances, rcnn_train_cfg):
        mask_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])

        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
            loss_dice = mask_preds.sum()
        else:
            if self.class_agnostic:
                selected_preds = mask_preds[:, 0]
                ce_labels = torch.zeros_like(pos_labels)
            else:
                selected_preds = mask_preds[
                    torch.arange(mask_preds.size(0), device=mask_preds.device),
                    pos_labels]
                ce_labels = pos_labels
            loss_mask = self.loss_mask(mask_preds, mask_targets, ce_labels)
            if self.ce_loss_weight == 0:
                loss_mask = loss_mask * 0
            loss_dice = self.loss_dice(selected_preds, mask_targets.float())
        return dict(
            loss_mask=dict(loss_mask=loss_mask, loss_dice=loss_dice),
            mask_targets=mask_targets,
        )


def _build_model_cfg(cfg):
    return detector_blueprint(cfg)


def _to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: _to_config_dict(v) for k, v in d.items()})
    if isinstance(d, (list, tuple)):
        return type(d)(_to_config_dict(v) for v in d)
    return d


def build_model(cfg):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ["TORCH_HOME"] = os.path.join(project_root, "pretrained_weight")

    DefaultScope.get_instance("mmdet_scope", scope_name="mmdet")

    model_cfg = _to_config_dict(ModelRecipe(cfg).export())
    model = MODELS.build(model_cfg)

    freeze_stages = cfg.get("freeze_stages", 0)
    if freeze_stages > 0:
        for name, param in model.backbone.named_parameters():
            freeze = name.startswith("body.stem")
            for s in range(freeze_stages):
                if (name.startswith(f"body.stages.{s}")
                        or name.startswith(f"body.stages_{s}")
                        or name.startswith(f"norms.{s}")):
                    freeze = True
            if freeze:
                param.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {total - trainable:,}")
    assert trainable < 200_000_000, (
        f"Model has {trainable:,} trainable params, exceeding 200M limit!")
    return model


def get_optimizer_params(model, cfg):
    backbone_lr = cfg.get("backbone_lr", 1e-5)
    head_lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 0.05)

    no_decay_keywords = ("norm", "bias")
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = "backbone" in name
        no_wd = any(k in name.lower() for k in no_decay_keywords)
        param_groups.append({
            "params": [param],
            "lr": backbone_lr if is_backbone else head_lr,
            "weight_decay": 0.0 if no_wd else weight_decay,
        })
    return param_groups
