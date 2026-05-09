class ModelRecipe:
    def __init__(self, settings):
        self.settings = settings

    def export(self):
        return detector_blueprint(self.settings)


def detector_blueprint(cfg):
    num_classes = cfg.get("num_classes", 4)
    fpn_ch = cfg.get("fpn_out_channels", 256)
    neck_type = cfg.get("neck_type", "FPN")
    fpn_num_outs = cfg.get("fpn_num_outs", 5)
    rpn_strides = cfg.get("rpn_strides", [4, 8, 16, 32, 64])
    roi_featmap_strides = cfg.get("roi_featmap_strides", [4, 8, 16, 32])
    rpn_head_type = cfg.get("rpn_head_type", "RPNHead")
    backbone_type = cfg.get("backbone_type", "ConvNeXtV2Backbone")
    backbone_name = cfg.get("backbone_name", "convnextv2_base")
    drop_path_rate = cfg.get("drop_path_rate", 0.4)

    if cfg.get("backbone_out_channels"):
        in_channels = cfg.get("backbone_out_channels")
    elif backbone_type == "ResNet":
        in_channels = [256, 512, 1024, 2048]
    elif backbone_name.startswith("convnextv2_large"):
        in_channels = [192, 384, 768, 1536]
    elif backbone_name.startswith(("convnextv2_tiny", "convnextv2_small")):
        in_channels = [96, 192, 384, 768]
    elif backbone_name.startswith(
            ("convnextv2_atto", "convnextv2_femto",
             "convnextv2_pico", "convnextv2_nano")):
        in_channels = [40, 80, 160, 320]
    else:
        in_channels = [128, 256, 512, 1024]

    iou_thrs = cfg.get("cascade_iou_thresholds", [0.5, 0.6, 0.7])
    stg_w = cfg.get("cascade_stage_weights", [1.0, 0.5, 0.25])
    target_stds_list = cfg.get("cascade_target_stds", [
        [0.1, 0.1, 0.2, 0.2],
        [0.05, 0.05, 0.1, 0.1],
        [0.033, 0.033, 0.067, 0.067],
    ])

    bbox_head_list = []
    for s in range(len(iou_thrs)):
        bbox_loss_type = cfg.get("bbox_loss_type", "SmoothL1Loss")
        if bbox_loss_type == "GIoULoss":
            bbox_loss = dict(
                type="GIoULoss",
                loss_weight=cfg.get("bbox_loss_weight", 10.0),
            )
            bbox_reg_decoded = True
        elif bbox_loss_type == "BalancedL1Loss":
            bbox_loss = dict(
                type="BalancedL1Loss",
                alpha=cfg.get("bbox_balanced_l1_alpha", 0.5),
                gamma=cfg.get("bbox_balanced_l1_gamma", 1.5),
                beta=cfg.get("bbox_balanced_l1_beta", 1.0),
                loss_weight=cfg.get("bbox_loss_weight", 1.0),
            )
            bbox_reg_decoded = cfg.get("bbox_reg_decoded_bbox", False)
        else:
            bbox_loss = dict(
                type="SmoothL1Loss",
                beta=cfg.get("bbox_smooth_l1_beta", 1.0),
                loss_weight=cfg.get("bbox_loss_weight", 1.0),
            )
            bbox_reg_decoded = cfg.get("bbox_reg_decoded_bbox", False)
        if cfg.get("bbox_loss_cls_type", "CrossEntropyLoss") == "FocalLoss":
            bbox_loss_cls = dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=cfg.get("bbox_focal_gamma", 2.0),
                alpha=cfg.get("bbox_focal_alpha", 0.25),
                loss_weight=cfg.get("bbox_loss_cls_weight", 1.0),
            )
        else:
            bbox_loss_cls = dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=cfg.get("bbox_loss_cls_weight", 1.0),
            )
        bbox_head = dict(
            type=cfg.get("bbox_head_type", "Shared2FCBBoxHead"),
            in_channels=fpn_ch,
            fc_out_channels=cfg.get("fc_out_channels", 1024),
            roi_feat_size=cfg.get("roi_out_size_bbox", 7),
            num_classes=num_classes,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=target_stds_list[s],
            ),
            reg_class_agnostic=cfg.get("reg_class_agnostic", True),
            reg_decoded_bbox=bbox_reg_decoded,
            loss_cls=bbox_loss_cls,
            loss_bbox=bbox_loss,
        )
        if bbox_head["type"] == "ConvFCBBoxHead":
            bbox_head.update(
                num_shared_convs=cfg.get("bbox_num_shared_convs", 2),
                num_shared_fcs=cfg.get("bbox_num_shared_fcs", 1),
                conv_out_channels=cfg.get("bbox_conv_out_channels", fpn_ch),
            )
        elif bbox_head["type"] == "DoubleConvFCBBoxHead":
            bbox_head.update(
                num_convs=cfg.get("bbox_num_convs", 4),
                num_fcs=cfg.get("bbox_num_fcs", 2),
                conv_out_channels=cfg.get("bbox_conv_out_channels", fpn_ch),
            )
        bbox_head_list.append(bbox_head)

    rcnn_train_cfgs = []
    for s, iou in enumerate(iou_thrs):
        rcnn_sampler_type = cfg.get("rcnn_sampler_type", "RandomSampler")
        if rcnn_sampler_type == "CombinedSampler":
            rcnn_sampler = dict(
                type="CombinedSampler",
                num=cfg.get("rcnn_batch_size", 512),
                pos_fraction=cfg.get("rcnn_positive_fraction", 0.25),
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
                pos_sampler=dict(type=cfg.get(
                    "rcnn_pos_sampler_type",
                    "InstanceBalancedPosSampler")),
                neg_sampler=dict(
                    type=cfg.get("rcnn_neg_sampler_type",
                                 "IoUBalancedNegSampler"),
                    floor_thr=cfg.get("rcnn_iou_balanced_floor_thr", -1),
                    floor_fraction=cfg.get(
                        "rcnn_iou_balanced_floor_fraction", 0),
                    num_bins=cfg.get("rcnn_iou_balanced_num_bins", 3),
                ),
            )
        else:
            rcnn_sampler = dict(
                type=rcnn_sampler_type,
                num=cfg.get("rcnn_batch_size", 512),
                pos_fraction=cfg.get("rcnn_positive_fraction", 0.25),
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
                loss_key=cfg.get("rcnn_ohem_loss_key", "loss_cls"),
            )
        rcnn_train_cfgs.append(dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=iou,
                neg_iou_thr=iou,
                min_pos_iou=iou,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=rcnn_sampler,
            mask_size=cfg.get("mask_size", 28),
            pos_weight=-1,
            debug=False,
        ))

    detector_type = cfg.get("detector_type", "CascadeRCNN")
    use_cascade_roi = detector_type == "CascadeRCNN"

    def make_roi_extractor(prefix, output_size):
        extractor_type = cfg.get(f"{prefix}_roi_extractor_type",
                                 "SingleRoIExtractor")
        roi_layer_type = cfg.get("roi_layer_type", "RoIAlign")
        roi_layer = dict(
            type=roi_layer_type,
            output_size=output_size,
        )
        if roi_layer_type == "RoIAlign":
            roi_layer["sampling_ratio"] = cfg.get("roi_sampling_ratio", 0)
        extractor = dict(
            type=extractor_type,
            roi_layer=roi_layer,
            out_channels=fpn_ch,
            featmap_strides=roi_featmap_strides,
        )
        if extractor_type == "GenericRoIExtractor":
            extractor["aggregation"] = cfg.get(
                f"{prefix}_roi_aggregation",
                cfg.get("roi_aggregation", "sum"),
            )
        return extractor

    roi_head_cfg = dict(
        type="CascadeRoIHead" if use_cascade_roi else "StandardRoIHead",
        bbox_roi_extractor=make_roi_extractor(
            "bbox",
            cfg.get("roi_out_size_bbox", 7),
        ),
        bbox_head=bbox_head_list if use_cascade_roi else bbox_head_list[0],
        mask_roi_extractor=make_roi_extractor(
            "mask",
            cfg.get("roi_out_size_mask", 14),
        ),
        mask_head=None,
    )
    if use_cascade_roi:
        roi_head_cfg.update(
            num_stages=len(iou_thrs),
            stage_loss_weights=stg_w,
        )

    if backbone_type == "ResNet":
        backbone_cfg = dict(
            type="ResNet",
            depth=cfg.get("backbone_depth", 50),
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=cfg.get("resnet_frozen_stages", -1),
            norm_cfg=dict(type="BN", requires_grad=True),
            norm_eval=True,
            style="pytorch",
            init_cfg=(
                dict(type="Pretrained",
                     checkpoint="torchvision://resnet50")
                if cfg.get("pretrained", True) else None
            ),
        )
    else:
        backbone_cfg = dict(
            type="ConvNeXtV2Backbone",
            model_name=backbone_name,
            pretrained=cfg.get("pretrained", True),
            drop_path_rate=drop_path_rate,
            out_indices=(0, 1, 2, 3),
            freeze_stages=cfg.get("freeze_stages", 0),
            img_size=cfg.get("backbone_img_size", None),
            norm_type=cfg.get("backbone_norm_type", "LayerNorm"),
        )

    neck_cfg = dict(
        type=neck_type,
        in_channels=in_channels,
        out_channels=fpn_ch,
        num_outs=fpn_num_outs,
    )
    if cfg.get("neck_no_norm_on_lateral") is not None:
        neck_cfg["no_norm_on_lateral"] = cfg.get("neck_no_norm_on_lateral")
    if cfg.get("neck_add_extra_convs") is not None:
        neck_cfg["add_extra_convs"] = cfg.get("neck_add_extra_convs")
    if cfg.get("neck_relu_before_extra_convs") is not None:
        neck_cfg["relu_before_extra_convs"] = cfg.get(
            "neck_relu_before_extra_convs")
    if cfg.get("neck_upsample_cfg") is not None:
        neck_cfg["upsample_cfg"] = cfg.get("neck_upsample_cfg")
    if neck_type == "NASFPN":
        neck_cfg["stack_times"] = cfg.get("nasfpn_stack_times", 1)
    if cfg.get("neck_norm_cfg") is not None:
        neck_cfg["norm_cfg"] = cfg.get("neck_norm_cfg")
    if cfg.get("neck_act_cfg") is not None:
        neck_cfg["act_cfg"] = cfg.get("neck_act_cfg")

    rpn_head_cfg = dict(
        type=rpn_head_type,
        in_channels=fpn_ch,
        feat_channels=fpn_ch,
    )
    if rpn_head_type == "GARPNHead":
        rpn_head_cfg.update(
            approx_anchor_generator=dict(
                type="AnchorGenerator",
                octave_base_scale=cfg.get("anchor_scale", 8),
                scales_per_octave=cfg.get("ga_scales_per_octave", 3),
                ratios=cfg.get("ga_approx_ratios",
                               cfg.get("anchor_ratios", [0.5, 1.0, 2.0])),
                strides=rpn_strides,
            ),
            square_anchor_generator=dict(
                type="AnchorGenerator",
                ratios=[1.0],
                scales=[cfg.get("anchor_scale", 8)],
                strides=rpn_strides,
            ),
            anchor_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=cfg.get("ga_anchor_target_stds",
                                    [0.07, 0.07, 0.14, 0.14]),
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=cfg.get("ga_bbox_target_stds",
                                    [0.07, 0.07, 0.11, 0.11]),
            ),
            loc_filter_thr=cfg.get("ga_loc_filter_thr", 0.01),
            loss_loc=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0,
                          alpha=0.25, loss_weight=1.0),
            loss_shape=dict(type="BoundedIoULoss", beta=0.2,
                            loss_weight=1.0),
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True,
                          loss_weight=1.0),
            loss_bbox=dict(type="SmoothL1Loss", beta=1.0,
                           loss_weight=1.0),
        )
    else:
        if cfg.get("rpn_loss_cls_type", "CrossEntropyLoss") == "FocalLoss":
            rpn_loss_cls = dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=cfg.get("rpn_focal_gamma", 2.0),
                alpha=cfg.get("rpn_focal_alpha", 0.25),
                loss_weight=cfg.get("rpn_loss_cls_weight", 1.0),
            )
        else:
            rpn_loss_cls = dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_weight=cfg.get("rpn_loss_cls_weight", 1.0),
            )
        rpn_head_cfg.update(
            anchor_generator=dict(
                type="AnchorGenerator",
                scales=[cfg.get("anchor_scale", 8)],
                ratios=cfg.get("anchor_ratios", [0.5, 1.0, 2.0]),
                strides=rpn_strides,
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
            ),
            loss_cls=rpn_loss_cls,
            loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0,
                           loss_weight=cfg.get("rpn_loss_bbox_weight", 1.0)),
        )

    rpn_sampler_type = cfg.get("rpn_sampler_type", "RandomSampler")
    if rpn_sampler_type == "CombinedSampler":
        rpn_sampler = dict(
            type="CombinedSampler",
            num=cfg.get("rpn_batch_size", 256),
            pos_fraction=cfg.get("rpn_positive_fraction", 0.5),
            neg_pos_ub=-1,
            add_gt_as_proposals=False,
            pos_sampler=dict(type=cfg.get(
                "rpn_pos_sampler_type",
                "InstanceBalancedPosSampler")),
            neg_sampler=dict(
                type=cfg.get("rpn_neg_sampler_type",
                             "IoUBalancedNegSampler"),
                floor_thr=cfg.get("rpn_iou_balanced_floor_thr", -1),
                floor_fraction=cfg.get("rpn_iou_balanced_floor_fraction", 0),
                num_bins=cfg.get("rpn_iou_balanced_num_bins", 3),
            ),
        )
    elif rpn_sampler_type == "PseudoSampler":
        rpn_sampler = dict(type="PseudoSampler")
    else:
        rpn_sampler = dict(
            type=rpn_sampler_type,
            num=cfg.get("rpn_batch_size", 256),
            pos_fraction=cfg.get("rpn_positive_fraction", 0.5),
            neg_pos_ub=-1,
            add_gt_as_proposals=False,
        )

    if cfg.get("rpn_assigner_type", "MaxIoUAssigner") == "ATSSAssigner":
        rpn_assigner = dict(
            type="ATSSAssigner",
            topk=cfg.get("rpn_atss_topk", 9),
            ignore_iof_thr=-1,
        )
    else:
        rpn_assigner = dict(
            type="MaxIoUAssigner",
            pos_iou_thr=cfg.get("rpn_pos_iou_thr", 0.7),
            neg_iou_thr=cfg.get("rpn_neg_iou_thr", 0.3),
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1,
        )

    rpn_train_cfg = dict(
        assigner=rpn_assigner,
        sampler=rpn_sampler,
        allowed_border=cfg.get("rpn_allowed_border", 0),
        pos_weight=-1,
        debug=False,
    )
    if rpn_head_type == "GARPNHead":
        rpn_train_cfg.update(
            ga_assigner=dict(
                type="ApproxMaxIoUAssigner",
                pos_iou_thr=cfg.get("rpn_pos_iou_thr", 0.7),
                neg_iou_thr=cfg.get("rpn_neg_iou_thr", 0.3),
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
            ),
            ga_sampler=dict(
                type="RandomSampler",
                num=cfg.get("rpn_batch_size", 256),
                pos_fraction=cfg.get("rpn_positive_fraction", 0.5),
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=cfg.get("ga_allowed_border", -1),
            center_ratio=cfg.get("ga_center_ratio", 0.2),
            ignore_ratio=cfg.get("ga_ignore_ratio", 0.5),
        )

    model_cfg = dict(
        type=detector_type,
        data_preprocessor=dict(
            type="DetDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_mask=True,
            pad_size_divisor=32,
        ),
        backbone=backbone_cfg,
        neck=neck_cfg,
        rpn_head=rpn_head_cfg,
        roi_head=roi_head_cfg,
        train_cfg=dict(
            rpn=rpn_train_cfg,
            rpn_proposal=dict(
                nms_pre=cfg.get("rpn_pre_nms_train", 2000),
                nms_post=cfg.get("rpn_post_nms_train", 2000),
                max_per_img=cfg.get("rpn_post_nms_train", 2000),
                nms=dict(type="nms",
                         iou_threshold=cfg.get("rpn_nms_thresh", 0.7)),
                min_bbox_size=0,
            ),
            rcnn=rcnn_train_cfgs if use_cascade_roi else rcnn_train_cfgs[0],
        ),
        test_cfg=dict(
            rpn=dict(
                nms_pre=cfg.get("rpn_pre_nms_test", 1000),
                nms_post=cfg.get("rpn_post_nms_test", 1000),
                max_per_img=cfg.get("rpn_post_nms_test", 1000),
                nms=dict(type="nms",
                         iou_threshold=cfg.get("rpn_nms_thresh", 0.7)),
                min_bbox_size=0,
            ),
            rcnn=dict(
                score_thr=cfg.get("score_threshold", 0.05),
                nms=dict(type="nms",
                         iou_threshold=cfg.get("nms_threshold", 0.5)),
                max_per_img=cfg.get("max_det", 300),
                mask_thr_binary=cfg.get("mask_threshold", 0.5),
            ),
        ),
    )
    mask_head = dict(
        type=cfg.get("mask_head_type", "FCNMaskHead"),
        num_convs=cfg.get("mask_num_convs", 4),
        in_channels=fpn_ch,
        conv_out_channels=fpn_ch,
        num_classes=num_classes,
        class_agnostic=cfg.get("mask_class_agnostic", False),
        loss_mask=dict(type="CrossEntropyLoss", use_mask=True,
                       loss_weight=1.0),
    )
    if mask_head["type"] == "SCNetMaskHead":
        mask_head["conv_to_res"] = cfg.get("mask_conv_to_res", True)
    if mask_head["type"] == "DiceFCNMaskHead":
        mask_head["dice_loss_weight"] = cfg.get("mask_dice_loss_weight", 1.0)
        mask_head["ce_loss_weight"] = cfg.get("mask_ce_loss_weight", 1.0)
    model_cfg["roi_head"]["mask_head"] = mask_head
    return model_cfg
