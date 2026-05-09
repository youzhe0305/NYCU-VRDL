# NYCU VRDL 2026 HW3 - Instance Segmentation on Medical Cell Images

* Student ID: 112550069
* Name: You Zhe, Xie

4-class instance segmentation on medical cell microscopy images using **Cascade Mask R-CNN** with a **ConvNeXt-V2-Base** backbone and **FPN**. Built for the Visual Recognition HW3 competition.

See [report](report.pdf) for the detailed method, results, and ablation study.

![alt text](figures/leaderboard.png)

## Introduction

This project tackles instance segmentation of 4 cell types in medical microscopy images (209 train / 101 test). The approach adapts Cascade Mask R-CNN with several key design choices:

- A **ConvNeXt-V2-Base backbone** (ImageNet pre-trained) extracts multi-scale features (C0–C3, channels 128/256/512/1024).
- **FPN Neck** projects features to 256 channels and produces a 5-level feature pyramid (P2–P6).
- **RPN** with cell-optimized anchors (scale=4, square-only) generates ~1000 proposals per image.
- **3-stage Cascade RoI Head** (IoU thresholds 0.5 → 0.6 → 0.7) progressively refines bbox and classification.
- **FCN Mask Head** produces 28×28 per-instance binary masks.
- CNN-based architecture chosen over Transformer-based (e.g., Mask2Former + Swin) because the small dataset (209 images) favors CNN's inductive biases.

**Best validation mAP@50: 0.724**

## Environment Setup

**Python** 3.9 | **CUDA** 12.x (tested with torch 2.8.0 + CUDA 12.8)

**Step 1 — Install PyTorch** (choose the wheel matching your CUDA version):

```bash
# CUDA 12.x
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
```

**Step 2 — Install remaining dependencies:**

```bash
pip install -r requirements.txt
```

## Dataset Structure

Place the dataset under the `data/` directory:

```
data/
├── train/                          # Training images (209 samples)
│   ├── XXXX.tif                    # Cell images
│   ├── XXXX_class1.tif … _class4.tif  # Per-class instance masks
├── test_release/                   # Test images (101 samples)
└── test_image_name_to_ids.json     # Image name → ID mapping for submission
```

### Data Preparation

Convert raw `.tif` masks to COCO annotation format:

```bash
cd Cascade_Mask_R-CNN
python prepare_coco_dataset.py
```

This generates a COCO-format JSON annotation file from the per-class mask `.tif` files.

## Usage

All commands below assume you are in the `HW3/Cascade_Mask_R-CNN/` directory.

### Training

```bash
python train.py --config ../configs/cascade_mask_rcnn_best.yaml
```

All hyperparameters are defined in `configs/cascade_mask_rcnn_best.yaml`. CLI arguments override the config file if provided:

| Argument | Default | Description |
|---|---|---|
| `--config` | — | Path to YAML config file |
| `--batch_size` | `2` | Batch size (reduce if OOM) |
| `--epochs` | `50` | Number of training epochs |
| `--device` | `cuda:0` | Training device |
| `--resume` | — | Resume from checkpoint (restores all state) |
| `--load_weight` | — | Load weights only (epoch resets to 1) |

Resume from a checkpoint:

```bash
python train.py --config ../configs/cascade_mask_rcnn_best.yaml \
    --resume ../checkpoints/cascade_mask_rcnn/best_model_XXXXXX.pth \
    --epochs 100
```

### Inference

Generate the test submission file (`test-results.json` in COCO RLE format):

```bash
python evaluate.py --config ../configs/cascade_mask_rcnn_best.yaml \
    --checkpoint ../checkpoints/cascade_mask_rcnn/best_model_XXXXXX.pth \
    --mode test --output ../test-results.json
```

Validate on the held-out val set:

```bash
python evaluate.py --config ../configs/cascade_mask_rcnn_best.yaml \
    --checkpoint ../checkpoints/cascade_mask_rcnn/best_model_XXXXXX.pth \
    --mode val
```

Do both at once:

```bash
python evaluate.py --config ../configs/cascade_mask_rcnn_best.yaml \
    --checkpoint ../checkpoints/cascade_mask_rcnn/best_model_XXXXXX.pth \
    --mode both --output ../test-results.json
```

## Model Architecture

```
ConvNeXt-V2-Base (ImageNet pretrained, drop_path=0.1)
    └── Multi-Scale Features: C0 (128ch), C1 (256ch), C2 (512ch), C3 (1024ch)
         └── FPN Neck → 5-level pyramid (P2–P6, 256ch each)
              └── RPN Head (anchor_scale=4, ratio=[1.0], NMS=0.45)
                   └── ~1000 proposals
                        └── Cascade RoI Head (3 stages)
                             ├── Stage 1: IoU ≥ 0.5, weight=1.0
                             ├── Stage 2: IoU ≥ 0.6, weight=0.5
                             └── Stage 3: IoU ≥ 0.7, weight=0.25
                                  └── FCN Mask Head → 28×28 per-instance binary mask
```

## Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW (differential LR for backbone vs head) |
| LR (head/neck/RPN) | 1e-4 |
| LR (backbone) | 1e-5 |
| Weight decay | 0.05 (0 for norm layers & bias) |
| Scheduler | Cosine annealing (5-epoch linear warmup) |
| Loss | CrossEntropy (cls) + SmoothL1 (bbox) + BCE (mask) |
| Augmentation | RandomResize, RandomCrop, HFlip, VFlip, ColorJitter, ElasticDeform |
| Batch size | 2 |
| Input resolution | 1024×1024 |
| Stochastic Depth | 0.1 |
| Mixed precision | AMP |

## Performance

| Stage | val segm_mAP_50 | Key Change |
|---|:---:|---|
| Baseline | 0.658 | Cascade Mask R-CNN + ConvNeXt-V2-Base initial setup |
| + drop_path_rate=0.1 | 0.664 | Reduced Stochastic Depth |
| + anchor_scale=4 | 0.708 | Smaller anchors for cell-scale proposals |
| + square-only anchors | 0.713 | ratio=[1.0], reduced memory (17.6→13.3 GB) |
| + RPN NMS=0.5 | 0.723 | Fewer redundant proposals |
| + RPN NMS=0.45 | **0.724** | Final best |

### Key Config Options (`configs/cascade_mask_rcnn_best.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 50 | Total training epochs |
| `batch_size` | 2 | Batch size |
| `lr` | 1e-4 | Learning rate for head/neck/RPN |
| `backbone_lr` | 1e-5 | Learning rate for backbone |
| `drop_path_rate` | 0.1 | Stochastic Depth rate |
| `anchor_scale` | 4 | RPN anchor base size |
| `anchor_ratios` | [1.0] | Square-only anchors |
| `rpn_nms_thresh` | 0.45 | RPN proposal NMS threshold |
| `img_scale` | 1024 | Input image scale |
| `score_threshold` | 0.05 | Detection confidence threshold |
| `mask_threshold` | 0.5 | Mask binarization threshold |
| `tta` | true | Test-time augmentation (original + H-flip + V-flip) |

## Output Format

The test submission (`test-results.json`) is a JSON list in COCO RLE format:

```json
[
  {
    "image_id": 1,
    "bbox": [x, y, width, height],
    "score": 0.95,
    "category_id": 1,
    "segmentation": {"size": [H, W], "counts": "..."}
  }
]
```

## File Structure

```
HW3/
├── Cascade_Mask_R-CNN/
│   ├── model.py                    # Model construction (backbone, FPN, Cascade heads)
│   ├── train.py                    # Training loop with AP50 validation
│   ├── evaluate.py                 # Inference, TTA, and submission generation
│   ├── dataset.py                  # Dataset, augmentations, data loaders
│   └── prepare_coco_dataset.py     # Convert .tif masks → COCO annotation JSON
├── configs/
│   ├── cascade_mask_rcnn_best.yaml # Best model hyperparameters
│   └── cascade_mask_rcnn.yaml      # Default hyperparameters
├── data/
│   ├── train/                      # Training images + per-class masks (209 samples)
│   ├── test_release/               # Test images (101 samples)
│   └── test_image_name_to_ids.json
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── report.md                       # Detailed report (Chinese)
├── report_EN.md                    # Detailed report (English)
└── (114-2) HW3 Slides.pdf         # Project spec
```
