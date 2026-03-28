import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import build_dataloaders
from model import build_model


def cutmix_batch(images, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = images.shape
    rand_idx = torch.randperm(B, device=images.device)

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[rand_idx, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return images, labels, labels[rand_idx], lam


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, epoch, total_epochs
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            if np.random.random() < 0.5:
                images, labels_a, labels_b, lam = cutmix_batch(images, labels)
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (
                    (1 - lam) * criterion(outputs, labels_b)
                )
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                labels_a = labels
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels_a).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


TIME_BUDGET_SECONDS = 3600  # 1 hour of wall-clock training time


def main(args):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        include_original=args.include_original,
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")

    model = build_model(num_classes=args.num_classes, dropout=args.dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    backbone_params = [p for n, p in model.backbone.named_parameters() if "fc" not in n]
    head_params = list(model.backbone.fc.parameters())
    backbone_weight_decay = args.backbone_weight_decay
    head_weight_decay = args.weight_decay
    optimizer = AdamW(
        [
            {
                "params": backbone_params,
                "lr": args.backbone_lr,
                "weight_decay": backbone_weight_decay,
            },
            {
                "params": head_params,
                "lr": args.lr,
                "weight_decay": head_weight_decay,
            },
        ],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    best_val_acc = 0.0
    best_ckpt_path = None

    training_log = {
        "config": vars(args),
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "epochs": [],
    }
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"train_{run_timestamp}.json")

    wall_start = time.time()
    training_seconds = 0.0

    # Track peak VRAM
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch, args.epochs,
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device,
            epoch, args.epochs,
        )
        scheduler.step()

        epoch_time = time.time() - t0
        training_seconds += epoch_time

        head_lr = optimizer.param_groups[1]["lr"]
        backbone_lr = optimizer.param_groups[0]["lr"]
        budget = f"{training_seconds:.0f}/{TIME_BUDGET_SECONDS}s"
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"LR: {head_lr:.6f} (head) / {backbone_lr:.6f} (backbone) | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s  (budget used: {budget})"
        )

        is_best = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            is_best = True
            best_ckpt_path = os.path.join(
                args.save_dir, f"best_model_{run_timestamp}.pth"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, best_ckpt_path)
            print(f"  -> Best model saved (val_acc={val_acc:.4f})")

        training_log["epochs"].append({
            "epoch": epoch,
            "lr": head_lr,
            "backbone_lr": backbone_lr,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "elapsed_sec": round(epoch_time, 2),
            "is_best": is_best,
        })

        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        if training_seconds >= TIME_BUDGET_SECONDS:
            print(
                f"Time budget of {TIME_BUDGET_SECONDS}s reached "
                f"after epoch {epoch}. Stopping."
            )
            break

    total_seconds = time.time() - wall_start
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    num_params_M = sum(p.numel() for p in model.parameters()) / 1e6

    training_log["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    training_log["best_val_acc"] = round(best_val_acc, 6)
    training_log["training_seconds"] = round(training_seconds, 1)
    training_log["total_seconds"] = round(total_seconds, 1)
    training_log["peak_vram_mb"] = round(peak_vram_mb, 1)
    training_log["num_params_M"] = round(num_params_M, 1)
    training_log["best_checkpoint"] = best_ckpt_path
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Training log saved to {log_path}")
    print("\n---")
    print(f"val_acc:          {best_val_acc:.5f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_params_M:     {num_params_M:.1f}")
    if best_ckpt_path:
        print(f"best_checkpoint:  {best_ckpt_path}")


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet152 Classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--backbone_weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument(
        "--include_original",
        type=str,
        default=None,
        help=(
            "Whether to include both original and augmented images "
            "(true/false)"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Merge config into args with proper type casting.
    # Priority: CLI argument (if provided) > YAML config > argparse default.
    type_map = {
        "data_dir": str,
        "save_dir": str,
        "log_dir": str,
        "epochs": int,
        "batch_size": int,
        "lr": float,
        "backbone_lr": float,
        "weight_decay": float,
        "backbone_weight_decay": float,
        "dropout": float,
        "label_smoothing": float,
        "num_classes": int,
        "num_workers": int,
        "img_size": int,
        "include_original": bool,
    }

    for key, current_value in list(vars(args).items()):
        if key == "config":
            continue

        if current_value is not None:
            # CLI explicitly provided; keep it (normalize bool for
            # include_original)
            if key == "include_original":
                val_str = str(current_value).lower()
                setattr(args, key, val_str in {"1", "true", "yes", "y"})
            continue

        if key in config:
            raw_val = config[key]
            target_type = type_map.get(key)

            if target_type is bool:
                if isinstance(raw_val, bool):
                    cast_val = raw_val
                elif isinstance(raw_val, str):
                    cast_val = raw_val.strip().lower() in {"1", "true", "yes", "y"}
                else:
                    cast_val = bool(raw_val)
            elif target_type is not None and raw_val is not None:
                cast_val = target_type(raw_val)
            else:
                cast_val = raw_val

            setattr(args, key, cast_val)

    # Fallback defaults if not set anywhere
    if args.backbone_lr is None:
        args.backbone_lr = 1e-4
    if args.backbone_weight_decay is None:
        args.backbone_weight_decay = args.weight_decay
    if args.data_dir is None:
        args.data_dir = "data"
    if args.save_dir is None:
        args.save_dir = "checkpoints"
    if args.log_dir is None:
        args.log_dir = "log"

    main(args)
