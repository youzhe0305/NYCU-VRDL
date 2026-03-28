import argparse
import csv
import os

import torch

from dataset import build_dataloaders
from model import build_model


@torch.no_grad()
def evaluate_validation(model, val_loader, device):
    """Evaluate model accuracy on the validation set."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


@torch.no_grad()
def generate_predictions(model, test_loader, output_path, device):
    """Generate prediction.csv for CodaBench submission."""
    model.eval()
    results = []

    for images, img_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        for name, pred in zip(img_names, predicted.cpu().numpy()):
            base_name = os.path.splitext(name)[0]
            results.append((base_name, int(pred)))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        for name, pred in sorted(results, key=lambda x: x[0]):
            writer.writerow([name, pred])

    print(f"Predictions saved to {output_path} ({len(results)} images)")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(num_classes=args.num_classes, dropout=0.0)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(
        f"Loaded checkpoint: epoch={checkpoint['epoch']}, "
        f"val_acc={checkpoint['val_acc']:.4f}"
    )

    _, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    if args.mode in ("val", "both"):
        evaluate_validation(model, val_loader, device)

    if args.mode in ("test", "both"):
        generate_predictions(model, test_loader, args.output, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet152 Classifier")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to dataset root")
    parser.add_argument("--output", type=str, default="prediction.csv",
                        help="Output path for prediction.csv")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["val", "test", "both"],
        help=(
            "val: validate only, test: generate predictions, "
            "both: do both"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=448)
    args = parser.parse_args()
    main(args)
