import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GeM(nn.Module):
    """Generalized Mean Pooling: p=1 → avg pool, p→∞ → max pool."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (1, 1)
        )
        return pooled.pow(1.0 / self.p)


class ResNetClassifier(nn.Module):
    """
    ResNet152 backbone (trainable with separate LR) with a custom classification head.

    Backbone can be fine-tuned with a lower learning rate; the classification
    head uses the main learning rate.
    """

    def __init__(self, num_classes=100, dropout=0.5):
        super().__init__()

        backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        self.backbone = backbone
        # Backbone is trainable (no freeze); use backbone_lr in optimizer
        # for fine-tuning.

        # Replace avgpool with GeM pooling
        self.backbone.avgpool = GeM(p=3.0)

        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes=100, dropout=0.5):
    model = ResNetClassifier(num_classes=num_classes, dropout=dropout)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {total_params - trainable_params:,}")
    assert total_params < 100_000_000, (
        f"Model has {total_params:,} params, exceeding 100M limit!"
    )
    return model
