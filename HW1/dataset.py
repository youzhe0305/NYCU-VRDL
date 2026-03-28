import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])


def get_val_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ClassificationDataset(Dataset):
    """ImageFolder-style dataset for train/val splits.

    When `include_original=True`, the dataset length doubles:
      - indices [0, N)  return the original image (resize + normalize only)
      - indices [N, 2N) return the augmented image (full augmentation)
    This lets the model see both clean and augmented views each epoch.
    """

    def __init__(self, root_dir, transform=None,
                 include_original=False, original_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.include_original = include_original
        self.original_transform = original_transform
        self.samples = []
        self.class_to_idx = {}

        class_names = sorted(
            os.listdir(root_dir), key=lambda x: int(x)
        )
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in sorted(os.listdir(class_dir)):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, idx))

    def __len__(self):
        if self.include_original:
            return len(self.samples) * 2
        return len(self.samples)

    def __getitem__(self, idx):
        n = len(self.samples)

        if self.include_original and idx >= n:
            img_path, label = self.samples[idx - n]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.include_original and self.original_transform:
            image = self.original_transform(image)
        elif self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    """Dataset for test split (no labels)."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = sorted(os.listdir(root_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


def build_dataloaders(data_dir, batch_size=32, num_workers=4, img_size=224,
                      include_original=False):
    train_dataset = ClassificationDataset(
        root_dir=os.path.join(data_dir, "train"),
        transform=get_train_transforms(img_size),
        include_original=include_original,
        original_transform=get_val_transforms(img_size),
    )
    val_dataset = ClassificationDataset(
        root_dir=os.path.join(data_dir, "val"),
        transform=get_val_transforms(img_size),
    )
    test_dataset = TestDataset(
        root_dir=os.path.join(data_dir, "test"),
        transform=get_val_transforms(img_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
