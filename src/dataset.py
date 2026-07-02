"""
Handles loading the dataset from disk and building DataLoaders
for train, validation, and test splits.
"""

import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from transforms import train_transform, val_transform, test_transform

# src/ -> project root -> data/processed
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

BATCH_SIZE = 32
NUM_WORKERS = 4


def get_train_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True):
    train_dataset = ImageFolder(
        root=os.path.join(DATA_DIR, "train"),
        transform=train_transform
    )
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def get_val_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    val_dataset = ImageFolder(
        root=os.path.join(DATA_DIR, "val"),
        transform=val_transform
    )
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


def get_test_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    test_dataset = ImageFolder(
        root=os.path.join(DATA_DIR, "test"),
        transform=test_transform
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


def get_class_names():
    train_dataset = ImageFolder(root=os.path.join(DATA_DIR, "train"))
    return train_dataset.classes


if __name__ == "__main__":
    train_loader = get_train_loader()
    val_loader = get_val_loader()
    test_loader = get_test_loader()

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Classes: {get_class_names()}")
