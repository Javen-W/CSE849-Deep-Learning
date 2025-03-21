import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

torch.manual_seed(123)

DATASET_ROOT = '/home/javen/Documents/Datasets/CSE849-HW3/'
NORMAL_MEAN = [0.438, 0.435, 0.422]
NORMAL_STD = [0.228, 0.225, 0.231]
batch_size = 64
num_workers = 4


def create_dataloaders():
    # TODO: Define the training transform
    train_tf = v2.Compose([
        # v2.RandomResizedCrop(size=(40, 40), scale=(0.8, 1.0)),  # Adjust size as needed
        v2.RandomHorizontalFlip(),
        # v2.RandomRotation(degrees=15),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=NORMAL_MEAN, std=NORMAL_STD),
        v2.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        v2.GaussianNoise(mean=0, sigma=0.1, clip=False),
    ])

    # TODO: Define the validation transform. No random augmentations here.
    val_tf = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=NORMAL_MEAN, std=NORMAL_STD),
    ])

    # TODO: Load the train dataset. Make sure to pass train_tf to it.
    train_dataset = ImageFolder(
        root=os.path.join(DATASET_ROOT, 'train'),
        transform=train_tf,
    )

    # TODO: Load the val dataset.
    val_dataset = ImageFolder(
        root=os.path.join(DATASET_ROOT, 'val'),
        transform=val_tf,
    )

    # TODO: Create the train dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # TODO: Create the val dataloader
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader

