import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

torch.manual_seed(123)


def create_dataloaders():
    train_tf = None # TODO: Define the train transform
    val_tf = None # TODO: Define the validation transform. No random augmentations here.

    train_dataset = None # TODO: Load the train dataset. Make sure to pass train_tf to it.
    val_dataset = None # TODO: Load the val dataset.

    train_loader = None # TODO: Create the train dataloader
    val_loader = None # TODO: Create the val dataloader

    return train_loader, val_loader

