#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for FashionMNIST model wrapper in `torchfl` package."""

import pytest
from torchvision import datasets, transforms
import os
from torchfl.models.wrapper.fashionmnist import FashionMNIST
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

data_transforms = {
    "train_single_channel": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ]
    ),
    "train_three_channel": transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


@pytest.fixture
def fashionmnist_train_single_channel_loader():
    """Fixture for FashionMNIST single channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.FashionMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def fashionmnist_test_single_channel_loader():
    """Fixture for FashionMNIST single channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.FashionMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def fashionmnist_train_three_channel_loader():
    """Fixture for FashionMNIST three channel train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.FashionMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        download=True,
        transform=data_transforms["train_three_channel"],
    )


@pytest.fixture
def fashionmnist_test_three_channel_loader():
    """Fixture for FashionMNIST three channel test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.FashionMNIST(
        root=os.path.join(os.pardir, "data"),
        train=False,
        download=True,
        transform=data_transforms["train_three_channel"],
    )


def test_fashionmnist_single_channel_wrapper(
    fashionmnist_train_single_channel_loader, fashionmnist_test_single_channel_loader
):
    """Testing the FashionMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        fashionmnist_train_three_channel_loader (Dataset): PyTorch Dataset object.
        fashionmnist_test_three_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = FashionMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=fashionmnist_train_single_channel_loader, batch_size=1),
        DataLoader(dataset=fashionmnist_test_single_channel_loader, batch_size=1),
    )


def test_fashionmnist_three_channel_wrapper(
    fashionmnist_train_three_channel_loader, fashionmnist_test_three_channel_loader
):
    """Testing the FashionMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        fashionmnist_train_three_channel_loader (Dataset): PyTorch Dataset object.
        fashionmnist_test_three_channel_loader (Dataset): PyTorch Dataset object.
    """
    model = FashionMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=fashionmnist_train_three_channel_loader, batch_size=1),
        DataLoader(dataset=fashionmnist_test_three_channel_loader, batch_size=1),
    )
