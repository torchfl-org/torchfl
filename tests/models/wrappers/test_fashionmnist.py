#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for FashionMNIST model wrapper in `torchfl` package."""

import pytest
from torchvision import transforms
from torchfl.datamodules.fashionmnist import FashionMNISTDataModule
from torchfl.models.wrapper.fashionmnist import FashionMNIST
from pytorch_lightning import Trainer

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
def fashionmnist_single_channel_data_module():
    """Fixture for FashionMNIST single channel data module.

    Returns:
        FashionMNISTDataModule: PyTorch LightningDataModule for FashionMNIST.
    """
    global data_transforms
    return FashionMNISTDataModule(
        train_transforms=data_transforms["train_single_channel"]
    )


@pytest.fixture
def fashionmnist_three_channel_data_module():
    """Fixture for FashionMNIST three channel data module.

    Returns:
        FashionMNISTDataModule: PyTorch LightningDataModule for FashionMNIST.
    """
    global data_transforms
    return FashionMNISTDataModule(
        train_transforms=data_transforms["train_three_channel"]
    )


def test_fashionmnist_single_channel_wrapper(fashionmnist_single_channel_data_module):
    """Testing the FashionMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        FashionMNISTDataModule: PyTorch LightningDataModule for FashionMNIST.
    """
    model = FashionMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 1}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=fashionmnist_single_channel_data_module)


def test_fashionmnist_three_channel_wrapper(fashionmnist_three_channel_data_module):
    """Testing the FashionMNIST model wrapper with PyTorch Lightning wrapper.

    Args:
        FashionMNISTDataModule: PyTorch LightningDataModule for FashionMNIST.
    """
    model = FashionMNIST(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=fashionmnist_three_channel_data_module)
