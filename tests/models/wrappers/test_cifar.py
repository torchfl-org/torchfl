#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for CIFAR model wrapper in `torchfl` package."""

import pytest
from torchvision import datasets, transforms
import os
from torchfl.models.wrapper.cifar import CIFAR10, CIFAR100
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
}


@pytest.fixture
def cifar10_train_loader():
    """Fixture for CIFAR10 train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.CIFAR10(
        root=os.path.join(os.pardir, "data"),
        train=True,
        download=True,
        transform=data_transforms["train"],
    )


@pytest.fixture
def cifar10_test_loader():
    """Fixture for CIFAR10 test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.CIFAR10(
        root=os.path.join(os.pardir, "data"),
        train=False,
        download=True,
        transform=data_transforms["train"],
    )


@pytest.fixture
def cifar100_train_loader():
    """Fixture for CIFAR100 train dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.CIFAR100(
        root=os.path.join(os.pardir, "data"),
        train=True,
        download=True,
        transform=data_transforms["train"],
    )


@pytest.fixture
def cifar100_test_loader():
    """Fixture for CIFAR100 test dataset.
    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.CIFAR100(
        root=os.path.join(os.pardir, "data"),
        train=False,
        download=True,
        transform=data_transforms["train"],
    )


def test_cifar10_wrapper(cifar10_train_loader, cifar10_test_loader):
    """Testing the CIFAR10 model wrapper with PyTorch Lightning wrapper.

    Args:
        cifar10_train_loader (Dataset): PyTorch Dataset object.
        cifar10_test_loader (Dataset): PyTorch Dataset object.
    """
    model = CIFAR10(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=cifar10_train_loader, batch_size=1),
        DataLoader(dataset=cifar10_test_loader, batch_size=1),
    )


def test_cifar100_wrapper(cifar100_train_loader, cifar100_test_loader):
    """Testing the CIFAR100 model wrapper with PyTorch lightning wrapper.

    Args:
        cifar100_train_loader (Dataset): PyTorch Dataset object.
        cifar100_test_loader (Dataset): PyTorch Dataset object.
    """
    model = CIFAR100(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(
        model,
        DataLoader(dataset=cifar100_train_loader, batch_size=1),
        DataLoader(dataset=cifar100_test_loader, batch_size=1),
    )
