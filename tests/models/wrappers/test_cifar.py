#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for CIFAR model wrapper in `torchfl` package."""

import pytest
from torchfl.datamodules.cifar import CIFARDataModule
from torchfl.models.wrapper.cifar import CIFAR10, CIFAR100
from pytorch_lightning import Trainer


@pytest.fixture
def cifar10_data_module():
    """Fixture for CIFAR10 PyTorch LightningDataModule

    Returns:
        CIFARDataModule: PyTorch LightningDataModule for CIFAR10.
    """
    return CIFARDataModule(dataset_name="cifar10")


@pytest.fixture
def cifar100_data_module():
    """Fixture for CIFAR100 PyTorch LightningDataModule

    Returns:
        CIFARDataModule: PyTorch LightningDataModule for CIFAR100.
    """
    return CIFARDataModule(dataset_name="cifar100")


def test_cifar10_wrapper(cifar10_data_module):
    """Testing the CIFAR10 model wrapper with PyTorch Lightning wrapper.

    Args:
        cifar10_data_module (CIFARDataModule): PyTorch LightningDataModule for CIFAR10.
    """
    model = CIFAR10(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=cifar10_data_module)


def test_cifar100_wrapper(cifar100_data_module):
    """Testing the CIFAR100 model wrapper with PyTorch lightning wrapper.

    Args:
        cifar100_data_module (CIFARDataModule): PyTorch LightningDataModule for CIFAR100.
    """
    model = CIFAR100(
        "densenet121", "sgd", {"lr": 0.1, "momentum": 0.9}, {"num_channels": 3}
    )
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model, datamodule=cifar100_data_module)
