#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for CIFAR PyTorch LightningDataModule module in `torchfl` package."""
import pytest
from torchfl.datamodules.cifar import CIFARDataModule
from collections import Counter


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


def test_cifar10_train_val_split(cifar10_data_module):
    """Testing the CIFAR10 dataset train and validation split with PyTorch Lightning wrapper.

    Args:
        cifar10_data_module (CIFARDataModule): PyTorch LightningDataModule for CIFAR10.
    """
    cifar10_data_module.prepare_data()
    cifar10_data_module.setup(stage="fit")
    train_dataloader = cifar10_data_module.train_dataloader()
    val_dataloader = cifar10_data_module.val_dataloader()
    assert len(train_dataloader.dataset) == 45000
    assert len(val_dataloader.dataset) == 5000


def test_cifar10_test_split(cifar10_data_module):
    """Testing the CIFAR10 dataset test split with PyTorch Lightning wrapper.

    Args:
        cifar10_data_module (CIFARDataModule): PyTorch LightningDataModule for CIFAR10.
    """
    cifar10_data_module.prepare_data()
    cifar10_data_module.setup(stage="test")
    test_dataloader = cifar10_data_module.test_dataloader()
    assert len(test_dataloader.dataset) == 10000


def test_cifar10_prediction_split(cifar10_data_module):
    """Testing the CIFAR10 dataset prediction split with PyTorch Lightning wrapper.

    Args:
        cifar10_data_module (CIFARDataModule): PyTorch LightningDataModule for CIFAR10.
    """
    cifar10_data_module.prepare_data()
    cifar10_data_module.setup(stage="predict")
    predict_dataloader = cifar10_data_module.predict_dataloader()
    assert len(predict_dataloader.dataset) == 10000


def test_cifar10_federated_iid_split(cifar10_data_module):
    """Testing the CIFAR10 dataset federated iid split with PyTorch Lightning wrapper.

    Args:
        cifar10_data_module (CIFARDataModule): PyTorch LightningDataModule for CIFAR10.
    """
    cifar10_data_module.prepare_data()
    cifar10_data_module.setup(stage="fit")
    dataloader = cifar10_data_module.federated_iid_dataloader()
    assert len(dataloader.keys()) == 10
    assert len(dataloader[0].dataset) == 5000
    frequency = Counter(list(dataloader[0].dataset.targets))
    assert len(frequency.keys()) == 10


def test_cifar10_federated_non_iid_split(cifar10_data_module):
    """Testing the CIFAR10 dataset federated non iid split with PyTorch Lightning wrapper.

    Args:
        cifar10_data_module (CIFARDataModule): PyTorch LightningDataModule for CIFAR10.
    """
    cifar10_data_module.prepare_data()
    cifar10_data_module.setup(stage="fit")
    dataloader = cifar10_data_module.federated_non_iid_dataloader()
    assert len(dataloader.keys()) == 10
    assert len(dataloader[0].dataset) == 5000
    frequency = Counter(list(dataloader[0].dataset.targets))
    assert len(frequency.keys()) == 2


def test_cifar100_train_val_split(cifar100_data_module):
    """Testing the CIFAR100 dataset train and validation split with PyTorch Lightning wrapper.

    Args:
        cifar100_data_module (CIFARDataModule): PyTorch LightningDataModule for CIFAR100.
    """
    cifar100_data_module.prepare_data()
    cifar100_data_module.setup(stage="fit")
    train_dataloader = cifar100_data_module.train_dataloader()
    val_dataloader = cifar100_data_module.val_dataloader()
    assert len(train_dataloader.dataset) == 45000
    assert len(val_dataloader.dataset) == 5000


def test_cifar100_test_split(cifar100_data_module):
    """Testing the CIFAR100 dataset test split with PyTorch Lightning wrapper.

    Args:
        cifar100_data_module (CIFARDataModule): PyTorch LightningDataModule for CIFAR100.
    """
    cifar100_data_module.prepare_data()
    cifar100_data_module.setup(stage="test")
    test_dataloader = cifar100_data_module.test_dataloader()
    assert len(test_dataloader.dataset) == 10000


def test_cifar100_prediction_split(cifar100_data_module):
    """Testing the CIFAR100 dataset prediction split with PyTorch Lightning wrapper.

    Args:
        cifar100_data_module (CIFARDataModule): PyTorch LightningDataModule for CIFAR100.
    """
    cifar100_data_module.prepare_data()
    cifar100_data_module.setup(stage="predict")
    predict_dataloader = cifar100_data_module.predict_dataloader()
    assert len(predict_dataloader.dataset) == 10000


def test_cifar100_federated_iid_split(cifar100_data_module):
    """Testing the CIFAR100 dataset federated iid split with PyTorch Lightning wrapper.

    Args:
        cifar100_data_module (CIFARDataModule): PyTorch LightningDataModule for CIFAR100.
    """
    cifar100_data_module.prepare_data()
    cifar100_data_module.setup(stage="fit")
    dataloader = cifar100_data_module.federated_iid_dataloader()
    assert len(dataloader.keys()) == 10
    assert len(dataloader[0].dataset) == 5000
    frequency = Counter(list(dataloader[0].dataset.targets))
    assert len(frequency.keys()) == 100


def test_cifar100_federated_non_iid_split(cifar100_data_module):
    """Testing the CIFAR100 dataset federated non iid split with PyTorch Lightning wrapper.

    Args:
        cifar100_data_module (CIFARDataModule): PyTorch LightningDataModule for CIFAR100.
    """
    cifar100_data_module.prepare_data()
    cifar100_data_module.setup(stage="fit")
    dataloader = cifar100_data_module.federated_non_iid_dataloader()
    assert len(dataloader.keys()) == 10
    assert len(dataloader[0].dataset) == 5000
    frequency = Counter(list(dataloader[0].dataset.targets))
    assert len(frequency.keys()) == 10
