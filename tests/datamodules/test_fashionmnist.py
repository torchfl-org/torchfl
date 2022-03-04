#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for FashionMNIST PyTorch LightningDataModule module in `torchfl` package."""
import pytest
from torchfl.datamodules.fashionmnist import FashionMNISTDataModule
from collections import Counter


@pytest.fixture
def fashionmnist_data_module():
    """Fixture for FashionMNIST PyTorch LightningDataModule

    Returns:
        FashionMNISTDataModule: PyTorch LightningDataModule for FashionMNIST.
    """
    return FashionMNISTDataModule()


def test_fashionmnist_train_val_split(fashionmnist_data_module):
    """Testing the fashionmnist dataset train and validation split with PyTorch Lightning wrapper.

    Args:
        fashionmnist_data_module (FashionMNISTDataModule): PyTorch LightningDataModule for fashionmnist.
    """
    fashionmnist_data_module.prepare_data()
    fashionmnist_data_module.setup(stage="fit")
    train_dataloader = fashionmnist_data_module.train_dataloader()
    val_dataloader = fashionmnist_data_module.val_dataloader()
    assert len(train_dataloader.dataset) == 54000
    assert len(val_dataloader.dataset) == 6000


def test_fashionmnist_test_split(fashionmnist_data_module):
    """Testing the fashionmnist dataset test split with PyTorch Lightning wrapper.

    Args:
        fashionmnist_data_module (FashionMNISTDataModule): PyTorch LightningDataModule for fashionmnist.
    """
    fashionmnist_data_module.prepare_data()
    fashionmnist_data_module.setup(stage="test")
    test_dataloader = fashionmnist_data_module.test_dataloader()
    assert len(test_dataloader.dataset) == 10000


def test_fashionmnist_prediction_split(fashionmnist_data_module):
    """Testing the fashionmnist dataset prediction split with PyTorch Lightning wrapper.

    Args:
        fashionmnist_data_module (FashionMNISTDataModule): PyTorch LightningDataModule for fashionmnist.
    """
    fashionmnist_data_module.prepare_data()
    fashionmnist_data_module.setup(stage="predict")
    predict_dataloader = fashionmnist_data_module.predict_dataloader()
    assert len(predict_dataloader.dataset) == 10000


def test_fashionmnist_federated_iid_split(fashionmnist_data_module):
    """Testing the fashionmnist dataset federated iid split with PyTorch Lightning wrapper.

    Args:
        fashionmnist_data_module (FashionMNISTDataModule): PyTorch LightningDataModule for fashionmnist.
    """
    fashionmnist_data_module.prepare_data()
    fashionmnist_data_module.setup(stage="fit")
    dataloader = fashionmnist_data_module.federated_iid_dataloader()
    assert len(dataloader.keys()) == 10
    assert len(dataloader[0].dataset) == 6000
    frequency = Counter(list(dataloader[0].dataset.targets))
    assert len(frequency.keys()) == 10


def test_fashionmnist_federated_non_iid_split(fashionmnist_data_module):
    """Testing the fashionmnist dataset federated non iid split with PyTorch Lightning wrapper.

    Args:
        fashionmnist_data_module (FashionMNISTDataModule): PyTorch LightningDataModule for fashionmnist.
    """
    fashionmnist_data_module.prepare_data()
    fashionmnist_data_module.setup(stage="fit")
    dataloader = fashionmnist_data_module.federated_non_iid_dataloader()
    assert len(dataloader.keys()) == 10
    all_freq = list()
    for i in range(10):
        assert len(dataloader[i].dataset) == 6000
        frequency = Counter(list(dataloader[i].dataset.targets))
        all_freq.append(len(frequency.keys()))
    assert max(all_freq) == 2
