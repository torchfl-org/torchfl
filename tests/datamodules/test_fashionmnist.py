#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for FashionMNIST PyTorch LightningDataModule module in `torchfl` package."""
import pytest
from torchfl.datamodules.fashionmnist import FashionMNISTDataModule
from collections import Counter


def collate_federated(agent_data):
    """Helper method for collating the federated dataset for an agent.
    Args:
        agent_data (DataLoader): data owned by a given agent
    Returns:
        List: list of labels
    """
    all = list()
    for i in agent_data:
        all.extend(i[1].tolist())
    return all


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
    collated_labels = collate_federated(dataloader[0])
    assert len(collated_labels) == 6000
    frequency = Counter(collated_labels)
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
    collated_labels = collate_federated(dataloader[0])
    assert len(collated_labels) == 6000
    frequency = Counter(collated_labels)
    assert len(frequency.keys()) == 2
