#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for dataloader module in `torchfl` package."""
import pytest
from torchfl import dataloader
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
def mnist_loader():
    """Fixture for MNIST dataset.

    Returns:
        DataLoader: DataLoader object for MNIST.
    """
    return dataloader.FLDataLoader(dataset="mnist")


@pytest.fixture
def emnist_digits_loader():
    """Fixture for EMNIST Digits dataset.

    Returns:
        DataLoader: DataLoader object for EMNIST Digits.
    """
    return dataloader.FLDataLoader(dataset="emnist_digits")


@pytest.fixture
def cifar10_loader():
    """Fixture for CIFAR10 dataset.

    Returns:
        DataLoader: DataLoader object for CIFAR10.
    """
    return dataloader.FLDataLoader(dataset="cifar10")


@pytest.fixture
def cifar100_loader():
    """Fixture for CIFAR100 dataset.

    Returns:
        DataLoader: DataLoader object for CIFAR100.
    """
    return dataloader.FLDataLoader(dataset="cifar100")


def test_mnist_iid(mnist_loader):
    """Testing the iid split for MNIST

    Args:
        mnist_loader (DataLoader): MNIST federated DataLoader object.
    """
    train_dataset = mnist_loader.train_iid()
    assert len(train_dataset.keys()) == 10
    collated_labels = collate_federated(train_dataset[0])
    assert len(collated_labels) == 6000
    frequency = Counter(collated_labels)
    assert len(frequency.keys()) == 10


def test_mnist_non_iid(mnist_loader):
    """Testing the non-iid split for MNIST

    Args:
        mnist_loader (DataLoader): MNIST federated DataLoader object.
    """
    train_dataset = mnist_loader.train_non_iid()
    assert len(train_dataset.keys()) == 10
    collated_labels = collate_federated(train_dataset[0])
    assert len(collated_labels) == 6000
    frequency = Counter(collated_labels)
    assert len(frequency.keys()) == 2


def test_mnist_validation(mnist_loader):
    """Testing the validation split for MNIST

    Args:
        mnist_loader (DataLoader): MNIST federated DataLoader object.
    """
    test_dataset = mnist_loader.test()
    assert len(test_dataset) == 10000
    assert len(test_dataset.classes) == 10


def test_emnist_digits_iid(emnist_digits_loader):
    """Testing the iid split for EMNIST Digits

    Args:
        mnist_loader (DataLoader): EMNIST Digits federated DataLoader object.
    """
    train_dataset = emnist_digits_loader.train_iid()
    assert len(train_dataset.keys()) == 10
    collated_labels = collate_federated(train_dataset[0])
    assert len(collated_labels) == 24000
    frequency = Counter(collated_labels)
    assert len(frequency.keys()) == 10


def test_emnist_digits_non_iid(emnist_digits_loader):
    """Testing the non-iid split for EMNIST Digits

    Args:
        mnist_loader (DataLoader): EMNIST Digits federated DataLoader object.
    """
    train_dataset = emnist_digits_loader.train_non_iid()
    assert len(train_dataset.keys()) == 10
    collated_labels = collate_federated(train_dataset[0])
    assert len(collated_labels) == 24000
    frequency = Counter(collated_labels)
    assert len(frequency.keys()) == 2


def test_emnist_digits_validation(emnist_digits_loader):
    """Testing the validation split for EMNIST Digits

    Args:
        mnist_loader (DataLoader): EMNIST Digits federated DataLoader object.
    """
    test_dataset = emnist_digits_loader.test()
    assert len(test_dataset) == 40000
    assert len(test_dataset.classes) == 10


def test_cifar10_iid(cifar10_loader):
    """Testing the iid split for CIFAR10

    Args:
        mnist_loader (DataLoader): CIFAR10 federated DataLoader object.
    """
    train_dataset = cifar10_loader.train_iid()
    assert len(train_dataset.keys()) == 10
    collated_labels = collate_federated(train_dataset[0])
    assert len(collated_labels) == 5000
    frequency = Counter(collated_labels)
    assert len(frequency.keys()) == 10


def test_cifar10_non_iid(cifar10_loader):
    """Testing the non-iid split for CIFAR10

    Args:
        mnist_loader (DataLoader): CIFAR10 federated DataLoader object.
    """
    train_dataset = cifar10_loader.train_non_iid()
    assert len(train_dataset.keys()) == 10
    collated_labels = collate_federated(train_dataset[0])
    assert len(collated_labels) == 5000
    frequency = Counter(collated_labels)
    assert len(frequency.keys()) == 2


def test_cifar10_validation(cifar10_loader):
    """Testing the validation split for CIFAR10

    Args:
        mnist_loader (DataLoader): CIFAR10 federated DataLoader object.
    """
    test_dataset = cifar10_loader.test()
    assert len(test_dataset) == 10000
    assert len(test_dataset.classes) == 10


def test_cifar100_iid(cifar100_loader):
    """Testing the iid split for CIFAR100

    Args:
        mnist_loader (DataLoader): CIFAR100 federated DataLoader object.
    """
    train_dataset = cifar100_loader.train_iid()
    assert len(train_dataset.keys()) == 10
    collated_labels = collate_federated(train_dataset[0])
    assert len(collated_labels) == 5000
    frequency = Counter(collated_labels)
    assert len(frequency.keys()) == 100


def test_cifar100_non_iid(cifar100_loader):
    """Testing the non-iid split for CIFAR100

    Args:
        mnist_loader (DataLoader): CIFAR100 federated DataLoader object.
    """
    train_dataset = cifar100_loader.train_non_iid()
    assert len(train_dataset.keys()) == 10
    collated_labels = collate_federated(train_dataset[0])
    assert len(collated_labels) == 5000
    frequency = Counter(collated_labels)
    assert len(frequency.keys()) == 10


def test_cifar100_validation(cifar100_loader):
    """Testing the validation split for CIFAR100

    Args:
        mnist_loader (DataLoader): CIFAR100 federated DataLoader object.
    """
    test_dataset = cifar100_loader.test()
    assert len(test_dataset) == 10000
    assert len(test_dataset.classes) == 100
