#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for models package in `torchfl` package."""
import pytest
from torchfl import dataloader
from torchfl.models import mnist


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
