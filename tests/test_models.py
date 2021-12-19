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


def test_mnist_mlp():
    return


def test_mnist_cnn():
    return


def test_emnist_digits_mlp():
    return


def test_emnist_digits_cnn():
    return


def test_cifar10_cnn():
    return


def test_cifar10_dpn26():
    return


def test_cifar10_dpn92():
    return


def test_cifar10_vgg19():
    return


def test_cifar10_resnet18():
    return


def test_cifar10_resnet34():
    return


def test_cifar10_resnet50():
    return


def test_cifar10_resnet101():
    return


def test_cifar10_resnet152():
    return
