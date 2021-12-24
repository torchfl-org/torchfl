#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for models package in `torchfl` package."""
import pytest
from torchfl.dataloader import FLDataLoader
from torchfl.models import mnist, emnist_digits, cifar10
from torch import Size, reshape


@pytest.fixture
def mnist_loader():
    """Fixture for MNIST dataset.

    Returns:
        Dataset: PyTorch Dataset object for MNIST.
    """
    return FLDataLoader.load_dataset("mnist", True)


@pytest.fixture
def emnist_digits_loader():
    """Fixture for EMNIST digits dataset.

    Returns:
        Dataset: PyTorch Dataset object for EMNIST digits.
    """
    return FLDataLoader.load_dataset("emnist_digits", True)


@pytest.fixture
def cifar10_loader():
    """Fixture for CIFAR10 dataset.

    Returns:
        Dataset: PyTorch Dataset object for CIFAR10.
    """
    return FLDataLoader.load_dataset("cifar10", True)


def test_mnist_mlp(mnist_loader):
    """Testing the MLP output for a single MNIST image.

    Args:
        mnist_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = mnist.MLP()
    model.zero_grad()
    out = model(reshape(mnist_loader[0][0], (1, 1, 28, 28)))
    assert out.size() == Size([1, 10])


def test_mnist_cnn(mnist_loader):
    """Testing the CNN output for a single MNIST image.

    Args:
        mnist_loader (DataLoader): PyTorch Dataset object for MNIST.
    """
    model = mnist.CNN()
    model.zero_grad()
    out = model(reshape(mnist_loader[0][0], (1, 1, 28, 28)))
    assert out.size() == Size([1, 10])


def test_emnist_digits_mlp(emnist_digits_loader):
    """Testing the MLP output for a single EMNIST digits image.

    Args:
        emnist_digits_loader (DataLoader): PyTorch Dataset object for EMNIST digits.
    """
    model = emnist_digits.MLP()
    model.zero_grad()
    out = model(reshape(emnist_digits_loader[0][0], (1, 1, 28, 28)))
    assert out.size() == Size([1, 10])


def test_emnist_digits_cnn(emnist_digits_loader):
    """Testing the CNN output for a single EMNIST digits image.

    Args:
        emnist_digits_loader (DataLoader): PyTorch Dataset object for EMNIST digits.
    """
    model = emnist_digits.CNN()
    model.zero_grad()
    out = model(reshape(emnist_digits_loader[0][0], (1, 1, 28, 28)))
    assert out.size() == Size([1, 10])


def test_cifar10_cnn(cifar10_loader):
    """Testing the CNN output for a single CIFAR10 image.

    Args:
        cifar10_loader (DataLoader): PyTorch Dataset object for CIFAR10.
    """
    model = cifar10.CNN()
    model.zero_grad()
    out = model(reshape(cifar10_loader[0][0], (1, 3, 32, 32)))
    assert out.size() == Size([1, 10])


def test_cifar10_dpn26(cifar10_loader):
    """Testing the DPN26 output for a single CIFAR10 image.

    Args:
        cifar10_loader (DataLoader): PyTorch Dataset object for CIFAR10.
    """
    model = cifar10.DPN26()
    model.zero_grad()
    out = model(reshape(cifar10_loader[0][0], (1, 3, 32, 32)))
    assert out.size() == Size([1, 10])


def test_cifar10_dpn92(cifar10_loader):
    """Testing the DPN92 output for a single CIFAR10 image.

    Args:
        cifar10_loader (DataLoader): PyTorch Dataset object for CIFAR10.
    """
    model = cifar10.DPN92()
    model.zero_grad()
    out = model(reshape(cifar10_loader[0][0], (1, 3, 32, 32)))
    assert out.size() == Size([1, 10])


def test_cifar10_vgg19(cifar10_loader):
    """Testing the VGG19 output for a single CIFAR10 image.

    Args:
        cifar10_loader (DataLoader): PyTorch Dataset object for CIFAR10.
    """
    model = cifar10.VGG19()
    model.zero_grad()
    out = model(reshape(cifar10_loader[0][0], (1, 3, 32, 32)))
    assert out.size() == Size([1, 10])


def test_cifar10_resnet18(cifar10_loader):
    """Testing the ResNet18 output for a single CIFAR10 image.

    Args:
        cifar10_loader (DataLoader): PyTorch Dataset object for CIFAR10.
    """
    model = cifar10.ResNet18()
    model.zero_grad()
    out = model(reshape(cifar10_loader[0][0], (1, 3, 32, 32)))
    assert out.size() == Size([1, 10])


def test_cifar10_resnet34(cifar10_loader):
    """Testing the ResNet34 output for a single CIFAR10 image.

    Args:
        cifar10_loader (DataLoader): PyTorch Dataset object for CIFAR10.
    """
    model = cifar10.ResNet34()
    model.zero_grad()
    out = model(reshape(cifar10_loader[0][0], (1, 3, 32, 32)))
    assert out.size() == Size([1, 10])


def test_cifar10_resnet50(cifar10_loader):
    """Testing the ResNet50 output for a single CIFAR10 image.

    Args:
        cifar10_loader (DataLoader): PyTorch Dataset object for CIFAR10.
    """
    model = cifar10.ResNet50()
    model.zero_grad()
    out = model(reshape(cifar10_loader[0][0], (1, 3, 32, 32)))
    assert out.size() == Size([1, 10])


def test_cifar10_resnet101(cifar10_loader):
    """Testing the ResNet101 output for a single CIFAR10 image.

    Args:
        cifar10_loader (DataLoader): PyTorch Dataset object for CIFAR10.
    """
    model = cifar10.ResNet101()
    model.zero_grad()
    out = model(reshape(cifar10_loader[0][0], (1, 3, 32, 32)))
    assert out.size() == Size([1, 10])


def test_cifar10_resnet152(cifar10_loader):
    """Testing the ResNet152 output for a single CIFAR10 image.

    Args:
        cifar10_loader (DataLoader): PyTorch Dataset object for CIFAR10.
    """
    model = cifar10.ResNet152()
    model.zero_grad()
    out = model(reshape(cifar10_loader[0][0], (1, 3, 32, 32)))
    assert out.size() == Size([1, 10])
