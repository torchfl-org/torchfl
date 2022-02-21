#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for SqueezeNet for FashionMNIST in `torchfl` package."""
import pytest
from torchvision import datasets, transforms
import os
from torchfl.models.core.fashionmnist.squeezenet import SqueezeNet1_0, SqueezeNet1_1
import torch

data_transforms = {
    "train_single_channel": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ]
    ),
    "train_3_channels": transforms.Compose(
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
def fashionmnist_single_channel_loader():
    """Fixture for FashionMNIST, single-channel dataset.

    Returns:
        Dataset: PyTorch Dataset object for FashionMNIST.
    """
    global data_transforms
    return datasets.FashionMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        download=True,
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def fashionmnist_3_channel_loader():
    """Fixture for FashionMNIST, multi-channel dataset.

    Returns:
        Dataset: PyTorch Dataset object for FashionMNIST.
    """
    global data_transforms
    return datasets.FashionMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        download=True,
        transform=data_transforms["train_3_channels"],
    )


def test_squeezenet_10_single_channel_ouput_shape(fashionmnist_single_channel_loader):
    """Testing the SqueezeNet output for a single-channel FashionMNIST image.

    Args:
        fashionmnist_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = SqueezeNet1_0(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(
        torch.reshape(fashionmnist_single_channel_loader[0][0], (1, 1, 224, 224))
    )
    assert out.size() == torch.Size([1, 10])


def test_squeezenet_10_3_channels_output_shape(fashionmnist_3_channel_loader):
    """Testing the SqueezeNet output for a 3-channel FashionMNIST image.

    Args:
        fashionmnist_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = SqueezeNet1_0(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(fashionmnist_3_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 10])


def test_squeezenet_11_single_channel_ouput_shape(fashionmnist_single_channel_loader):
    """Testing the SqueezeNet output for a single-channel FashionMNIST image.

    Args:
        fashionmnist_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = SqueezeNet1_1(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(
        torch.reshape(fashionmnist_single_channel_loader[0][0], (1, 1, 224, 224))
    )
    assert out.size() == torch.Size([1, 10])


def test_squeezenet_11_3_channels_output_shape(fashionmnist_3_channel_loader):
    """Testing the SqueezeNet output for a 3-channel FashionMNIST image.

    Args:
        fashionmnist_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = SqueezeNet1_1(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(fashionmnist_3_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 10])
