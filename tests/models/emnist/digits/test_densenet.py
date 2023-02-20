#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for DenseNet in `torchfl` package."""
import os

import pytest
import torch
from torchvision import datasets
from torchvision import transforms

from torchfl.compatibility import TORCHFL_DIR
from torchfl.models.core.emnist.digits.densenet import DenseNet121
from torchfl.models.core.emnist.digits.densenet import DenseNet161
from torchfl.models.core.emnist.digits.densenet import DenseNet169
from torchfl.models.core.emnist.digits.densenet import DenseNet201

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
def single_channel_loader():
    """Fixture for single-channel dataset.

    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(TORCHFL_DIR, "data"),
        train=True,
        download=True,
        split="digits",
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture
def three_channel_loader():
    """Fixture for multi-channel dataset.

    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(TORCHFL_DIR, "data"),
        train=True,
        download=True,
        split="digits",
        transform=data_transforms["train_3_channels"],
    )


def test_densenet121_single_channel_ouput_shape(single_channel_loader):
    """Testing the output for a single-channel FashionMNIST image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = DenseNet121(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 10])


def test_densenet121_3_channels_output_shape(three_channel_loader):
    """Testing the output for a 3-channel image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = DenseNet121(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 10])


def test_densenet161_single_channel_ouput_shape(single_channel_loader):
    """Testing the output for a single-channel image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = DenseNet161(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 10])


def test_densenet161_3_channels_output_shape(three_channel_loader):
    """Testing the output for a 3-channel image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = DenseNet161(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 10])


def test_densenet169_single_channel_ouput_shape(single_channel_loader):
    """Testing the output for a single-channel image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = DenseNet169(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 10])


def test_densenet169_3_channels_output_shape(three_channel_loader):
    """Testing the output for a 3-channel image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = DenseNet169(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 10])


def test_densenet201_single_channel_ouput_shape(single_channel_loader):
    """Testing the output for a single-channel image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = DenseNet201(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 10])


def test_densenet201_3_channels_output_shape(three_channel_loader):
    """Testing the output for a 3-channel image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = DenseNet201(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 10])
