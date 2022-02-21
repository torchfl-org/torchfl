#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for ShuffleNetV2 for EMNIST in `torchfl` package."""
import pytest
from torchvision import datasets, transforms
import os
from torchfl.models.core.emnist.byclass.shufflenetv2 import (
    ShuffleNetv2_x0_5,
    ShuffleNetv2_x1_0,
    ShuffleNetv2_x1_5,
    ShuffleNetv2_x2_0,
)
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
def single_channel_loader():
    """Fixture for single-channel dataset.

    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        download=True,
        split="byclass",
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
        root=os.path.join(os.pardir, "data"),
        train=True,
        download=True,
        split="byclass",
        transform=data_transforms["train_3_channels"],
    )


def test_shufflenetv2_x05_single_channel_ouput_shape(single_channel_loader):
    """Testing the ShuffleNetV2 output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ShuffleNetv2_x0_5(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_shufflenetv2_x05_3_channels_output_shape(three_channel_loader):
    """Testing the ShuffleNetV2 output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ShuffleNetv2_x0_5(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_shufflenetv2_x10_single_channel_ouput_shape(single_channel_loader):
    """Testing the ShuffleNetV2 output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ShuffleNetv2_x1_0(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_shufflenetv2_x10_3_channels_output_shape(three_channel_loader):
    """Testing the ShuffleNetV2 output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ShuffleNetv2_x1_0(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_shufflenetv2_x15_single_channel_ouput_shape(single_channel_loader):
    """Testing the ShuffleNetV2 output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ShuffleNetv2_x1_5(pre_trained=False, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_shufflenetv2_x15_3_channels_output_shape(three_channel_loader):
    """Testing the ShuffleNetV2 output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ShuffleNetv2_x1_5(pre_trained=False, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_shufflenetv2_x20_single_channel_ouput_shape(single_channel_loader):
    """Testing the ShuffleNetV2 output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ShuffleNetv2_x2_0(pre_trained=False, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_shufflenetv2_x20_3_channels_output_shape(three_channel_loader):
    """Testing the ShuffleNetV2 output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ShuffleNetv2_x2_0(pre_trained=False, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])
