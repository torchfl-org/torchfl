#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for AlexNet for EMNIST in `torchfl` package."""
import pytest
from torchvision import datasets, transforms
import os
from torchfl.models.core.emnist.byclass.alexnet import AlexNet
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
def emnist_single_channel_loader():
    """Fixture for EMNIST, single-channel dataset.

    Returns:
        Dataset: PyTorch Dataset object for EMNIST.
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
def emnist_3_channel_loader():
    """Fixture for EMNIST, multi-channel dataset.

    Returns:
        Dataset: PyTorch Dataset object for EMNIST.
    """
    global data_transforms
    return datasets.EMNIST(
        root=os.path.join(os.pardir, "data"),
        train=True,
        split="byclass",
        download=True,
        transform=data_transforms["train_3_channels"],
    )


def test_alexnet_single_channel_ouput_shape(emnist_single_channel_loader):
    """Testing the AlexNet output for a single-channel EMNIST image.

    Args:
        emnist_loader (Dataset): PyTorch Dataset object for EMNIST.
    """
    model = AlexNet(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(emnist_single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_alexnet_3_channels_output_shape(emnist_3_channel_loader):
    """Testing the AlexNet output for a 3-channel EMNIST image.

    Args:
        emnist_loader (Dataset): PyTorch Dataset object for EMNIST.
    """
    model = AlexNet(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(emnist_3_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])
