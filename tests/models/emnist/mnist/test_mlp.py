#!/usr/bin/env python

"""Tests for LeNet in `torchfl` package."""
import os

import pytest
import torch
from torchvision import datasets, transforms

from torchfl.compatibility import TORCHFL_DIR
from torchfl.models.core.emnist.mnist.mlp import MLP

data_transforms = {
    "train_single_channel": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ]
    ),
    "train_3_channels": transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


@pytest.fixture()
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
        split="mnist",
        transform=data_transforms["train_single_channel"],
    )


@pytest.fixture()
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
        split="mnist",
        transform=data_transforms["train_3_channels"],
    )


def test_single_channel_ouput_shape(single_channel_loader):
    """Testing the AlexNet output for a single-channel image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = MLP(num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 28, 28)))
    assert out.size() == torch.Size([1, 10])


def test_3_channels_output_shape(three_channel_loader):
    """Testing the output for a 3-channel FashionMNIST image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = MLP(num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 28, 28)))
    assert out.size() == torch.Size([1, 10])
