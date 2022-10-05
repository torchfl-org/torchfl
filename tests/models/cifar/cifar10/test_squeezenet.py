#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for SqueezeNet in `torchfl` package."""
import pytest
from torchvision import datasets, transforms
import os
from torchfl.compatibility import TORCHFL_DIR
from torchfl.models.core.cifar.cifar10.squeezenet import SqueezeNet1_0, SqueezeNet1_1
import torch

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
}


@pytest.fixture
def three_channel_loader():
    """Fixture for multi-channel dataset.

    Returns:
        Dataset: PyTorch Dataset object.
    """
    global data_transforms
    return datasets.CIFAR10(
        root=os.path.join(TORCHFL_DIR, "data"),
        train=True,
        download=True,
        transform=data_transforms["train"],
    )


def test_squeezenet10_3_channels_output_shape(three_channel_loader):
    """Testing the output for a 3-channel image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = SqueezeNet1_0(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 10])


def test_squeezenet11_3_channels_output_shape(three_channel_loader):
    """Testing the output for a 3-channel image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = SqueezeNet1_1(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 10])
