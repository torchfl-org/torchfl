#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for DenseNet in `torchfl` package."""
import pytest
from torchvision import datasets, transforms
import os
from torchfl.models.core.cifar.cifar10.lenet import LeNet
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
        root=os.path.join(os.pardir, "data"),
        train=True,
        download=True,
        transform=data_transforms["train"],
    )


def test_densenet121_3_channels_output_shape(three_channel_loader):
    """Testing the output for a 3-channel image.

    Args:
        loader (Dataset): PyTorch Dataset object.
    """
    model = LeNet(num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 10])
