#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for AlexNet for CIFAR in `torchfl` package."""
import pytest
from torchvision import datasets, transforms
import os
from torchfl.models.core.cifar.cifar100.alexnet import AlexNet
import torch

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ]
    )
}


@pytest.fixture
def cifar100_loader():
    """Fixture for CIFAR100, single-channel dataset.

    Returns:
        Dataset: PyTorch Dataset object for CIFAR100.
    """
    global data_transforms
    return datasets.CIFAR100(
        root=os.path.join(os.pardir, "data"),
        train=True,
        download=True,
        transform=data_transforms["train"],
    )


def test_alexnet_ouput_shape(cifar100_loader):
    """Testing the AlexNet output for a single-channel CIFAR100 image.

    Args:
        fashionmnist_loader (Dataset): PyTorch Dataset object for CIFAR100.
    """
    model = AlexNet(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(cifar100_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 100])
