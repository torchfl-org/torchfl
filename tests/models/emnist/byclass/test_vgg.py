#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for VGG for EMNIST in `torchfl` package."""
import pytest
from torchvision import datasets, transforms
import os
from torchfl.models.core.emnist.byclass.vgg import (
    VGG11,
    VGG11_BN,
    VGG13,
    VGG13_BN,
    VGG16,
    VGG16_BN,
    VGG19,
    VGG19_BN,
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


def test_vgg11_single_channel_ouput_shape(single_channel_loader):
    """Testing the VGG output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG11(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg11_3_channel_ouput_shape(three_channel_loader):
    """Testing the VGG output for a three-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG11(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg11bn_single_channel_ouput_shape(single_channel_loader):
    """Testing the VGG output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG11_BN(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg11bn_3_channel_ouput_shape(three_channel_loader):
    """Testing the VGG output for a three-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG11_BN(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg13_single_channel_ouput_shape(single_channel_loader):
    """Testing the VGG output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG13(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg13_3_channel_ouput_shape(three_channel_loader):
    """Testing the VGG output for a three-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG13(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg13bn_single_channel_ouput_shape(single_channel_loader):
    """Testing the VGG output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG13_BN(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg13bn_3_channel_ouput_shape(three_channel_loader):
    """Testing the VGG output for a three-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG13_BN(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg16_single_channel_ouput_shape(single_channel_loader):
    """Testing the VGG output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG16(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg16_3_channel_ouput_shape(three_channel_loader):
    """Testing the VGG output for a three-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG16(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg16bn_single_channel_ouput_shape(single_channel_loader):
    """Testing the VGG output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG16_BN(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg16bn_3_channel_ouput_shape(three_channel_loader):
    """Testing the VGG output for a three-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG16_BN(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg19_single_channel_ouput_shape(single_channel_loader):
    """Testing the VGG output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG19(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg19_3_channel_ouput_shape(three_channel_loader):
    """Testing the VGG output for a three-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG19(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg19bn_single_channel_ouput_shape(single_channel_loader):
    """Testing the VGG output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG19_BN(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 62])


def test_vgg19bn_3_channel_ouput_shape(three_channel_loader):
    """Testing the VGG output for a three-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = VGG19_BN(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 62])
