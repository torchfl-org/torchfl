#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for MobileNet for EMNIST balanced in `torchfl` package."""
import pytest
from torchvision import datasets, transforms
import os
from torchfl.models.core.emnist.bymerge.resnet import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNext50_32X4D,
    ResNext101_32X8D,
    WideResNet50_2,
    WideResNet101_2,
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
        split="bymerge",
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
        split="bymerge",
        transform=data_transforms["train_3_channels"],
    )


def test_resnet18_single_channel_ouput_shape(single_channel_loader):
    """Testing the ResNet output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNet18(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet18_3_channels_output_shape(three_channel_loader):
    """Testing the ResNet output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNet18(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet34_single_channel_ouput_shape(single_channel_loader):
    """Testing the ResNet output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNet34(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet34_3_channels_output_shape(three_channel_loader):
    """Testing the ResNet output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNet34(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet50_single_channel_ouput_shape(single_channel_loader):
    """Testing the ResNet output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNet50(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet50_3_channels_output_shape(three_channel_loader):
    """Testing the ResNet output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNet50(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet101_single_channel_ouput_shape(single_channel_loader):
    """Testing the ResNet output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNet101(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet101_3_channels_output_shape(three_channel_loader):
    """Testing the ResNet output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNet101(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet152_single_channel_ouput_shape(single_channel_loader):
    """Testing the ResNet output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNet152(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet152_3_channels_output_shape(three_channel_loader):
    """Testing the ResNet output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNet152(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet5032x4d_single_channel_ouput_shape(single_channel_loader):
    """Testing the ResNet output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNext50_32X4D(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet5032x4d_3_channels_output_shape(three_channel_loader):
    """Testing the ResNet output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNext50_32X4D(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet10132x8d_single_channel_ouput_shape(single_channel_loader):
    """Testing the ResNet output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNext101_32X8D(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_resnet10132x8d_3_channels_output_shape(three_channel_loader):
    """Testing the ResNet output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = ResNext101_32X8D(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_wideresnet50_2_single_channel_ouput_shape(single_channel_loader):
    """Testing the ResNet output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = WideResNet50_2(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_wideresnet50_2_3_channels_output_shape(three_channel_loader):
    """Testing the ResNet output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = WideResNet50_2(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_wideresnet101_2_single_channel_ouput_shape(single_channel_loader):
    """Testing the ResNet output for a single-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = WideResNet101_2(pre_trained=True, feature_extract=True, num_channels=1)
    model.zero_grad()
    out = model(torch.reshape(single_channel_loader[0][0], (1, 1, 224, 224)))
    assert out.size() == torch.Size([1, 47])


def test_wideresnet101_2_3_channels_output_shape(three_channel_loader):
    """Testing the ResNet output for a 3-channel EMNIST image.

    Args:
        EMNIST_loader (Dataset): PyTorch Dataset object for MNIST.
    """
    model = WideResNet101_2(pre_trained=True, feature_extract=True, num_channels=3)
    model.zero_grad()
    out = model(torch.reshape(three_channel_loader[0][0], (1, 3, 224, 224)))
    assert out.size() == torch.Size([1, 47])
