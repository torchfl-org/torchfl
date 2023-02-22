#!/usr/bin/env python

"""Tests for ResNet in `torchfl` package."""

import torch

from torchfl.models.sota.resnet import (
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


def test_resnet18_single_channel_ouput_shape():
    model = ResNet18(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet18_three_channel_ouput_shape():
    model = ResNet18(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet34_single_channel_ouput_shape():
    model = ResNet34(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet34_three_channel_ouput_shape():
    model = ResNet34(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet50_single_channel_ouput_shape():
    model = ResNet50(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet50_three_channel_ouput_shape():
    model = ResNet50(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet101_single_channel_ouput_shape():
    model = ResNet101(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet101_three_channel_ouput_shape():
    model = ResNet101(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet152_single_channel_ouput_shape():
    model = ResNet152(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet152_three_channel_ouput_shape():
    model = ResNet152(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet50_32x4d_single_channel_ouput_shape():
    model = ResNext50_32X4D(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet50_32x4d_three_channel_ouput_shape():
    model = ResNext50_32X4D(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet101_32x8d_single_channel_ouput_shape():
    model = ResNext101_32X8D(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_resnet101_32x8d_three_channel_ouput_shape():
    model = ResNext101_32X8D(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_wideresnet50_2_single_channel_ouput_shape():
    model = WideResNet50_2(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_wideresnet50_2_three_channel_ouput_shape():
    model = WideResNet50_2(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_wideresnet101_2_single_channel_ouput_shape():
    model = WideResNet101_2(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_wideresnet101_2_three_channel_ouput_shape():
    model = WideResNet101_2(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])
