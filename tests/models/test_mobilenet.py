#!/usr/bin/env python

"""Tests for MobileNet in `torchfl` package."""

import torch

from torchfl.models.sota.mobilenet import (
    MobileNetV2,
    MobileNetV3Large,
    MobileNetV3Small,
)


def test_mobilenetv2_single_channel_ouput_shape():
    model = MobileNetV2(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_mobilenetv2_three_channel_ouput_shape():
    model = MobileNetV2(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_mobilenetv3large_single_channel_ouput_shape():
    model = MobileNetV3Large(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_mobilenetv3large_three_channel_ouput_shape():
    model = MobileNetV3Large(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_mobilenetv3small_single_channel_ouput_shape():
    model = MobileNetV3Small(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_mobilenetv3small_three_channel_ouput_shape():
    model = MobileNetV3Small(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])
