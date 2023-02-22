#!/usr/bin/env python

"""Tests for SqueezeNet in `torchfl` package."""

import torch

from torchfl.models.sota.squeezenet import SqueezeNet1_0, SqueezeNet1_1


def test_squeezenet1_0_single_channel_ouput_shape():
    model = SqueezeNet1_0(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_squeezenet1_0_three_channel_ouput_shape():
    model = SqueezeNet1_0(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_squeezenet1_1_single_channel_ouput_shape():
    model = SqueezeNet1_1(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_squeezenet1_1_three_channel_ouput_shape():
    model = SqueezeNet1_1(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])
