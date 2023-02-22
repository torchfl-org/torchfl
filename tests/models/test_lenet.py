#!/usr/bin/env python

"""Tests for LeNet in `torchfl` package."""

import torch

from torchfl.models.sota.lenet import LeNet


def test_lenet_single_channel_ouput_shape():
    model = LeNet(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_lenet_three_channel_ouput_shape():
    model = LeNet(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])
