#!/usr/bin/env python

"""Tests for AlexNet in `torchfl` package."""
import torch

from torchfl.models.sota.alexnet import AlexNet


def test_alexnet_single_channel_output_shape():
    model = AlexNet(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_alexnet_three_channel_output_shape():
    model = AlexNet(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])
