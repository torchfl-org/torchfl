#!/usr/bin/env python

"""Tests for MLP in `torchfl` package."""

import torch

from torchfl.models.sota.mlp import MLP


def test_mlp_single_channel_ouput_shape():
    model = MLP(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 28, 28))
    assert out.size() == torch.Size([1, 10])


def test_mlp_three_channel_ouput_shape():
    model = MLP(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 28, 28))
    assert out.size() == torch.Size([1, 10])
