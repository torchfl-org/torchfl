#!/usr/bin/env python

"""Tests for ShuffleNetv2 in `torchfl` package."""

import torch

from torchfl.models.sota.shufflenetv2 import (
    ShuffleNetv2_x0_5,
    ShuffleNetv2_x1_0,
    ShuffleNetv2_x1_5,
    ShuffleNetv2_x2_0,
)


def test_shufflenetv2_x0_5_single_channel_ouput_shape():
    model = ShuffleNetv2_x0_5(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_shufflenetv2_x0_5_three_channel_ouput_shape():
    model = ShuffleNetv2_x0_5(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_shufflenetv2_x1_0_single_channel_ouput_shape():
    model = ShuffleNetv2_x1_0(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_shufflenetv2_x1_0_three_channel_ouput_shape():
    model = ShuffleNetv2_x1_0(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_shufflenetv2_x1_5_single_channel_ouput_shape():
    model = ShuffleNetv2_x1_5(num_channels=1, pre_trained=False)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_shufflenetv2_x1_5_three_channel_ouput_shape():
    model = ShuffleNetv2_x1_5(num_channels=3, pre_trained=False)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_shufflenetv2_x2_0_single_channel_ouput_shape():
    model = ShuffleNetv2_x2_0(num_channels=1, pre_trained=False)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_shufflenetv2_x2_0_three_channel_ouput_shape():
    model = ShuffleNetv2_x2_0(num_channels=3, pre_trained=False)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])
