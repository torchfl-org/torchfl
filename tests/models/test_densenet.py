#!/usr/bin/env python

"""Tests for DenseNet in `torchfl` package."""

import torch

from torchfl.models.sota.densenet import (
    DenseNet121,
    DenseNet161,
    DenseNet169,
    DenseNet201,
)


def test_densenet121_single_channel_ouput_shape():
    model = DenseNet121(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_densenet121_three_channel_ouput_shape():
    model = DenseNet121(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_densenet161_single_channel_ouput_shape():
    model = DenseNet161(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_densenet161_three_channel_ouput_shape():
    model = DenseNet161(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_densenet169_single_channel_ouput_shape():
    model = DenseNet169(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_densenet169_three_channel_ouput_shape():
    model = DenseNet169(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_densenet201_single_channel_ouput_shape():
    model = DenseNet201(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_densenet201_three_channel_ouput_shape():
    model = DenseNet201(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])
