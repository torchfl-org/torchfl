#!/usr/bin/env python

"""Tests for VGG in `torchfl` package."""

import torch

from torchfl.models.sota.vgg import (
    VGG11,
    VGG11_BN,
    VGG13,
    VGG13_BN,
    VGG16,
    VGG16_BN,
    VGG19,
    VGG19_BN,
)


def test_vgg11_single_channel_ouput_shape():
    model = VGG11(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg11_three_channel_ouput_shape():
    model = VGG11(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg11_bn_single_channel_ouput_shape():
    model = VGG11_BN(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg11_bn_three_channel_ouput_shape():
    model = VGG11_BN(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg13_single_channel_ouput_shape():
    model = VGG13(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg13_three_channel_ouput_shape():
    model = VGG13(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg13_bn_single_channel_ouput_shape():
    model = VGG13_BN(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg13_bn_three_channel_ouput_shape():
    model = VGG13_BN(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg16_single_channel_ouput_shape():
    model = VGG16(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg16_three_channel_ouput_shape():
    model = VGG16(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg16_bn_single_channel_ouput_shape():
    model = VGG16_BN(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg16_bn_three_channel_ouput_shape():
    model = VGG16_BN(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg19_single_channel_ouput_shape():
    model = VGG19(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg19_three_channel_ouput_shape():
    model = VGG19(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg19_bn_single_channel_ouput_shape():
    model = VGG19_BN(num_channels=1)
    model.zero_grad()
    out = model(torch.randn(1, 1, 224, 224))
    assert out.size() == torch.Size([1, 10])


def test_vgg19_bn_three_channel_ouput_shape():
    model = VGG19_BN(num_channels=3)
    model.zero_grad()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.size() == torch.Size([1, 10])
