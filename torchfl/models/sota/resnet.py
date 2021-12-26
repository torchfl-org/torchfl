#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the general ResNet architecture using PyTorch."""

import torch.nn as nn
from types import SimpleNamespace
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME, RESNET_BLOCKS_BY_NAME


class ResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """Constructor

        Args:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                stride=1 if not subsample else 2,
                bias=False,
            ),  # No bias needed as the Batch Norm handles it
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = (
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        )
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out


class PreActResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """Constructor

        Args:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=3,
                padding=1,
                stride=1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        )

        # 1x1 convolution needs to apply non-linearity as well as not done on skip connection
        self.downsample = (
            nn.Sequential(
                nn.BatchNorm2d(c_in),
                act_fn(),
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False),
            )
            if subsample
            else None
        )

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        num_channels=3,
        num_blocks=[3, 3, 3],
        c_hidden=[16, 32, 64],
        act_fn_name="relu",
        block_name="ResNetBlock",
        **kwargs
    ):
        """Constructor

        Args:
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            num_channels (int, optional): Number of channels for the images in the dataset. Defaults to 3.
            num_blocks (list, optional): List with the number of ResNet blocks to use. The first block of each group uses downsampling, except the first. Defaults to [3, 3, 3].
            c_hidden (list, optional): List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go. Defaults to [16, 32, 64].
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
            block_name (str, optional): Type of ResNet block to be used. Defaults to "ResNetBlock". Accepted: ["ResNetBlock", "PreActResNetBlock"].
        """
        super().__init__()
        assert block_name in RESNET_BLOCKS_BY_NAME
        self.hparams = SimpleNamespace(
            model_name="resnet",
            block_type=block_name,
            num_classes=num_classes,
            num_channels=num_channels,
            c_hidden=c_hidden,
            num_blocks=num_blocks,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
            block_class=RESNET_BLOCKS_BY_NAME[block_name],
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        c_hidden = self.hparams.c_hidden

        # A first convolution on the original image to scale up the channel size
        if (
            self.hparams.block_class == PreActResNetBlock
        ):  # => Don't apply non-linearity on output
            self.input_net = nn.Sequential(
                nn.Conv2d(
                    self.hparams.num_channels,
                    c_hidden[0],
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
        else:
            self.input_net = nn.Sequential(
                nn.Conv2d(
                    self.hparams.num_channels,
                    c_hidden[0],
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(c_hidden[0]),
                self.hparams.act_fn(),
            )

        # Creating the ResNet blocks
        blocks = []
        for block_idx, block_count in enumerate(self.hparams.num_blocks):
            for bc in range(block_count):
                subsample = (
                    bc == 0 and block_idx > 0
                )  # Subsample the first block of each group, except the very first one.
                blocks.append(
                    self.hparams.block_class(
                        c_in=c_hidden[block_idx if not subsample else (block_idx - 1)],
                        act_fn=self.hparams.act_fn,
                        subsample=subsample,
                        c_out=c_hidden[block_idx],
                    )
                )
        self.blocks = nn.Sequential(*blocks)

        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden[-1], self.hparams.num_classes),
        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        # Fan-out focuses on the gradient distribution, and is commonly used in ResNets
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=self.hparams.act_fn_name
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x
