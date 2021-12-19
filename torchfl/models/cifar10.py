#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Initializes the PyTorch models for CIFAR-10 dataset."""

from torch.functional import Tensor
from torchfl.models.abstract import TorchModel
from torch.nn import (
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    Dropout2d,
    Dropout,
    Linear,
)
import torch
from torch.nn.functional import relu, avg_pool2d
from typing import List


class CNN(TorchModel):
    """Implementation of CNN for CIFAR10."""

    def __init__(self) -> None:
        """Constructor"""
        super(CNN, self).__init__()
        self.conv_model: Sequential = Sequential(
            Conv2d(3, 32, 3),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 128, 3, 1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 128, 3, 1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout2d(p=0.05),
            Conv2d(128, 256, 3, 1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, 3, 1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_model: Sequential = Sequential(
            Dropout(p=0.1),
            Linear(2304, 1024),
            ReLU(inplace=True),
            Linear(1024, 512),
            ReLU(inplace=True),
            Dropout(p=0.1),
            Linear(512, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation through the network.

        Args:
            x (Tensor): The input image.

        Returns:
            Tensor: The output tensor after forward propagation.
        """
        x = self.conv_model(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        return self.linear_model(x)


class Bottleneck(TorchModel):
    """Implementation of Bottleneck for CIFAR10."""

    def __init__(
        self,
        last_planes: int,
        in_planes: int,
        out_planes: int,
        dense_depth: int,
        stride: int,
        first_layer: bool,
    ) -> None:
        """Constructor

        Args:
            last_planes (int): last planes in the bottleneck.
            in_planes (int): incoming planes in the bottleneck.
            out_planes (int): outgoing planes in the bottleneck.
            dense_depth (int): depth of the layers in the bottleneck.
            stride (int): strides used for the convolutional layers.
            first_layer (bool): if the layer is the first one.
        """
        super(Bottleneck, self).__init__()
        self.out_planes: int = out_planes
        self.dense_depth: int = dense_depth
        self.conv_model: Sequential = Sequential(
            Conv2d(last_planes, in_planes, 1, False),
            BatchNorm2d(in_planes),
            ReLU(inplace=True),
            Conv2d(in_planes, in_planes, 3, stride, 1, 32, False),
            BatchNorm2d(in_planes),
            ReLU(inplace=True),
            Conv2d(in_planes, out_planes + dense_depth, 1, False),
            BatchNorm2d(out_planes + dense_depth),
        )
        self.shortcut: Sequential = Sequential()
        if first_layer:
            self.shortcut = Sequential(
                Conv2d(last_planes, out_planes + dense_depth, 1, stride, False),
                BatchNorm2d(out_planes + dense_depth),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation through the network.

        Args:
            x (Tensor): The input image.

        Returns:
            Tensor: The output tensor after forward propagation.
        """
        out: Tensor = self.conv_model(x)
        x = self.shortcut(x)
        d: int = self.out_planes
        out = torch.cat(
            [x[:, :d, :, :] + out[:, :d, :, :], x[:, d:, :, :], out[:, d:, :, :]], 1
        )
        return relu(out)


class DPN26(TorchModel):
    """Implementation of DPN26 for CIFAR10."""

    def __init__(self) -> None:
        """Constructor"""
        super(DPN26, self).__init__()
        in_planes, out_planes = (96, 192, 384, 768), (256, 512, 1024, 2048)
        num_blocks, dense_depth = (2, 2, 2, 2), (16, 32, 24, 128)

        self.last_planes: int = 64
        self.conv_model: Sequential = Sequential(
            Conv2d(3, 64, 3, 1, 1, False), BatchNorm2d(64), ReLU(inplace=True)
        )
        self.layer1: Sequential = self._make_layer(
            in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1
        )
        self.layer2: Sequential = self._make_layer(
            in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2
        )
        self.layer3: Sequential = self._make_layer(
            in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2
        )
        self.layer4: Sequential = self._make_layer(
            in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2
        )
        self.linear: Linear = Linear(
            out_planes[3] + (num_blocks[3] + 1) * dense_depth[3], 10
        )

    def _make_layer(
        self,
        in_planes: int,
        out_planes: int,
        num_blocks: int,
        dense_depth: int,
        stride: int,
    ) -> Sequential:
        """Helper method for creating layers

        Args:
            in_planes (int): incoming planes for a layer.
            out_planes (int): outgoing planes for a layer.
            num_blocks (int): number of blocks in the layer.
            dense_depth (int): depth of the layers.
            stride (int): stride used in the convolutional layer.

        Returns:
            Sequential: PyTorch Sequential model generated after the processing.
        """
        strides: List[int] = [stride] + [1] * (num_blocks - 1)
        layers: List[Bottleneck] = list()
        for i, stride in enumerate(strides):
            layers.append(
                Bottleneck(
                    self.last_planes, in_planes, out_planes, dense_depth, stride, i == 0
                )
            )
            self.last_planes = out_planes + (i + 2) * dense_depth
        return Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation through the network.

        Args:
            x (Tensor): The input image.

        Returns:
            Tensor: The output tensor after forward propagation.
        """
        x = self.conv_model(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class DPN92(TorchModel):
    """Implementation of DPN92 for CIFAR10."""

    def __init__(self) -> None:
        """Constructor"""
        super(DPN92, self).__init__()
        in_planes, out_planes = (96, 192, 384, 768), (256, 512, 1024, 2048)
        num_blocks, dense_depth = (3, 4, 20, 3), (16, 32, 24, 128)

        self.last_planes: int = 64
        self.conv_model: Sequential = Sequential(
            Conv2d(3, 64, 3, 1, 1, False), BatchNorm2d(64), ReLU(inplace=True)
        )
        self.layer1: Sequential = self._make_layer(
            in_planes[0], out_planes[0], num_blocks[0], dense_depth[0], stride=1
        )
        self.layer2: Sequential = self._make_layer(
            in_planes[1], out_planes[1], num_blocks[1], dense_depth[1], stride=2
        )
        self.layer3: Sequential = self._make_layer(
            in_planes[2], out_planes[2], num_blocks[2], dense_depth[2], stride=2
        )
        self.layer4: Sequential = self._make_layer(
            in_planes[3], out_planes[3], num_blocks[3], dense_depth[3], stride=2
        )
        self.linear: Linear = Linear(
            out_planes[3] + (num_blocks[3] + 1) * dense_depth[3], 10
        )

    def _make_layer(
        self,
        in_planes: int,
        out_planes: int,
        num_blocks: int,
        dense_depth: int,
        stride: int,
    ) -> Sequential:
        """Helper method for creating layers

        Args:
            in_planes (int): incoming planes for a layer.
            out_planes (int): outgoing planes for a layer.
            num_blocks (int): number of blocks in the layer.
            dense_depth (int): depth of the layers.
            stride (int): stride used in the convolutional layer.

        Returns:
            Sequential: PyTorch Sequential model generated after the processing.
        """
        strides: List[int] = [stride] + [1] * (num_blocks - 1)
        layers: List[Bottleneck] = list()
        for i, stride in enumerate(strides):
            layers.append(
                Bottleneck(
                    self.last_planes, in_planes, out_planes, dense_depth, stride, i == 0
                )
            )
            self.last_planes = out_planes + (i + 2) * dense_depth
        return Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation through the network.

        Args:
            x (Tensor): The input image.

        Returns:
            Tensor: The output tensor after forward propagation.
        """
        x = self.conv_model(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        return self.linear(x)
