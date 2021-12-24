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
    AvgPool2d,
)
import torch
from torch.nn.functional import relu, avg_pool2d
from typing import List, Any, Union, Type


class CNN(TorchModel):
    """Implementation of CNN for CIFAR10."""

    def __init__(self) -> None:
        """Constructor"""
        super(CNN, self).__init__()
        self.conv_model: Sequential = Sequential(
            Conv2d(3, 64, kernel_size=3),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 128, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout2d(p=0.05),
            Conv2d(128, 256, kernel_size=3, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
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
    """Implementation of Bottleneck for DPN and VGG models for CIFAR10."""

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
            Conv2d(last_planes, in_planes, kernel_size=1, bias=False),
            BatchNorm2d(in_planes),
            ReLU(inplace=True),
            Conv2d(
                in_planes,
                in_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=32,
                bias=False,
            ),
            BatchNorm2d(in_planes),
            ReLU(inplace=True),
            Conv2d(in_planes, out_planes + dense_depth, kernel_size=1, bias=False),
            BatchNorm2d(out_planes + dense_depth),
        )
        self.shortcut: Sequential = Sequential()
        if first_layer:
            self.shortcut = Sequential(
                Conv2d(
                    last_planes,
                    out_planes + dense_depth,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
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
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
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
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
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


class VGG19(TorchModel):
    """Implementation of VGG19 for CIFAR10."""

    def __init__(self):
        """Constructor"""
        super(VGG19, self).__init__()
        self.features: Sequential = self._make_layers(
            [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                512,
                "M",
            ]
        )
        self.classifier: Linear = Linear(512, 10)

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation through the network.

        Args:
            x (Tensor): The input image.

        Returns:
            Tensor: The output tensor after forward propagation.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    @staticmethod
    def _make_layers(cfg: List[Any]) -> Sequential:
        """Helper method for creating layers.

        Args:
            cfg (List[Any]): configuration used for creating every layer.

        Returns:
            Sequential: Sequential model by PyTorch.
        """
        layers: List[Any] = list()
        in_channels: int = 3
        for x in cfg:
            if x == "M":
                layers += [MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    Conv2d(in_channels, x, kernel_size=3, padding=1),
                    BatchNorm2d(x),
                    ReLU(inplace=True),
                ]
                in_channels = x
        layers += [AvgPool2d(kernel_size=1, stride=1)]
        return Sequential(*layers)


class ResNetBasicBlock(TorchModel):
    """Implementation of BasicBlock of the ResNet model variants for CIFAR10."""

    expansion: int = 1

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1) -> None:
        """Constructor

        Args:
            in_planes (int): incoming planes for a layer.
            out_planes (int): outgoing planes for a layer.
            stride (int, optional): stride for the convolutional layers. Defaults to 1.
        """
        super(ResNetBasicBlock, self).__init__()
        self.conv_model: Sequential = Sequential(
            Conv2d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(out_planes),
            ReLU(inplace=True),
            Conv2d(
                out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
            ),
            BatchNorm2d(out_planes),
        )
        self.shortcut: Sequential = Sequential()
        if (stride != 1) or (in_planes != (self.expansion * out_planes)):
            self.shortcut = Sequential(
                Conv2d(
                    in_planes,
                    (self.expansion * out_planes),
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(self.expansion * out_planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation through the network.

        Args:
            x (Tensor): The input image.

        Returns:
            Tensor: The output tensor after forward propagation.
        """
        out: Tensor = self.conv_model(x)
        out += self.shortcut(x)
        return relu(out)


class ResNetBottleneck(TorchModel):
    """Implementation of Bottleneck of the ResNet model variants for CIFAR10."""

    expansion: int = 4

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1) -> None:
        """Constructor

        Args:
            in_planes (int): incoming planes for a layer.
            out_planes (int): outgoing planes for a layer.
            stride (int, optional): stride for the convolutional layers. Defaults to 1.
        """
        super(ResNetBottleneck, self).__init__()
        self.conv_model: Sequential = Sequential(
            Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            BatchNorm2d(out_planes),
            ReLU(inplace=True),
            Conv2d(
                out_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(out_planes),
            ReLU(inplace=True),
            Conv2d(
                out_planes, (self.expansion * out_planes), kernel_size=1, bias=False
            ),
            BatchNorm2d(self.expansion * out_planes),
        )
        self.shortcut: Sequential = Sequential()
        if (stride != 1) or (in_planes != (self.expansion * out_planes)):
            self.shortcut = Sequential(
                Conv2d(
                    in_planes,
                    (self.expansion * out_planes),
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm2d(self.expansion * out_planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation through the network.

        Args:
            x (Tensor): The input image.

        Returns:
            Tensor: The output tensor after forward propagation.
        """
        out: Tensor = self.conv_model(x)
        out += self.shortcut(x)
        return relu(out)


class ResNet(TorchModel):
    """Implementation of the ResNet base class for CIFAR10."""

    def __init__(
        self,
        block: Union[Type[ResNetBasicBlock], Type[ResNetBottleneck]],
        num_blocks: List[int],
        num_classes: int = 10,
    ) -> None:
        """Constructor

        Args:
            block (Union[ResNetBasicBlock, ResNetBottleneck]): block type used for the construction of ResNet
            num_blocks (List[int]): number of blocks for evert layer
            num_classes (int, optional): number of classes for prediction. Defaults to 10.
        """
        super(ResNet, self).__init__()
        self.in_planes: int = 64
        self.conv_model: Sequential = Sequential(
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            ReLU(inplace=True),
        )
        self.layer1: Sequential = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2: Sequential = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3: Sequential = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4: Sequential = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear: Linear = Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Union[Type[ResNetBasicBlock], Type[ResNetBottleneck]],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> Sequential:
        """Helper method used for the creation of layers.

        Args:
            block (Union[ResNetBasicBlock, ResNetBottleneck]): block type.
            planes (int): number of planes in the layer.
            num_blocks (int): number of blocks in the layer.
            stride (int): strides used in the layer.

        Returns:
            Sequential: Sequential model by PyTorch.
        """
        strides: List[int] = [stride] + [1] * (num_blocks - 1)
        layers: List[Union[Type[ResNetBasicBlock], Type[ResNetBottleneck]]] = list()
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
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
        x = self.linear(x)
        return x


def ResNet18() -> ResNet:  # noqa: N802
    """ResNet18 for CIFAR10.

    Returns:
        ResNet: Base ResNet implementation for CIFAR10.
    """
    return ResNet(ResNetBasicBlock, [2, 2, 2, 2])


def ResNet34() -> ResNet:  # noqa: N802
    """ResNet34 for CIFAR10.

    Returns:
        ResNet: Base ResNet implementation for CIFAR10.
    """
    return ResNet(ResNetBasicBlock, [3, 4, 6, 3])


def ResNet50() -> ResNet:  # noqa: N802
    """ResNet50 for CIFAR10.

    Returns:
        ResNet: Base ResNet implementation for CIFAR10.
    """
    return ResNet(ResNetBottleneck, [3, 4, 6, 3])


def ResNet101() -> ResNet:  # noqa: N802
    """ResNet101 for CIFAR10.

    Returns:
        ResNet: Base ResNet implementation for CIFAR10.
    """
    return ResNet(ResNetBottleneck, [3, 4, 23, 3])


def ResNet152() -> ResNet:  # noqa: N802
    """ResNet152 for CIFAR10.

    Returns:
        ResNet: Base ResNet implementation for CIFAR10.
    """
    return ResNet(ResNetBottleneck, [3, 8, 36, 3])
