#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Initializes the PyTorch models for EMNIST dataset."""

from torch.functional import Tensor
from torchfl.models.abstract import TorchModel
from torch.nn import (
    Sequential,
    Linear,
    ReLU,
    Dropout,
    Conv2d,
    BatchNorm2d,
    MaxPool2d,
    Dropout2d,
)


class MLP(TorchModel):
    """Implementation of MLP for EMNIST Digits."""

    def __init__(self) -> None:
        """Constructor"""
        super(MLP, self).__init__()
        self.model: Sequential = Sequential(
            Linear(28 * 28, 512), ReLU(), Dropout(), Linear(512, 10)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation through the network.

        Args:
            x (Tensor): The input image.

        Returns:
            Tensor: The output tensor after forward propagation.
        """
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        return self.model(x)


class CNN(TorchModel):
    """Implementation of CNN for EMNIST Digits."""

    def __init__(self) -> None:
        """Constructor"""
        super(CNN, self).__init__()
        self.conv_model: Sequential = Sequential(
            Conv2d(1, 32, kernel_size=5),
            BatchNorm2d(32),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=5, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout2d(p=0.05),
        )
        self.linear_model: Sequential = Sequential(
            Dropout(p=0.1),
            Linear(1600, 1024),
            ReLU(inplace=True),
            Linear(1024, 512),
            ReLU(inplace=True),
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
