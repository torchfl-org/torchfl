#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Initializes the PyTorch models for MNIST dataset."""
from torch.functional import Tensor
from torchfl.models.abstract import TorchModel
from torch.nn import Sequential, Linear, ReLU, Dropout, Conv2d, MaxPool2d, Dropout2d


class MLP(TorchModel):
    """Implementation of MLP for MNIST."""

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
    """Implementation of CNN for MNIST."""

    def __init__(self) -> None:
        """Constructor"""
        super(CNN, self).__init__()
        self.conv_model: Sequential = Sequential(
            Conv2d(1, 10, kernel_size=5),
            MaxPool2d(2, 2),
            ReLU(inplace=True),
            Conv2d(10, 20, kernel_size=5),
            Dropout2d(),
            MaxPool2d(2, 2),
            ReLU(inplace=True),
        )
        self.linear_model: Sequential = Sequential(
            ReLU(), Linear(320, 50), Dropout(), Linear(50, 10)
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
