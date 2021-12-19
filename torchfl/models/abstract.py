#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Initializes the abstract PyTorch models used for training and testing."""

from torch.nn import Module
from abc import ABC, abstractmethod
from torch.functional import Tensor


class TorchModel(Module, ABC):
    """Abstract class used as the base for implementation of PyTorch models."""

    @abstractmethod
    def __init__(self):
        """Default constructor"""
        super(TorchModel, self).__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Abstract method used for forward propagating through the model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after forward propagation.
        """
        return x
