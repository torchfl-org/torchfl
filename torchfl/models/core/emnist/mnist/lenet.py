#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the LeNet model implementations for MNIST dataset."""

from torchfl.models.sota.lenet import LeNet as BaseLeNet  # type: ignore[attr-defined]


class LeNet(BaseLeNet):
    def __init__(self, num_channels=1) -> None:
        """Constructor

        Args:
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(LeNet, self).__init__(
            num_classes=10, num_channels=num_channels, act_fn_name="tanh"
        )
