#!/usr/bin/env python

"""Contains the LeNet model implementations for CIFAR100 dataset."""

from torchfl.models.sota.lenet import LeNet as BaseLeNet


class LeNet(BaseLeNet):
    def __init__(self, num_channels=3) -> None:
        """Constructor

        Args:
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
        """
        super().__init__(
            num_classes=100, num_channels=num_channels, act_fn_name="tanh"
        )
