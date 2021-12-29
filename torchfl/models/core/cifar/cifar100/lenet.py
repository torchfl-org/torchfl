#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the LeNet model implementations for CIFAR100 dataset."""

from torchfl.models.sota.lenet import LeNet as BaseLeNet


class LeNet(BaseLeNet):
    def __init__(self) -> None:
        """Constructor"""
        super(LeNet, self).__init__(num_classes=100, num_channels=3, act_fn_name="tanh")
