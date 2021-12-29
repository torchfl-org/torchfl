#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the MLP model implementations for FashionMNIST dataset."""

from torchfl.models.sota.mlp import MLP as BaseMLP


class MLP(BaseMLP):
    def __init__(self) -> None:
        """Constructor"""
        super(MLP, self).__init__(
            num_classes=10, num_channels=1, img_w=28, img_h=28, hidden_dims=[256, 128]
        )
