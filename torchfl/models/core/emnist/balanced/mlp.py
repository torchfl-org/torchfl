#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the MLP model implementations for EMNIST (balanced) dataset."""

from torchfl.models.sota.mlp import MLP as BaseMLP  # type: ignore[attr-defined]


class MLP(BaseMLP):
    def __init__(self, num_channels=1, img_w=28, img_h=28) -> None:
        """Constructor

        Args:
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
            - img_w (int, optional): Width of the input image. Defaults to 28.
            - img_h (int, optional): Height of the input image. Defaults to 28.
        """
        super(MLP, self).__init__(
            num_classes=47,
            num_channels=num_channels,
            img_w=img_w,
            img_h=img_h,
            hidden_dims=[256, 128],
        )
