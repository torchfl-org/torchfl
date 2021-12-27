#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the general AlexNet architecture using PyTorch."""

import torch.nn as nn
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME
from torchfl.models.sota.mlp import LinearBlock


class AlexNet(nn.Module):
    def __init__(
        self, num_classes=10, num_channels=1, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            num_channels (int, optional): Number of channels for the images in the dataset. Defaults to 3.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(CNN, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="alexnet",
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )
        self._create_network()

    def _create_network(self):
        self.input_net = nn.Sequential(
            nn.Conv2d(
                self.hparams.num_channels, 96, kernel_size=11, stride=4, padding=0
            ),
            self.hparams.act_fn(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv_net = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            self.hparams.act_fn(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            self.hparams.act_fn(),
            nn.Conv2d(384, 384, 3, 1, 1),
            self.hparams.act_fn(),
            nn.Conv2d(384, 256, 3, 1, 1),
            self.hparams.act_fn(),
            nn.MaxPool2d(3, 2),
            nn.Flatten(),
            LinearBlock(256 * 6 * 6, 4096, self.hparams.act_fn, True),
            LinearBlock(4096, 4096, self.hparams.act_fn, True),
        )
        self.output_net = nn.Sequential(nn.Linear(4096, self.hparams.num_classes))

    def forward(self, x):
        return self.output_net(self.conv_net(self.input_net(x)))
