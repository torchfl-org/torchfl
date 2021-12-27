#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the general LeNet architecture using PyTorch."""

import torch.nn as nn
from torchfl.models.sota.mlp import LinearBlock
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME


class LeNet(nn.Module):
    def __init__(
        self, num_classes=10, num_channels=1, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            num_channels (int, optional): Number of channels for the images in the dataset. Defaults to 3.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(LeNet, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="lenet",
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
            hidden_conv_dim=[10, 20],
            hidden_linear_dim=[320, 50],
        )
        self._create_network()

    def _create_network(self):
        self.input_net = nn.Sequential(
            nn.Conv2d(self.hparams.num_channels, 6, kernel_size=5, stride=1, padding=2),
            self.hparams.act_fn(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.conv_net = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            self.hparams.act_fn(),
            nn.Flatten(start_dim=1),
            LinearBlock(120, 84, self.hparams.act_fn, False),
        )
        self.output_net = nn.Sequential(nn.Linear(84, self.hparams.num_classes))

    def forward(self, x):
        return self.output_net(self.conv_net(self.input_net(x)))
