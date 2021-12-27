#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Contains the custom base model implementations for FashionMNIST dataset."""

import torch.nn as nn
from torchfl.models.sota.mlp import LinearBlock
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME


class CNN(nn.Module):
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
            model_name="cnn",
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
            nn.Conv2d(self.hparams.num_channels, 10, kernel_size=5),
            nn.MaxPool2d(2, 2),
            self.hparams.act_fn(),
        )
        self.conv_net = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2, 2),
            self.hparams.act_fn(),
            nn.Flatten(),
            LinearBlock(320, 50, self.hparams.act_fn, True),
        )
        self.output_net = nn.Sequential(nn.Linear(50, self.hparams.num_classes))

    def forward(self, x):
        return self.output_net(self.conv_net(self.input_net(x)))
