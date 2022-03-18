#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the general MLP architecture using PyTorch.

Contains:
    - LinearBlock
    - MLP
"""

import torch.nn as nn
from types import SimpleNamespace
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME


class LinearBlock(nn.Module):
    """LinearBlock base definition"""

    def __init__(self, c_in, c_out, act_fn, dropout=False):
        """Constructor

        Args:
            - c_in - Number of incoming channels.
            - c_out - Number of outgoing channels.
            - act_fn - Activation class constructor (e.g. nn.ReLU).
            - dropout - If the dropout needs to be added.
        """
        super().__init__()
        if dropout:
            self.linear_block = nn.Sequential(
                nn.Linear(c_in, c_out), act_fn(), nn.Dropout()
            )
        else:
            self.linear_block = nn.Sequential(nn.Linear(c_in, c_out), act_fn())

    def forward(self, x):
        """Forward propagation

        Args:
            - x (torch.Tensor): Input Tensor

        Returns:
            - torch.Tensor: Returns the tensor after forward propagation
        """
        return self.linear_block(x)


class MLP(nn.Module):
    """MLP base definition"""

    def __init__(
        self,
        num_classes=10,
        num_channels=1,
        img_w=28,
        img_h=28,
        hidden_dims=(256, 128),
        act_fn_name="relu",
        **kwargs
    ):
        """Constructor

        Args:
            - num_classes (int, optional): Number of classification outputs. Defaults to 10.
            - num_channels (int, optional): Number of channels for the images in the dataset. Defaults to 1.
            - img_w (int, optional): Width of the incoming image. Defaults to 28.
            - img_h (int, optional): Heigh of the incoming image. Defaults to 28.
            - hidden_dims (List[int], optional): Dimensions of the hidden layers. Defaults to [256, 128].
            - act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super().__init__()
        self.hparams = SimpleNamespace(
            model_name="mlp",
            num_classes=num_classes,
            num_channels=num_channels,
            img_w=img_w,
            img_h=img_h,
            hidden_dims=hidden_dims,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
            pre_trained=False,
            feature_extract=False,
            finetune=False,
            quantized=False,
        )
        self._create_network()

    def _create_network(self):
        # input layer
        self.input_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.hparams.num_channels * self.hparams.img_w * self.hparams.img_h,
                self.hparams.hidden_dims[0],
            ),
            self.hparams.act_fn(),
        )

        # hidden layers
        layers = list()
        for i in range(len(self.hparams.hidden_dims) - 1):
            layers.append(
                LinearBlock(
                    self.hparams.hidden_dims[i],
                    self.hparams.hidden_dims[i + 1],
                    self.hparams.act_fn,
                    False,
                )
            )
        self.hidden_net = nn.Sequential(*layers)

        # output layer
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.hidden_dims[-1], self.hparams.num_classes)
        )

    def forward(self, x):
        """Forward propagation

        Args:
            - x (torch.Tensor): Input Tensor

        Returns:
            - torch.Tensor: Returns the tensor after forward propagation
        """
        return self.output_net(self.hidden_net(self.input_net(x)))
