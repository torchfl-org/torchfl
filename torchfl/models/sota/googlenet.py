#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the general GoogleNet architecture using PyTorch.

Contains:
    - InceptionBlock
    - GoogleNet
"""

import torch.nn as nn
import torch
from types import SimpleNamespace
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME


class InceptionBlock(nn.Module):
    def __init__(self, c_in, c_red: dict, c_out: dict, act_fn):
        """Constructor

        Args:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
            nn.BatchNorm2d(c_out["1x1"]),
            act_fn(),
        )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
            nn.BatchNorm2d(c_red["3x3"]),
            act_fn(),
            nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out["3x3"]),
            act_fn(),
        )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
            nn.BatchNorm2d(c_red["5x5"]),
            act_fn(),
            nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out["5x5"]),
            act_fn(),
        )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn(),
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out


class GoogleNet(nn.Module):
    def __init__(self, num_classes=10, num_channels=3, act_fn_name="relu", **kwargs):
        """Constructor

        Args:
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            num_channels (int, optional): Number of channels for the images in the dataset. Defaults to 3.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super().__init__()
        self.hparams = SimpleNamespace(
            model_name="googlenet",
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
            pre_trained=False,
            feature_extract=False,
            finetune=False,
            quantized=False,
        )
        self._create_network()
        self._init_params()

    def _create_network(self):
        # A first convolution on the original image to scale up the channel size
        self.input_net = nn.Sequential(
            nn.Conv2d(self.hparams.num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            self.hparams.act_fn(),
        )
        # Stacking inception blocks
        self.inception_blocks = nn.Sequential(
            InceptionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                64,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.hparams.act_fn,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),  # 32x32 => 16x16
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                96,
                c_red={"3x3": 32, "5x5": 16},
                c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24},
                act_fn=self.hparams.act_fn,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),  # 16x16 => 8x8
            InceptionBlock(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
            InceptionBlock(
                128,
                c_red={"3x3": 48, "5x5": 16},
                c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
                act_fn=self.hparams.act_fn,
            ),
        )
        # Mapping to classification output
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, self.hparams.num_classes),
        )

    def _init_params(self):
        # initialize the convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.inception_blocks(x)
        x = self.output_net(x)
        return x
