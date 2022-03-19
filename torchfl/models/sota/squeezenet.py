#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the pre-trained SqueezeNet architectures using PyTorch and torchvision.

Contains:
    - SqueezeNet1_0
    - SqueezeNet1_1
"""

from torchvision import models
from types import SimpleNamespace
import torch.nn as nn
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME
from torchvision.models.squeezenet import Fire


class SqueezeNet1_0(models.SqueezeNet):
    """SqueezeNet1_0 base definition"""

    def __init__(
        self,
        pre_trained=True,
        feature_extract=True,
        num_classes=10,
        num_channels=3,
        act_fn_name="relu",
        **kwargs
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            - num_classes (int, optional): Number of classification outputs. Defaults to 10.
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
            - act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(SqueezeNet1_0, self).__init__(version="1_0")
        self.hparams = SimpleNamespace(
            model_name="squeezenet1_0",
            pre_trained=pre_trained,
            feature_extract=bool(pre_trained and feature_extract),
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if pre_trained:
            pretrained_model = models.squeezenet1_0(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.features = nn.Sequential(
                nn.Conv2d(num_channels, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )

        self.classifier[1] = nn.Conv2d(
            512, self.hparams.num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        self.num_classes = self.hparams.num_classes


class SqueezeNet1_1(models.SqueezeNet):
    """SqueezeNet1_1 base definition"""

    def __init__(
        self,
        pre_trained=True,
        feature_extract=True,
        num_classes=10,
        num_channels=3,
        act_fn_name="relu",
        **kwargs
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            - num_classes (int, optional): Number of classification outputs. Defaults to 10.
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
            - act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(SqueezeNet1_1, self).__init__(version="1_1")
        self.hparams = SimpleNamespace(
            model_name="squeezenet1_1",
            pre_trained=pre_trained,
            feature_extract=bool(pre_trained and feature_extract),
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            num_channels=num_channels,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if pre_trained:
            pretrained_model = models.squeezenet1_1(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.features = nn.Sequential(
                nn.Conv2d(num_channels, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        self.classifier[1] = nn.Conv2d(
            512, self.hparams.num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        self.num_classes = self.hparams.num_classes
