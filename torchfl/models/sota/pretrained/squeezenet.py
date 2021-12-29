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


class SqueezeNet1_0(models.squeezenet1_0(pretrained=True, progress=True)):
    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(SqueezeNet1_0, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="squeezenet1_0",
            pre_trained=True,
            feature_extract=feature_extract,
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if feature_extract:
            for param in self.parameters():
                param.requires_grad = False

        self.classifier[1] = nn.Conv2d(
            512, self.hparams.num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        self.num_classes = self.hparams.num_classes


class SqueezeNet1_1(models.squeezenet1_1(pretrained=True, progress=True)):
    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(SqueezeNet1_1, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="squeezenet1_1",
            pre_trained=True,
            feature_extract=feature_extract,
            finetune=bool(not feature_extract),
            quantized=False,
            num_classes=num_classes,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if feature_extract:
            for param in self.parameters():
                param.requires_grad = False

        self.classifier[1] = nn.Conv2d(
            512, self.hparams.num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        self.num_classes = self.hparams.num_classes
