#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the pre-trained, quantized ResNet architectures using PyTorch and torchvision.

Contains:
    - ResNet18
    - ResNet50
    - ResNext101_32X8D
"""

from torchvision import models
from types import SimpleNamespace
import torch.nn as nn
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME


class ResNet18(
    models.quantization.resnet18(pretrained=True, progress=True, quantize=True)
):
    """Quantized ResNet18 base definition"""

    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNet18, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="resnet18",
            pre_trained=True,
            feature_extract=feature_extract,
            finetune=bool(not feature_extract),
            quantized=True,
            num_classes=num_classes,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if feature_extract:
            for param in self.parameters():
                param.requires_grad = False

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNet50(
    models.quantization.resnet50(pretrained=True, progress=True, quantize=True)
):
    """Quantized ResNet50 base definition"""

    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNet50, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="resnet50",
            pre_trained=True,
            feature_extract=feature_extract,
            finetune=bool(not feature_extract),
            quantized=True,
            num_classes=num_classes,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if feature_extract:
            for param in self.parameters():
                param.requires_grad = False

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNext101_32X8D(
    models.quantization.resnext101_32x8d(pretrained=True, progress=True, quantize=True)
):
    """Quantized ResNet101_32X8D base definition"""

    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNext101_32X8D, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="resnext101_32x8d",
            pre_trained=True,
            feature_extract=feature_extract,
            finetune=bool(not feature_extract),
            quantized=True,
            num_classes=num_classes,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if feature_extract:
            for param in self.parameters():
                param.requires_grad = False

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)
