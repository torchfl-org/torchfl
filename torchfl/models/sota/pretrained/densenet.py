#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the pre-trained DenseNet architectures using PyTorch and torchvision.

Contains:
    - DenseNet121
    - DenseNet161
    - DenseNet169
    - DenseNet201
"""

from torchvision import models
from types import SimpleNamespace
import torch.nn as nn
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME


class DenseNet121(models.densenet121(pretrained=True, progress=True)):
    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(DenseNet121, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="densenet121",
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

        in_features = self.classifier.in_features
        self.classifier = nn.Linear(in_features, self.hparams.num_classes)


class DenseNet169(models.densenet169(pretrained=True, progress=True)):
    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(DenseNet169, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="densenet169",
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

        in_features = self.classifier.in_features
        self.classifier = nn.Linear(in_features, self.hparams.num_classes)


class DenseNet121(models.densenet121(pretrained=True, progress=True)):
    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(DenseNet121, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="densenet121",
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

        in_features = self.classifier.in_features
        self.classifier = nn.Linear(in_features, self.hparams.num_classes)


class DenseNet201(models.densenet201(pretrained=True, progress=True)):
    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(DenseNet201, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="densenet201",
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

        in_features = self.classifier.in_features
        self.classifier = nn.Linear(in_features, self.hparams.num_classes)
