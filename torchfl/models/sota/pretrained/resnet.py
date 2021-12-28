#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the pre-trained ResNet architectures using PyTorch and torchvision.

Contains:
    - ResNet18
    - ResNet34
    - ResNet50
    - ResNet101
    - ResNet152
    - ResNext50_32X4D
    - ResNext101_32X8D
    - WideResNet50_2
    - WideResNet101_2
"""

from torchvision import models
from types import SimpleNamespace
import torch.nn as nn


class ResNet18(models.resnet18(pretrained=True, progress=True)):
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
            quantized=False,
            num_classes=num_classes,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if feature_extract:
            for param in self.parameters():
                param.requires_grad = False

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNet34(models.resnet34(pretrained=True, progress=True)):
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
            model_name="resnet34",
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

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNet50(models.resnet50(pretrained=True, progress=True)):
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
            quantized=False,
            num_classes=num_classes,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if feature_extract:
            for param in self.parameters():
                param.requires_grad = False

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNet101(models.resnet101(pretrained=True, progress=True)):
    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNet101, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="resnet101",
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

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNet152(models.resnet152(pretrained=True, progress=True)):
    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNet152, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="resnet152",
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

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNext50_32X4D(models.resnext50_32x4d(pretrained=True, progress=True)):
    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(ResNext50_32X4D, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="resnext50_32x4d",
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

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ResNext101_32X8D(models.resnext101_32x8d(pretrained=True, progress=True)):
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
            quantized=False,
            num_classes=num_classes,
            act_fn_name=act_fn_name,
            act_fn=ACTIVATION_FUNCTIONS_BY_NAME[act_fn_name],
        )

        if feature_extract:
            for param in self.parameters():
                param.requires_grad = False

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class WideResNet50_2(models.wide_resnet50_2(pretrained=True, progress=True)):
    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(WideResNet50_2, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="wide_resnet50_2",
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

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class WideResNet101_2(models.wide_resnet101_2(pretrained=True, progress=True)):
    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(WideResNet101_2, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="wide_resnet101_2",
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

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)
