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


class DenseNet121(models.DenseNet):
    """DenseNet121 base definition."""

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
        super(DenseNet121, self).__init__(
            growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64
        )
        self.hparams = SimpleNamespace(
            model_name="densenet121",
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
            pretrained_model = models.densenet121(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())

            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            out_channels = self.features[0].out_channels
            self.features[0] = nn.Conv2d(
                in_channels=num_channels,
                out_channels=out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        in_features = self.classifier.in_features
        self.classifier = nn.Linear(in_features, self.hparams.num_classes)


class DenseNet161(models.DenseNet):
    """DenseNet161 base definition."""

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
        super(DenseNet161, self).__init__(
            growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96
        )
        self.hparams = SimpleNamespace(
            model_name="densenet161",
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
            pretrained_model = models.densenet161(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())

            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            out_channels = self.features[0].out_channels
            self.features[0] = nn.Conv2d(
                in_channels=num_channels,
                out_channels=out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        in_features = self.classifier.in_features
        self.classifier = nn.Linear(in_features, self.hparams.num_classes)


class DenseNet169(models.DenseNet):
    """DenseNet169 base definition."""

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
        super(DenseNet169, self).__init__(
            growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64
        )
        self.hparams = SimpleNamespace(
            model_name="densenet169",
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
            pretrained_model = models.densenet169(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())

            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            out_channels = self.features[0].out_channels
            self.features[0] = nn.Conv2d(
                in_channels=num_channels,
                out_channels=out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        in_features = self.classifier.in_features
        self.classifier = nn.Linear(in_features, self.hparams.num_classes)


class DenseNet201(models.DenseNet):
    """DenseNet201 base definition."""

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
        super(DenseNet201, self).__init__(
            growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64
        )
        self.hparams = SimpleNamespace(
            model_name="densenet201",
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
            pretrained_model = models.densenet201(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())

            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            out_channels = self.features[0].out_channels
            self.features[0] = nn.Conv2d(
                in_channels=num_channels,
                out_channels=out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        in_features = self.classifier.in_features
        self.classifier = nn.Linear(in_features, self.hparams.num_classes)
