#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the pre-trained ShuffleNetv2 architectures using PyTorch and torchvision.

Contains:
    - ShuffleNetv2_x0_5
    - ShuffleNetv2_x1_0
    - ShuffleNetv2_x1_5
    - ShuffleNetv2_x2_0
"""

from torchvision import models
from types import SimpleNamespace
import torch.nn as nn
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME


class ShuffleNetv2_x0_5(models.ShuffleNetV2):
    """ShuffleNetv2_x0_5 base definition"""

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
        super(ShuffleNetv2_x0_5, self).__init__(
            stages_repeats=[4, 8, 4], stages_out_channels=[24, 48, 96, 192, 1024]
        )
        self.hparams = SimpleNamespace(
            model_name="shufflenet_v2_x0_5",
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
            pretrained_model = models.shufflenet_v2_x0_5(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.conv1 = nn.Sequential(
                nn.Conv2d(num_channels, 24, 3, 2, 1, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ShuffleNetv2_x1_0(models.ShuffleNetV2):
    """ShuffleNetv2_x1_0 base definition"""

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
        super(ShuffleNetv2_x1_0, self).__init__(
            stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024]
        )
        self.hparams = SimpleNamespace(
            model_name="shufflenet_v2_x1_0",
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
            pretrained_model = models.shufflenet_v2_x1_0(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.conv1 = nn.Sequential(
                nn.Conv2d(num_channels, 24, 3, 2, 1, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ShuffleNetv2_x1_5(models.ShuffleNetV2):
    """ShuffleNetv2_x1_5 base definition"""

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

        Raise:
            - NotImplementedError: pretrained models not implemented.
        """
        super(ShuffleNetv2_x1_5, self).__init__(
            stages_repeats=[4, 8, 4], stages_out_channels=[24, 176, 352, 704, 1024]
        )
        self.hparams = SimpleNamespace(
            model_name="shufflenet_v2_x1_5",
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
            raise NotImplementedError(
                "pretrained shufflenetv2_x2.0 is not supported as of now"
            )

        if num_channels != 3:
            self.conv1 = nn.Sequential(
                nn.Conv2d(num_channels, 24, 3, 2, 1, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)


class ShuffleNetv2_x2_0(models.ShuffleNetV2):
    """ShuffleNetv2_x2_0 base definition"""

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

        Raise:
            - NotImplementedError: pretrained models not implemented.
        """
        super(ShuffleNetv2_x2_0, self).__init__(
            stages_repeats=[4, 8, 4], stages_out_channels=[24, 244, 488, 976, 2048]
        )
        self.hparams = SimpleNamespace(
            model_name="shufflenet_v2_x2_0",
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
            raise NotImplementedError(
                "pretrained shufflenetv2_x2.0 is not supported as of now"
            )

        if num_channels != 3:
            self.conv1 = nn.Sequential(
                nn.Conv2d(num_channels, 24, 3, 2, 1, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
            )

        in_features = self.fc.in_features
        self.fc = nn.Linear(in_features, self.hparams.num_classes)
