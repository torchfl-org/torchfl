#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the pre-trained MobileNet architectures using PyTorch and torchvision.

Contains:
    - MobileNetV2
    - MobileNetV3Small
    - MobileNetV3Large
"""

from torchvision import models
from types import SimpleNamespace
import torch.nn as nn
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME
from torchvision.ops.misc import ConvNormActivation
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf
from functools import partial


class MobileNetV2(models.mobilenet.MobileNetV2):
    """MobileNetV2 base definition"""

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
        super(MobileNetV2, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="mobilenet_v2",
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
            pretrained_model = models.mobilenet.mobilenet_v2(
                pretrained=True, progress=True
            )
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.features[0] = ConvNormActivation(
                num_channels,
                _make_divisible(32 * 1.0, 8),
                stride=2,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU6,
            )

        self.classifier[-1] = nn.Linear(self.last_channel, self.hparams.num_classes)


class MobileNetV3Small(models.mobilenet.MobileNetV3):
    """MobileNetV3Small base definition"""

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
        inverted_residual_setting, last_channel = _mobilenet_v3_conf(
            "mobilenet_v3_small"
        )
        super(MobileNetV3Small, self).__init__(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
        )
        self.hparams = SimpleNamespace(
            model_name="mobilenet_v3_small",
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
            pretrained_model = models.mobilenet.mobilenet_v3_small(
                pretrained=pre_trained, progress=True
            )
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False
        if num_channels != 3:
            self.features[0] = ConvNormActivation(
                num_channels,
                inverted_residual_setting[0].input_channels,
                kernel_size=3,
                stride=2,
                norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                activation_layer=nn.Hardswish,
            )

        in_features = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_features, self.hparams.num_classes)


class MobileNetV3Large(models.mobilenet.MobileNetV3):
    """MobileNetV3Large base definition"""

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
        inverted_residual_setting, last_channel = _mobilenet_v3_conf(
            "mobilenet_v3_large"
        )
        super(MobileNetV3Large, self).__init__(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
        )
        self.hparams = SimpleNamespace(
            model_name="mobilenet_v3_large",
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
            pretrained_model = models.mobilenet.mobilenet_v3_large(
                pretrained=pre_trained, progress=True
            )
            self.load_state_dict(pretrained_model.state_dict())
            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False
        if num_channels != 3:
            self.features[0] = ConvNormActivation(
                num_channels,
                inverted_residual_setting[0].input_channels,
                kernel_size=3,
                stride=2,
                norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                activation_layer=nn.Hardswish,
            )

        in_features = self.classifier[-1].in_features
        self.classifier[-1] = nn.Linear(in_features, self.hparams.num_classes)
