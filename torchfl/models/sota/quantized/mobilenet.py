#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the pre-trained MobileNet architectures using PyTorch and torchvision.

Contains:
    - MobileNetV2
    - MobileNetV3Large
"""

from torchvision import models
from types import SimpleNamespace
import torch.nn as nn
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME


class MobileNetV2(
    models.quantization.mobilenet.mobilenet_v2(
        pretrained=True, progress=True, quantize=True
    )
):
    """Quantized MobileNetV2 base definition"""

    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(MobileNetV2, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="mobilenet_v2",
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

        in_features = self.classifier[1].in_features
        self.classifier[1] = nn.Linear(in_features, self.hparams.num_classes)


class MobileNetV3Large(
    models.quantization.mobilenet.mobilenet_v3_large(
        pretrained=True, progress=True, quantize=True
    )
):
    """Quantized MobileNetV3Large base definition"""

    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(MobileNetV3Large, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="mobilenet_v3_large",
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

        in_features = self.classifier[1].in_features
        self.classifier[1] = nn.Linear(in_features, self.hparams.num_classes)
