#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the pre-trained AlexNet using PyTorch and torchvision."""

from torchvision import models
from types import SimpleNamespace
import torch.nn as nn


class AlexNet(models.alexnet(pretrained=True, progress=True)):
    def __init__(
        self, feature_extract=True, num_classes=10, act_fn_name="relu", **kwargs
    ) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Only trains the sequential layers of the pre-trained model. If False, the entire model is finetuned. Defaults to True.
            num_classes (int, optional): Number of classification outputs. Defaults to 10.
            act_fn_name (str, optional): Activation function to be used. Defaults to "relu". Accepted: ["tanh", "relu", "leakyrelu", "gelu"].
        """
        super(AlexNet, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="alexnet",
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

        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(in_features, self.hparams.num_classes)
