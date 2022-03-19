#!/usr/bin/env python
# -*- coding: utf-8 -*-
# type: ignore

"""Implementation of the pre-trained AlexNet using PyTorch and torchvision."""

from torchvision import models
from types import SimpleNamespace
import torch.nn as nn
from torchfl.compatibility import ACTIVATION_FUNCTIONS_BY_NAME


class AlexNet(models.AlexNet):
    """AlexNet base definition"""

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
        super(AlexNet, self).__init__()
        self.hparams = SimpleNamespace(
            model_name="alexnet",
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
            pretrained_model = models.alexnet(pretrained=True, progress=True)
            self.load_state_dict(pretrained_model.state_dict())

            if feature_extract:
                for param in self.parameters():
                    param.requires_grad = False

        if num_channels != 3:
            self.features[0] = nn.Conv2d(
                num_channels, 64, kernel_size=11, stride=4, padding=2
            )

        in_features = self.classifier[6].in_features
        self.classifier[6] = nn.Linear(in_features, self.hparams.num_classes)
