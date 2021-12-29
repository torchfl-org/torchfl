#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the quantized ResNet model implementations for FashionMNIST dataset.

Contains:
    - ResNet18
    - ResNet50
    - ResNext101_32X8D
"""

import torchfl.models.sota.quantized.resnet as BaseResNet


class ResNet18(BaseResNet.ResNet18):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(ResNet18, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )


class ResNet50(BaseResNet.ResNet50):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(ResNet50, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )


class ResNext101_32X8D(BaseResNet.ResNext101_32X8D):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(ResNext101_32X8D, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )
