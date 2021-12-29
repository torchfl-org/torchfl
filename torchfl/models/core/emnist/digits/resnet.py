#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the ResNet model implementations for EMNIST (digits) dataset.

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

import torchfl.models.sota.resnet as BaseResNet


class ResNet18(BaseResNet.ResNet18):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(ResNet18, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )


class ResNet34(BaseResNet.ResNet34):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(ResNet34, self).__init__(
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


class ResNet101(BaseResNet.ResNet101):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(ResNet101, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )


class ResNet152(BaseResNet.ResNet152):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(ResNet152, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )


class ResNext50_32X4D(BaseResNet.ResNext50_32X4D):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(ResNext50_32X4D, self).__init__(
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


class WideResNet50_2(BaseResNet.WideResNet50_2):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(WideResNet50_2, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )


class WideResNet101_2(BaseResNet.WideResNet101_2):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(WideResNet101_2, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )
