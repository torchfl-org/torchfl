#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the ResNet model implementations for EMNIST (balanced) dataset.

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
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(ResNet18, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=47,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class ResNet34(BaseResNet.ResNet34):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(ResNet34, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=47,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class ResNet50(BaseResNet.ResNet50):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(ResNet50, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=47,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class ResNet101(BaseResNet.ResNet101):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(ResNet101, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=47,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class ResNet152(BaseResNet.ResNet152):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(ResNet152, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=47,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class ResNext50_32X4D(BaseResNet.ResNext50_32X4D):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(ResNext50_32X4D, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=47,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class ResNext101_32X8D(BaseResNet.ResNext101_32X8D):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(ResNext101_32X8D, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=47,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class WideResNet50_2(BaseResNet.WideResNet50_2):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(WideResNet50_2, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=47,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class WideResNet101_2(BaseResNet.WideResNet101_2):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(WideResNet101_2, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=47,
            num_channels=num_channels,
            act_fn_name="relu",
        )
