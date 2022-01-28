#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the MobileNet model implementations for FashionMNIST dataset.

Contains:
    - MobileNetV2
    - MobileNetV3Small
    - MobileNetV3Large
"""

import torchfl.models.sota.mobilenet as BaseMobileNet


class MobileNetV2(BaseMobileNet.MobileNetV2):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=3) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(MobileNetV2, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=100,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class MobileNetV3Small(BaseMobileNet.MobileNetV3Small):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=3) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(MobileNetV3Small, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=100,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class MobileNetV3Large(BaseMobileNet.MobileNetV3Large):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=3) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(MobileNetV3Large, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=100,
            num_channels=num_channels,
            act_fn_name="relu",
        )
