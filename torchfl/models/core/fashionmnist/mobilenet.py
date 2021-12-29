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
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(MobileNetV2, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )


class MobileNetV3Small(BaseMobileNet.MobileNetV3Small):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(MobileNetV3Small, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )


class MobileNetV3Large(BaseMobileNet.MobileNetV3Large):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(MobileNetV3Large, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )
