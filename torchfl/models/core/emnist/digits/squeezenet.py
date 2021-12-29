#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the SqueezeNet model implementations for EMNIST (digits) dataset.

Contains:
    - SqueezeNet1_0
    - SqueezeNet1_1
"""

import torchfl.models.sota.squeezenet as BaseSqueezeNet


class SqueezeNet1_0(BaseSqueezeNet.SqueezeNet1_0):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(SqueezeNet1_0, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )


class SqueezeNet1_1(BaseSqueezeNet.SqueezeNet1_1):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(SqueezeNet1_1, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )
