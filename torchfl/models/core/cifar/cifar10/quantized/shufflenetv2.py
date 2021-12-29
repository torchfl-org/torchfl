#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the quantized ShuffleNetv2 model implementations for CIFAR10 dataset.


Contains:
    - ShuffleNetv2_x0_5
    - ShuffleNetv2_x1_0
"""

import torchfl.models.sota.quantized.shufflenetv2 as BaseShuffleNetv2


class ShuffleNetv2_x0_5(BaseShuffleNetv2.ShuffleNetv2_x0_5):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(ShuffleNetv2_x0_5, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )


class ShuffleNetv2_x1_0(BaseShuffleNetv2.ShuffleNetv2_x1_0):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(ShuffleNetv2_x1_0, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )
