#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the SqueezeNet model implementations for EMNIST (balanced) dataset.

Contains:
    - SqueezeNet1_0
    - SqueezeNet1_1
"""

from torchfl.models.sota.squeezenet import (  # type: ignore[attr-defined]
    SqueezeNet1_0 as BaseSqueezeNet1_0,
    SqueezeNet1_1 as BaseSqueezeNet1_1,
)


class SqueezeNet1_0(BaseSqueezeNet1_0):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(SqueezeNet1_0, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=47,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class SqueezeNet1_1(BaseSqueezeNet1_1):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(SqueezeNet1_1, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=47,
            num_channels=num_channels,
            act_fn_name="relu",
        )
