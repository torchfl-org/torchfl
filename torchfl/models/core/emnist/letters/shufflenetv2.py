#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the ShuffleNetv2 model implementations for EMNIST (letters) dataset.

Contains:
    - ShuffleNetv2_x0_5
    - ShuffleNetv2_x1_0
    - ShuffleNetv2_x1_5
    - ShuffleNetv2_x2_0
"""

from torchfl.models.sota.shufflenetv2 import (  # type: ignore[attr-defined]
    ShuffleNetv2_x0_5 as BaseShuffleNetv2_x0_5,
    ShuffleNetv2_x1_0 as BaseShuffleNetv2_x1_0,
    ShuffleNetv2_x1_5 as BaseShuffleNetv2_x1_5,
    ShuffleNetv2_x2_0 as BaseShuffleNetv2_x2_0,
)


class ShuffleNetv2_x0_5(BaseShuffleNetv2_x0_5):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(ShuffleNetv2_x0_5, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=26,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class ShuffleNetv2_x1_0(BaseShuffleNetv2_x1_0):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(ShuffleNetv2_x1_0, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=26,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class ShuffleNetv2_x1_5(BaseShuffleNetv2_x1_5):
    def __init__(
        self, pre_trained=False, feature_extract=False, num_channels=1
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to False.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(ShuffleNetv2_x1_5, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=26,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class ShuffleNetv2_x2_0(BaseShuffleNetv2_x2_0):
    def __init__(
        self, pre_trained=False, feature_extract=False, num_channels=1
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to False.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(ShuffleNetv2_x2_0, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=26,
            num_channels=num_channels,
            act_fn_name="relu",
        )
