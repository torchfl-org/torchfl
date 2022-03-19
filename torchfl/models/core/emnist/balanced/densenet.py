#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the DenseNet model implementations for EMNIST (balanced) dataset.

Contains:
    - DenseNet121
    - DenseNet161
    - DenseNet169
    - DenseNet201
"""

from torchfl.models.sota.densenet import (  # type: ignore[attr-defined]
    DenseNet121 as BaseDenseNet121,
    DenseNet161 as BaseDenseNet161,
    DenseNet169 as BaseDenseNet169,
    DenseNet201 as BaseDenseNet201,
)


class DenseNet121(BaseDenseNet121):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(DenseNet121, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=47,
            act_fn_name="relu",
        )


class DenseNet161(BaseDenseNet161):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(DenseNet161, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=47,
            act_fn_name="relu",
        )


class DenseNet169(BaseDenseNet169):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(DenseNet169, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=47,
            act_fn_name="relu",
        )


class DenseNet201(BaseDenseNet201):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(DenseNet201, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=47,
            act_fn_name="relu",
        )
