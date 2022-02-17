#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the DenseNet model implementations for EMNIST (letters) dataset.

Contains:
    - DenseNet121
    - DenseNet161
    - DenseNet169
    - DenseNet201
"""

import torchfl.models.sota.densenet as BaseDenseNet


class DenseNet121(BaseDenseNet.DenseNet121):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(DenseNet121, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=26,
            act_fn_name="relu",
        )


class DenseNet161(BaseDenseNet.DenseNet161):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(DenseNet161, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=26,
            act_fn_name="relu",
        )


class DenseNet169(BaseDenseNet.DenseNet169):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(DenseNet169, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=26,
            act_fn_name="relu",
        )


class DenseNet201(BaseDenseNet.DenseNet201):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(DenseNet201, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=26,
            act_fn_name="relu",
        )
