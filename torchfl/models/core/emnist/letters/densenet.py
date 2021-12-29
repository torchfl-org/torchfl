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
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(DenseNet121, self).__init__(
            feature_extract=feature_extract, num_classes=26, act_fn_name="relu"
        )


class DenseNet161(BaseDenseNet.DenseNet161):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(DenseNet161, self).__init__(
            feature_extract=feature_extract, num_classes=26, act_fn_name="relu"
        )


class DenseNet169(BaseDenseNet.DenseNet169):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(DenseNet169, self).__init__(
            feature_extract=feature_extract, num_classes=26, act_fn_name="relu"
        )


class DenseNet201(BaseDenseNet.DenseNet201):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(DenseNet201, self).__init__(
            feature_extract=feature_extract, num_classes=26, act_fn_name="relu"
        )
