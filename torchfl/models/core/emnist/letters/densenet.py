#!/usr/bin/env python

"""Contains the DenseNet model implementations for EMNIST (letters) dataset.

Contains:
    - DenseNet121
    - DenseNet161
    - DenseNet169
    - DenseNet201
"""

from torchfl.models.sota.densenet import DenseNet121 as BaseDenseNet121
from torchfl.models.sota.densenet import DenseNet161 as BaseDenseNet161
from torchfl.models.sota.densenet import DenseNet169 as BaseDenseNet169
from torchfl.models.sota.densenet import DenseNet201 as BaseDenseNet201


class DenseNet121(BaseDenseNet121):
    def __init__(
        self, pre_trained=True, feature_extract=False, num_channels=1
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super().__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=26,
            act_fn_name="relu",
        )


class DenseNet161(BaseDenseNet161):
    def __init__(
        self, pre_trained=True, feature_extract=False, num_channels=1
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super().__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=26,
            act_fn_name="relu",
        )


class DenseNet169(BaseDenseNet169):
    def __init__(
        self, pre_trained=True, feature_extract=False, num_channels=1
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super().__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=26,
            act_fn_name="relu",
        )


class DenseNet201(BaseDenseNet201):
    def __init__(
        self, pre_trained=True, feature_extract=False, num_channels=1
    ) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super().__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=26,
            act_fn_name="relu",
        )
