#!/usr/bin/env python

"""Contains the MobileNet model implementations for EMNIST (balanced) dataset.

Contains:
    - MobileNetV2
    - MobileNetV3Small
    - MobileNetV3Large
"""

from torchfl.models.sota.mobilenet import MobileNetV2 as BaseMobileNetV2
from torchfl.models.sota.mobilenet import (
    MobileNetV3Large as BaseMobileNetV3Large,
)
from torchfl.models.sota.mobilenet import (
    MobileNetV3Small as BaseMobileNetV3Small,
)


class MobileNetV2(BaseMobileNetV2):
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
            num_classes=62,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class MobileNetV3Small(BaseMobileNetV3Small):
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
            num_classes=62,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class MobileNetV3Large(BaseMobileNetV3Large):
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
            num_classes=62,
            num_channels=num_channels,
            act_fn_name="relu",
        )
