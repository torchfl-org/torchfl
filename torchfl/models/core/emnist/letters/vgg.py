#!/usr/bin/env python

"""Contains the VGG model implementations for EMNIST (letters) dataset.

Contains:
    - VGG11
    - VGG11_BN
    - VGG13
    - VGG13_BN
    - VGG16
    - VGG16_BN
    - VGG19
    - VGG19_BN
"""

from torchfl.models.sota.vgg import VGG11 as BaseVGG11
from torchfl.models.sota.vgg import VGG11_BN as BaseVGG11_BN
from torchfl.models.sota.vgg import VGG13 as BaseVGG13
from torchfl.models.sota.vgg import VGG13_BN as BaseVGG13_BN
from torchfl.models.sota.vgg import VGG16 as BaseVGG16
from torchfl.models.sota.vgg import VGG16_BN as BaseVGG16_BN
from torchfl.models.sota.vgg import VGG19 as BaseVGG19
from torchfl.models.sota.vgg import VGG19_BN as BaseVGG19_BN


class VGG11(BaseVGG11):
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
            num_classes=26,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG11_BN(BaseVGG11_BN):
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
            num_classes=26,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG13(BaseVGG13):
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
            num_classes=26,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG13_BN(BaseVGG13_BN):
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
            num_classes=26,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG16(BaseVGG16):
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
            num_classes=26,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG16_BN(BaseVGG16_BN):
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
            num_classes=26,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG19(BaseVGG19):
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
            num_classes=26,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG19_BN(BaseVGG19_BN):
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
            num_classes=26,
            num_channels=num_channels,
            act_fn_name="relu",
        )
