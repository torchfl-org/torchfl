#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the VGG model implementations for EMNIST (byclass) dataset.

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

import torchfl.models.sota.vgg as BaseVGG


class VGG11(BaseVGG.VGG11):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(VGG11, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=62,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG11_BN(BaseVGG.VGG11_BN):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(VGG11_BN, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=62,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG13(BaseVGG.VGG13):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(VGG13, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=62,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG13_BN(BaseVGG.VGG13_BN):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(VGG13_BN, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=62,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG16(BaseVGG.VGG16):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(VGG16, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=62,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG16_BN(BaseVGG.VGG16_BN):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(VGG16_BN, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=62,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG19(BaseVGG.VGG19):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(VGG19, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=62,
            num_channels=num_channels,
            act_fn_name="relu",
        )


class VGG19_BN(BaseVGG.VGG19_BN):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=1) -> None:
        """Constructor

        Args:
            pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            num_channels (int, optional): Number of incoming channels. Defaults to 1.
        """
        super(VGG19_BN, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_classes=62,
            num_channels=num_channels,
            act_fn_name="relu",
        )
