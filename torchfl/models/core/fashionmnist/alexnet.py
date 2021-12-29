#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the AlexNet model implementations for FashionMNIST dataset."""

from torchfl.models.sota.alexnet import AlexNet as BaseAlexNet


class AlexNet(BaseAlexNet):
    def __init__(self, feature_extract=False) -> None:
        """Constructor

        Args:
            feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
        """
        super(AlexNet, self).__init__(
            feature_extract=feature_extract, num_classes=10, act_fn_name="relu"
        )
