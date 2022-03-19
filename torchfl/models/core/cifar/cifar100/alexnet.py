#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the AlexNet model implementations for CIFAR100 dataset."""

from torchfl.models.sota.alexnet import (  # type: ignore[attr-defined]
    AlexNet as BaseAlexNet,
)  # type: ignore[attr-defined]


class AlexNet(BaseAlexNet):
    def __init__(self, pre_trained=True, feature_extract=False, num_channels=3) -> None:
        """Constructor

        Args:
            - pre_trained (bool, optional): Use the model pre-trained on the ImageNet dataset. Defaults to True.
            - feature_extract (bool, optional): Use transfer learning and only train the classifier. Otherwise, finetune the whole model. Defaults to False.
            - num_channels (int, optional): Number of incoming channels. Defaults to 3.
        """
        super(AlexNet, self).__init__(
            pre_trained=pre_trained,
            feature_extract=feature_extract,
            num_channels=num_channels,
            num_classes=100,
            act_fn_name="relu",
        )
