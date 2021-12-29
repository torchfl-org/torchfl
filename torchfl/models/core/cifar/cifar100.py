#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the model implementations for CIFAR100 dataset."""

from torchfl.models.sota.densenet import DenseNet as BaseDensenet
from torchfl.models.sota.resnet import ResNet as BaseResNet


class DenseNet(BaseDensenet):
    def __init__(self) -> None:
        """Constructor"""
        super(DenseNet, self).__init__(num_classes=100, num_channels=3)


class ResNet18(BaseResNet):
    def __init__(self) -> None:
        """Constructor"""
        super(ResNet18, self).__init__(
            num_classes=100,
            num_channels=3,
            num_blocks=[2, 2, 2, 2],
            c_hidden=[16, 32, 64, 128],
            act_fn_name="relu",
            block_name="ResNetBlock",
        )


class ResNet34(BaseResNet):
    def __init__(self) -> None:
        """Constructor"""
        super(ResNet34, self).__init__(
            num_classes=100,
            num_channels=3,
            num_blocks=[3, 4, 6, 3],
            c_hidden=[16, 32, 64, 128],
            act_fn_name="relu",
            block_name="ResNetBlock",
        )


class ResNet50(BaseResNet):
    def __init__(self) -> None:
        """Constructor"""
        super(ResNet50, self).__init__(
            num_classes=100,
            num_channels=3,
            num_blocks=[3, 4, 6, 3],
            c_hidden=[16, 32, 64, 128],
            act_fn_name="relu",
            block_name="PreActResNetBlock",
        )


class ResNet101(BaseResNet):
    def __init__(self) -> None:
        """Constructor"""
        super(ResNet50, self).__init__(
            num_classes=10,
            num_channels=3,
            num_blocks=[3, 4, 23, 3],
            c_hidden=[16, 32, 64, 128],
            act_fn_name="relu",
            block_name="PreActResNetBlock",
        )


class ResNet152(BaseResNet):
    def __init__(self) -> None:
        """Constructor"""
        super(ResNet50, self).__init__(
            num_classes=100,
            num_channels=3,
            num_blocks=[3, 4, 36, 3],
            c_hidden=[16, 32, 64, 128],
            act_fn_name="relu",
            block_name="PreActResNetBlock",
        )
