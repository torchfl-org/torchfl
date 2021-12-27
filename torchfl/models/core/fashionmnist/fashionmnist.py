#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Contains the model implementations for FashionMNIST dataset."""

from torchfl.models.sota.mlp import MLP as BaseMLP
from torchfl.models.core.fashionmnist.base import CNN as BaseCNN
from torchfl.models.sota.densenet import DenseNet as BaseDensenet
from torchfl.models.sota.googlenet import GoogleNet as BaseGoogleNet
from torchfl.models.sota.resnet import ResNet as BaseResNet


class MLP(BaseMLP):
    def __init__(self) -> None:
        """Constructor"""
        super(MLP, self).__init__(
            num_classes=10, num_channels=1, img_w=28, img_h=28, hidden_dims=[256, 128]
        )


class CNN(BaseCNN):
    def __init__(self) -> None:
        """Constructor"""
        super(CNN, self).__init__(num_classes=10, num_channels=1, act_fn_name="relu")


class DenseNet(BaseDensenet):
    def __init__(self) -> None:
        """Constructor"""
        super(DenseNet, self).__init__(num_classes=10, num_channels=1)


class GoogleNet(BaseGoogleNet):
    def __init__(self) -> None:
        """Constructor"""
        super(GoogleNet, self).__init__(num_classes=10, num_channels=1)


class ResNet18(BaseResNet):
    def __init__(self) -> None:
        """Constructor"""
        super(ResNet18, self).__init__(
            num_classes=10,
            num_channels=1,
            num_blocks=[2, 2, 2, 2],
            c_hidden=[16, 32, 64, 128],
            act_fn_name="relu",
            block_name="ResNetBlock",
        )


class ResNet34(BaseResNet):
    def __init__(self) -> None:
        """Constructor"""
        super(ResNet34, self).__init__(
            num_classes=10,
            num_channels=1,
            num_blocks=[3, 4, 6, 3],
            c_hidden=[16, 32, 64, 128],
            act_fn_name="relu",
            block_name="ResNetBlock",
        )


class ResNet50(BaseResNet):
    def __init__(self) -> None:
        """Constructor"""
        super(ResNet50, self).__init__(
            num_classes=10,
            num_channels=1,
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
            num_channels=1,
            num_blocks=[3, 4, 23, 3],
            c_hidden=[16, 32, 64, 128],
            act_fn_name="relu",
            block_name="PreActResNetBlock",
        )


class ResNet152(BaseResNet):
    def __init__(self) -> None:
        """Constructor"""
        super(ResNet50, self).__init__(
            num_classes=10,
            num_channels=1,
            num_blocks=[3, 4, 36, 3],
            c_hidden=[16, 32, 64, 128],
            act_fn_name="relu",
            block_name="PreActResNetBlock",
        )
